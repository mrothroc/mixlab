// Package main implements the mixlab command-line modes.
package train

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"slices"
	"strconv"
	"strings"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/data"
)

type GenerateOptions struct {
	ConfigPath         string
	SafetensorsLoad    string
	MaxTokens          int
	Temperature        float32
	TopK               int
	Prompt             string
	SequenceVocabulary string
	NumSamples         int
	GenerationBatch    int
	GenerationSeed     int64
	EOSTokenID         *int
	OutputPath         string
}

func runGenerate(configPath, safetensorsLoad string, maxTokens int, temperature float32, topK int, prompt string) error {
	return runGenerateWithOptions(GenerateOptions{
		ConfigPath: configPath, SafetensorsLoad: safetensorsLoad, MaxTokens: maxTokens,
		Temperature: temperature, TopK: topK, Prompt: prompt, NumSamples: 1,
	})
}

func runGenerateWithOptions(opts GenerateOptions) error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	if opts.ConfigPath == "" {
		return fmt.Errorf("-config is required for generate mode")
	}
	if opts.SafetensorsLoad == "" {
		return fmt.Errorf("-safetensors-load is required for generate mode")
	}
	cfg, err := LoadArchConfig(opts.ConfigPath)
	if err != nil {
		return err
	}
	vocab, err := loadSequenceVocabularyForConfig(opts.SequenceVocabulary, cfg)
	if err != nil {
		return err
	}
	plan, err := buildGenerationPlan(opts, cfg, vocab)
	if err != nil {
		return err
	}
	if countTTTMLPBlocks(cfg.Blocks) > 0 && plan.batchSize > 1 {
		return fmt.Errorf("-gen-batch=%d is not supported for TTT-MLP generation; use -gen-batch=1 until batched persistent inference state is available", plan.batchSize)
	}
	if err := configureMLXMemoryLimitsTo("generate", os.Stderr); err != nil {
		return err
	}
	output, err := openGenerationOutput(opts.OutputPath)
	if err != nil {
		return err
	}
	runErr := func() error {
		if countTTTMLPBlocks(cfg.Blocks) > 0 {
			if _, _, statefulErr := arch.BuildTTTMLPStatefulInferenceIRProgram(cfg, 1, make([]int, countTTTMLPBlocks(cfg.Blocks))); statefulErr == nil {
				return runGenerateTTTMLPStateful(opts.ConfigPath, opts.SafetensorsLoad, cfg, opts, plan, vocab, output.writer)
			}
		}
		return runGenerateReplay(cfg, opts, plan, vocab, output.writer)
	}()
	return errors.Join(runErr, output.Close())
}

func runGenerateReplay(cfg *ArchConfig, opts GenerateOptions, plan generationPlan, vocab *data.NucleotideVocabulary, output generationWriter) error {
	genCfg := *cfg
	genCfg.Training.BatchTokens = plan.batchSize * genCfg.SeqLen
	if err := configureCharFeaturesForConfigPath(&genCfg, opts.ConfigPath, opts.SafetensorsLoad); err != nil {
		return err
	}

	var (
		prog *arch.Program
		err  error
	)
	if plan.batchSize == 1 {
		prog, err = BuildEvalIRProgramFromConfig(&genCfg)
	} else {
		prog, err = arch.BuildGenerationIRProgramFromConfig(&genCfg)
	}
	if err != nil {
		return fmt.Errorf("build IR program: %w", err)
	}
	shapes, err := computeWeightShapes(&genCfg)
	if err != nil {
		return fmt.Errorf("compute weight shapes: %w", err)
	}
	loadedWeights, err := loadSafetensorsWeights(opts.SafetensorsLoad, shapes)
	if err != nil {
		return fmt.Errorf("load safetensors %q: %w", opts.SafetensorsLoad, err)
	}
	trainer, err := initGPUTrainer(prog, &genCfg, loadedWeights, nil)
	if err != nil {
		return fmt.Errorf("init GPU trainer: %w", err)
	}
	defer trainer.CloseTrainer()
	evaluator, ok := trainer.(causalGenerationEvaluator)
	if !ok {
		return fmt.Errorf("trainer does not support causal generation output readback")
	}
	if plan.batchSize > 1 {
		batchEvaluator, ok := trainer.(batchedCausalGenerationEvaluator)
		if !ok {
			return fmt.Errorf("trainer does not support batched causal generation")
		}
		return runBatchedGenerationSamples(cfg, batchEvaluator, opts, plan, vocab, output)
	}
	return runGenerationSamples(plan, vocab, output, func(_ int, rng *rand.Rand) (generationSample, error) {
		return generateReplaySample(cfg, evaluator, opts, plan.eosTokenID, rng, vocab)
	})
}

func runGenerateTTTMLPStateful(configPath, safetensorsLoad string, cfg *ArchConfig, opts GenerateOptions, plan generationPlan, vocab *data.NucleotideVocabulary, output generationWriter) error {
	session, err := NewTTTMLPInferenceSession(configPath, safetensorsLoad)
	if err != nil {
		return err
	}
	runErr := runGenerationSamples(plan, vocab, output, func(_ int, rng *rand.Rand) (generationSample, error) {
		return generateTTTMLPStatefulSample(session, cfg, opts, plan.eosTokenID, rng, vocab)
	})
	return errors.Join(runErr, session.Close())
}

func generationPromptTokens(prompt string, vocabSize int, rng *rand.Rand) ([]int, error) {
	return generationPromptTokensWithVocabulary(prompt, vocabSize, rng, nil)
}

func generationPromptTokensWithVocabulary(prompt string, vocabSize int, rng *rand.Rand, vocab *data.NucleotideVocabulary) ([]int, error) {
	if vocab != nil {
		if prompt == "" {
			bos, _ := vocab.SpecialTokenID("bos")
			return []int{bos}, nil
		}
		const sequencePrefix = "sequence:"
		if strings.HasPrefix(prompt, sequencePrefix) {
			ids, err := vocab.Encode(strings.TrimSpace(strings.TrimPrefix(prompt, sequencePrefix)))
			if err != nil {
				return nil, err
			}
			bos, _ := vocab.SpecialTokenID("bos")
			return append([]int{bos}, ids...), nil
		}
	}
	if prompt == "" {
		return []int{rng.Intn(vocabSize)}, nil
	}
	const prefix = "token_ids:"
	if !strings.HasPrefix(prompt, prefix) {
		return nil, fmt.Errorf("-prompt must use the form %q", prefix+"0,1,2")
	}
	body := strings.TrimSpace(strings.TrimPrefix(prompt, prefix))
	if body == "" {
		return nil, fmt.Errorf("-prompt token list is empty")
	}
	parts := strings.Split(body, ",")
	tokens := make([]int, 0, len(parts))
	for _, part := range parts {
		v, err := strconv.Atoi(strings.TrimSpace(part))
		if err != nil {
			return nil, fmt.Errorf("parse prompt token %q: %w", part, err)
		}
		if v < 0 || v >= vocabSize {
			return nil, fmt.Errorf("prompt token %d out of range [0,%d)", v, vocabSize)
		}
		tokens = append(tokens, v)
	}
	return tokens, nil
}

func generationBatch(context []int, seqLen int) ([]int, []int, int) {
	xTok := make([]int, seqLen)
	yTok := make([]int, seqLen)
	copy(xTok, context)
	if len(context) > 1 {
		copy(yTok, context[1:])
	}
	lastPos := len(context) - 1
	if lastPos < 0 {
		lastPos = 0
	}
	yTok[lastPos] = xTok[lastPos]
	return xTok, yTok, lastPos
}

func sampleNextToken(logits []float32, temperature float32, topK int, rng *rand.Rand) (int, error) {
	if len(logits) == 0 {
		return 0, fmt.Errorf("empty logits")
	}
	type candidate struct {
		token int
		logit float64
	}
	candidates := make([]candidate, len(logits))
	for i, logit := range logits {
		candidates[i] = candidate{token: i, logit: float64(logit) / float64(temperature)}
	}
	slices.SortFunc(candidates, func(a, b candidate) int {
		switch {
		case a.logit > b.logit:
			return -1
		case a.logit < b.logit:
			return 1
		default:
			return 0
		}
	})
	if topK > 0 && topK < len(candidates) {
		candidates = candidates[:topK]
	}

	maxLogit := candidates[0].logit
	weights := make([]float64, len(candidates))
	total := 0.0
	for i, cand := range candidates {
		weight := math.Exp(cand.logit - maxLogit)
		weights[i] = weight
		total += weight
	}
	if total == 0 || math.IsNaN(total) || math.IsInf(total, 0) {
		return candidates[0].token, nil
	}
	draw := rng.Float64() * total
	accum := 0.0
	for i, weight := range weights {
		accum += weight
		if draw <= accum {
			return candidates[i].token, nil
		}
	}
	return candidates[len(candidates)-1].token, nil
}

func formatTokenIDs(tokens []int) string {
	parts := make([]string, len(tokens))
	for i, token := range tokens {
		parts[i] = strconv.Itoa(token)
	}
	return strings.Join(parts, ",")
}
