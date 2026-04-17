// Package main implements the mixlab command-line modes.
package train

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"slices"
	"strconv"
	"strings"
)

func runGenerate(configPath, safetensorsLoad string, maxTokens int, temperature float32, topK int, prompt string) error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	if configPath == "" {
		return fmt.Errorf("-config is required for generate mode")
	}
	if safetensorsLoad == "" {
		return fmt.Errorf("-safetensors-load is required for generate mode")
	}
	if maxTokens < 0 {
		return fmt.Errorf("-max-tokens must be >= 0")
	}
	if temperature <= 0 {
		return fmt.Errorf("-temperature must be > 0")
	}
	if topK < 0 {
		return fmt.Errorf("-top-k must be >= 0")
	}

	cfg, err := LoadArchConfig(configPath)
	if err != nil {
		return err
	}
	genCfg := *cfg
	genCfg.Training.BatchTokens = genCfg.SeqLen

	prog, err := BuildIRProgramFromConfig(&genCfg)
	if err != nil {
		return fmt.Errorf("build IR program: %w", err)
	}
	shapes, err := computeWeightShapes(&genCfg)
	if err != nil {
		return fmt.Errorf("compute weight shapes: %w", err)
	}
	loadedWeights, err := loadSafetensorsWeights(safetensorsLoad, shapes)
	if err != nil {
		return fmt.Errorf("load safetensors %q: %w", safetensorsLoad, err)
	}
	trainer, err := initGPUTrainer(prog, &genCfg, loadedWeights)
	if err != nil {
		return fmt.Errorf("init GPU trainer: %w", err)
	}
	defer trainer.CloseTrainer()

	rng := rand.New(rand.NewSource(cfg.Training.Seed))
	context, err := generationPromptTokens(prompt, cfg.VocabSize, rng)
	if err != nil {
		return err
	}
	if len(context) > cfg.SeqLen {
		return fmt.Errorf("prompt length %d exceeds seq_len %d", len(context), cfg.SeqLen)
	}

	for i := 0; i < maxTokens; i++ {
		if len(context) >= cfg.SeqLen {
			fmt.Printf("stopped at seq_len limit (%d tokens)\n", cfg.SeqLen)
			break
		}
		xTok, yTok, lastPos := generationBatch(context, cfg.SeqLen)
		if _, err := trainer.EvaluateGPU(xTok, yTok, 1, cfg.SeqLen); err != nil {
			return fmt.Errorf("generate step %d: %w", i, err)
		}
		logits, err := readTrainerOutput(trainer, "logits", []int{cfg.SeqLen, cfg.VocabSize})
		if err != nil {
			return fmt.Errorf("read logits: %w", err)
		}
		next, err := sampleNextToken(logits[lastPos*cfg.VocabSize:(lastPos+1)*cfg.VocabSize], temperature, topK, rng)
		if err != nil {
			return fmt.Errorf("sample token at step %d: %w", i, err)
		}
		context = append(context, next)
	}

	fmt.Printf("generated token_ids:%s\n", formatTokenIDs(context))
	return nil
}

func generationPromptTokens(prompt string, vocabSize int, rng *rand.Rand) ([]int, error) {
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
