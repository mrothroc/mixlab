package train

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"

	"github.com/mrothroc/mixlab/arch"
)

type diffusionGenerationEvaluator interface {
	gpuObjectiveEvaluator
	ReadOutput(name string, shape []int) ([]float32, error)
}

type gpuObjectiveOutputEvaluator interface {
	EvaluateObjectiveGPUWithOutputs(batch objectiveBatch, batchSize, seqLen int, outputNames []string) (float32, error)
}

func evaluateObjectiveAndCacheOutputs(evaluator diffusionGenerationEvaluator, batch objectiveBatch, batchSize, seqLen int, outputNames ...string) (float32, error) {
	if outputEvaluator, ok := evaluator.(gpuObjectiveOutputEvaluator); ok {
		return outputEvaluator.EvaluateObjectiveGPUWithOutputs(batch, batchSize, seqLen, outputNames)
	}
	return evaluator.EvaluateObjectiveGPU(batch, batchSize, seqLen)
}

type diffusionGenerationResult struct {
	tokens          []int
	commits         []diffusionCommit
	trace           []diffusionSamplerTraceEntry
	blocks          int
	steps           int
	stoppedAtSeqLen bool
}

type GenerateDiffusionOptions struct {
	ConfigPath                   string
	SafetensorsLoad              string
	MaxTokens                    int
	Prompt                       string
	DiffusionStepsPerBlock       int
	DiffusionConfidenceThreshold *float64
	DiffusionCommitFloor         int
	DiffusionTemperature         float32
	DiffusionTopK                int
	DiffusionTraceOut            string
}

type diffusionGenerationRuntimeOptions struct {
	stepsPerBlock       int
	confidenceThreshold *float64
	commitFloor         int
	temperature         float32
	topK                int
	rng                 *rand.Rand
}

func runGenerateDiffusion(configPath, safetensorsLoad string, maxTokens int, prompt string) error {
	return runGenerateDiffusionWithOptions(GenerateDiffusionOptions{
		ConfigPath:      configPath,
		SafetensorsLoad: safetensorsLoad,
		MaxTokens:       maxTokens,
		Prompt:          prompt,
	})
}

func runGenerateDiffusionWithOptions(opts GenerateDiffusionOptions) error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	if opts.ConfigPath == "" {
		return fmt.Errorf("-config is required for generate-diffusion mode")
	}
	if opts.SafetensorsLoad == "" {
		return fmt.Errorf("-safetensors-load is required for generate-diffusion mode")
	}
	if opts.MaxTokens < 0 {
		return fmt.Errorf("-max-tokens must be >= 0")
	}
	if opts.DiffusionStepsPerBlock < 0 {
		return fmt.Errorf("-diffusion-steps-per-block must be >= 0")
	}
	if opts.DiffusionCommitFloor < 0 {
		return fmt.Errorf("-diffusion-commit-floor must be >= 0")
	}
	if opts.DiffusionConfidenceThreshold != nil {
		v := *opts.DiffusionConfidenceThreshold
		if math.IsNaN(v) || v < 0 || v > 1 {
			return fmt.Errorf("-diffusion-confidence-threshold must be in [0,1]")
		}
	}
	if opts.DiffusionTemperature < 0 {
		return fmt.Errorf("-diffusion-temperature must be >= 0")
	}
	if opts.DiffusionTopK < 0 {
		return fmt.Errorf("-diffusion-top-k must be >= 0")
	}

	cfg, err := LoadArchConfig(opts.ConfigPath)
	if err != nil {
		return err
	}
	if !cfg.Training.UsesBlockDiffusionObjective() {
		return fmt.Errorf("generate-diffusion requires training.objective=%q or hybrid_secondary_objective=%q, got objective=%q secondary=%q", arch.ObjectiveBlockDiffusion, arch.ObjectiveBlockDiffusion, cfg.Training.EffectiveObjective(), cfg.Training.EffectiveHybridSecondaryObjective())
	}
	genCfg := *cfg
	genCfg.Training.BatchTokens = genCfg.SeqLen
	if err := configureCharFeaturesForConfigPath(&genCfg, opts.ConfigPath, opts.SafetensorsLoad); err != nil {
		return err
	}

	prog, err := buildGenerateDiffusionIRProgram(&genCfg)
	if err != nil {
		return fmt.Errorf("build diffusion generation IR program: %w", err)
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
	evaluator, ok := trainer.(diffusionGenerationEvaluator)
	if !ok {
		return fmt.Errorf("trainer does not support diffusion generation logits; ensure you are using the MLX backend")
	}

	rng := rand.New(rand.NewSource(cfg.Training.Seed))
	context, err := generationPromptTokens(opts.Prompt, cfg.VocabSize, rng)
	if err != nil {
		return err
	}
	if len(context) > cfg.SeqLen {
		return fmt.Errorf("prompt length %d exceeds seq_len %d", len(context), cfg.SeqLen)
	}

	result, err := generateDiffusionTokensWithOptions(&genCfg, evaluator, context, opts.MaxTokens, diffusionGenerationRuntimeOptions{
		stepsPerBlock:       opts.DiffusionStepsPerBlock,
		confidenceThreshold: opts.DiffusionConfidenceThreshold,
		commitFloor:         opts.DiffusionCommitFloor,
		temperature:         opts.DiffusionTemperature,
		topK:                opts.DiffusionTopK,
		rng:                 rng,
	})
	if err != nil {
		return err
	}
	if err := writeDiffusionTrace(opts.DiffusionTraceOut, result.trace); err != nil {
		return err
	}
	if result.stoppedAtSeqLen {
		fmt.Printf("stopped at seq_len limit (%d tokens)\n", cfg.SeqLen)
	}
	fmt.Printf("generated token_ids:%s\n", formatTokenIDs(result.tokens))
	return nil
}

func buildGenerateDiffusionIRProgram(cfg *ArchConfig) (*arch.Program, error) {
	if cfg == nil {
		return nil, fmt.Errorf("nil config")
	}
	state := TrainingProgramState{
		RecurrenceActive:     true,
		HeadUntied:           cfg.MTPUntieEnabled(),
		MTPAuxInactive:       true,
		DistillationInactive: true,
		Data2VecInactive:     true,
		InvarianceInactive:   true,
		PLLMarginInactive:    true,
		ZLossInactive:        true,
		DropoutInactive:      true,
		Objective:            arch.ObjectiveBlockDiffusion,
	}
	return BuildTrainingIRProgramFromConfig(cfg, state)
}

func generateDiffusionTokens(cfg *ArchConfig, evaluator diffusionGenerationEvaluator, prompt []int, maxTokens int) (diffusionGenerationResult, error) {
	return generateDiffusionTokensWithOptions(cfg, evaluator, prompt, maxTokens, diffusionGenerationRuntimeOptions{})
}

func generateDiffusionTokensWithOptions(cfg *ArchConfig, evaluator diffusionGenerationEvaluator, prompt []int, maxTokens int, opts diffusionGenerationRuntimeOptions) (diffusionGenerationResult, error) {
	if cfg == nil {
		return diffusionGenerationResult{}, fmt.Errorf("nil config")
	}
	if evaluator == nil {
		return diffusionGenerationResult{}, fmt.Errorf("nil diffusion generation evaluator")
	}
	if maxTokens < 0 {
		return diffusionGenerationResult{}, fmt.Errorf("maxTokens must be >= 0, got %d", maxTokens)
	}
	if cfg.SeqLen <= 0 {
		return diffusionGenerationResult{}, fmt.Errorf("invalid seq_len=%d", cfg.SeqLen)
	}
	if cfg.VocabSize <= 0 {
		return diffusionGenerationResult{}, fmt.Errorf("invalid vocab_size=%d", cfg.VocabSize)
	}
	if !cfg.Training.UsesBlockDiffusionObjective() {
		return diffusionGenerationResult{}, fmt.Errorf("generate-diffusion requires training.objective=%q or hybrid_secondary_objective=%q, got objective=%q secondary=%q", arch.ObjectiveBlockDiffusion, arch.ObjectiveBlockDiffusion, cfg.Training.EffectiveObjective(), cfg.Training.EffectiveHybridSecondaryObjective())
	}
	if len(prompt) == 0 {
		return diffusionGenerationResult{}, fmt.Errorf("prompt must contain at least one token")
	}
	if len(prompt) > cfg.SeqLen {
		return diffusionGenerationResult{}, fmt.Errorf("prompt length %d exceeds seq_len %d", len(prompt), cfg.SeqLen)
	}
	for i, token := range prompt {
		if token < 0 || token >= cfg.VocabSize {
			return diffusionGenerationResult{}, fmt.Errorf("prompt token %d at position %d out of range [0,%d)", token, i, cfg.VocabSize)
		}
	}

	spec, err := diffusionGenerationSpec(cfg)
	if err != nil {
		return diffusionGenerationResult{}, err
	}
	if err := applyDiffusionGenerationOverrides(&spec, opts); err != nil {
		return diffusionGenerationResult{}, err
	}
	rng := opts.rng
	if rng == nil {
		rng = rand.New(rand.NewSource(cfg.Training.Seed))
	}

	tokens := append([]int(nil), prompt...)
	result := diffusionGenerationResult{tokens: tokens}
	if maxTokens == 0 {
		return result, nil
	}

	available := cfg.SeqLen - len(tokens)
	toGenerate := maxTokens
	if toGenerate > available {
		toGenerate = available
		result.stoppedAtSeqLen = true
	}
	generated := 0
	for generated < toGenerate {
		blockLen := spec.BlockSize
		if remaining := toGenerate - generated; blockLen > remaining {
			blockLen = remaining
		}
		if blockLen <= 0 {
			break
		}
		blockStart := len(tokens)
		blockEnd := blockStart + blockLen
		for pos := blockStart; pos < blockEnd; pos++ {
			tokens = append(tokens, cfg.Training.MLMMaskTokenID)
		}

		samplerCfg := diffusionSamplerConfig{
			blockIndex:          result.blocks,
			blockStart:          blockStart,
			blockEnd:            blockEnd,
			stepsPerBlock:       spec.StepsPerBlock,
			confidenceThreshold: float32(spec.ConfidenceThreshold),
			commitFloor:         spec.CommitFloor,
			maskTokenID:         cfg.Training.MLMMaskTokenID,
		}
		blockResult, err := runDiffusionSamplerSteps(samplerCfg, tokens, nil, func(input diffusionSamplerStepInput) ([]diffusionTokenPrediction, error) {
			batch, err := diffusionGenerationBatch(input.tokens, input.unresolved, blockStart, blockEnd, cfg.SeqLen)
			if err != nil {
				return nil, err
			}
			batch = expandBatchForMultiheadDiffusion(cfg, batch, 1, cfg.SeqLen)
			evalBatchSize := 1
			if batch.batchSizeOverride > 0 {
				evalBatchSize = batch.batchSizeOverride
			}
			outputName := diffusionLogitsOutputName(cfg)
			if _, err := evaluateObjectiveAndCacheOutputs(evaluator, batch, evalBatchSize, cfg.SeqLen, outputName); err != nil {
				return nil, err
			}
			logits, err := evaluator.ReadOutput(outputName, []int{cfg.SeqLen, cfg.VocabSize})
			if err != nil {
				return nil, fmt.Errorf("read %s: %w", outputName, err)
			}
			if len(logits) != cfg.SeqLen*cfg.VocabSize {
				return nil, fmt.Errorf("logits length mismatch: got=%d want=%d", len(logits), cfg.SeqLen*cfg.VocabSize)
			}
			if opts.temperature > 0 {
				return diffusionPredictionsFromLogitsSampled(logits, input.unresolved, cfg.VocabSize, opts.temperature, opts.topK, rng)
			}
			return diffusionPredictionsFromLogits(logits, input.unresolved, cfg.VocabSize)
		})
		if err != nil {
			return diffusionGenerationResult{}, fmt.Errorf("diffusion block [%d,%d): %w", blockStart, blockEnd, err)
		}

		tokens = blockResult.tokens
		result.commits = append(result.commits, blockResult.commits...)
		result.trace = append(result.trace, blockResult.trace...)
		result.steps += blockResult.steps
		result.blocks++
		result.tokens = append([]int(nil), tokens...)
		generated += blockLen
	}
	return result, nil
}

func applyDiffusionGenerationOverrides(spec *arch.DiffusionSpec, opts diffusionGenerationRuntimeOptions) error {
	if spec == nil {
		return fmt.Errorf("nil diffusion generation spec")
	}
	if opts.stepsPerBlock < 0 {
		return fmt.Errorf("diffusion steps_per_block override must be >= 0, got %d", opts.stepsPerBlock)
	}
	if opts.stepsPerBlock > 0 {
		spec.StepsPerBlock = opts.stepsPerBlock
	}
	if opts.confidenceThreshold != nil {
		v := *opts.confidenceThreshold
		if math.IsNaN(v) || v < 0 || v > 1 {
			return fmt.Errorf("diffusion confidence_threshold override must be in [0,1], got %g", v)
		}
		spec.ConfidenceThreshold = v
	}
	if opts.commitFloor < 0 {
		return fmt.Errorf("diffusion commit_floor override must be >= 0, got %d", opts.commitFloor)
	}
	if opts.commitFloor > 0 {
		spec.CommitFloor = opts.commitFloor
	}
	if spec.StepsPerBlock <= 0 {
		return fmt.Errorf("diffusion steps_per_block=%d must be > 0", spec.StepsPerBlock)
	}
	if spec.CommitFloor <= 0 || spec.CommitFloor > spec.BlockSize {
		return fmt.Errorf("diffusion commit_floor=%d must be in [1,block_size=%d]", spec.CommitFloor, spec.BlockSize)
	}
	if opts.temperature < 0 {
		return fmt.Errorf("diffusion temperature must be >= 0, got %g", opts.temperature)
	}
	if opts.topK < 0 {
		return fmt.Errorf("diffusion top_k must be >= 0, got %d", opts.topK)
	}
	return nil
}

func writeDiffusionTrace(path string, trace []diffusionSamplerTraceEntry) error {
	if path == "" {
		return nil
	}
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create diffusion trace %q: %w", path, err)
	}
	enc := json.NewEncoder(f)
	for _, entry := range trace {
		if err := enc.Encode(entry); err != nil {
			_ = f.Close()
			return fmt.Errorf("write diffusion trace %q: %w", path, err)
		}
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("close diffusion trace %q: %w", path, err)
	}
	return nil
}

func diffusionGenerationBatch(tokens []int, unresolved []int, blockStart, blockEnd, seqLen int) (objectiveBatch, error) {
	if seqLen <= 0 {
		return objectiveBatch{}, fmt.Errorf("invalid seq_len=%d", seqLen)
	}
	if len(tokens) > seqLen {
		return objectiveBatch{}, fmt.Errorf("token length %d exceeds seq_len %d", len(tokens), seqLen)
	}
	if blockStart < 0 || blockEnd <= blockStart || blockEnd > seqLen {
		return objectiveBatch{}, fmt.Errorf("invalid diffusion block [%d,%d) for seq_len %d", blockStart, blockEnd, seqLen)
	}
	x := make([]int, seqLen)
	copy(x, tokens)
	y := append([]int(nil), x...)
	lossMask := make([]float32, seqLen)
	for _, pos := range unresolved {
		if pos < blockStart || pos >= blockEnd {
			return objectiveBatch{}, fmt.Errorf("unresolved position %d outside diffusion block [%d,%d)", pos, blockStart, blockEnd)
		}
		lossMask[pos] = 1
	}
	return objectiveBatch{
		x:                   x,
		y:                   y,
		lossMask:            lossMask,
		unmaskedX:           append([]int(nil), x...),
		diffusionBlockStart: []int32{int32(blockStart)},
		diffusionBlockEnd:   []int32{int32(blockEnd)},
		diffusionTimestep:   []float32{float32(countPositiveMask(lossMask)) / float32(blockEnd-blockStart)},
	}, nil
}

func diffusionGenerationSpec(cfg *ArchConfig) (arch.DiffusionSpec, error) {
	if cfg == nil {
		return arch.DiffusionSpec{}, fmt.Errorf("nil config")
	}
	if cfg.Training.MLMMaskTokenID < 0 || cfg.Training.MLMMaskTokenID >= cfg.VocabSize {
		return arch.DiffusionSpec{}, fmt.Errorf("invalid mlm_mask_token_id=%d for vocab_size=%d", cfg.Training.MLMMaskTokenID, cfg.VocabSize)
	}
	spec, err := diffusionSpecForObjectiveBatch(cfg, cfg.SeqLen)
	if err != nil {
		return arch.DiffusionSpec{}, err
	}
	if spec.StepsPerBlock <= 0 {
		spec.StepsPerBlock = spec.BlockSize
	}
	if spec.CommitFloor <= 0 {
		spec.CommitFloor = 1
	}
	if spec.CommitFloor > spec.BlockSize {
		return arch.DiffusionSpec{}, fmt.Errorf("diffusion commit_floor=%d must be <= block_size=%d", spec.CommitFloor, spec.BlockSize)
	}
	if math.IsNaN(spec.ConfidenceThreshold) || spec.ConfidenceThreshold < 0 || spec.ConfidenceThreshold > 1 {
		return arch.DiffusionSpec{}, fmt.Errorf("invalid diffusion confidence_threshold=%g", spec.ConfidenceThreshold)
	}
	return spec, nil
}
