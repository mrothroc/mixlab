package train

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"

	"github.com/mrothroc/mixlab/arch"
)

type diffusionGenerationEvaluator interface {
	gpuObjectiveEvaluator
	ReadOutput(name string, shape []int) ([]float32, error)
}

type diffusionGenerationResult struct {
	tokens          []int
	commits         []diffusionCommit
	blocks          int
	steps           int
	stoppedAtSeqLen bool
}

func runGenerateDiffusion(configPath, safetensorsLoad string, maxTokens int, prompt string) error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	if configPath == "" {
		return fmt.Errorf("-config is required for generate-diffusion mode")
	}
	if safetensorsLoad == "" {
		return fmt.Errorf("-safetensors-load is required for generate-diffusion mode")
	}
	if maxTokens < 0 {
		return fmt.Errorf("-max-tokens must be >= 0")
	}

	cfg, err := LoadArchConfig(configPath)
	if err != nil {
		return err
	}
	if cfg.Training.EffectiveObjective() != arch.ObjectiveBlockDiffusion {
		return fmt.Errorf("generate-diffusion requires training.objective=%q, got %q", arch.ObjectiveBlockDiffusion, cfg.Training.EffectiveObjective())
	}
	genCfg := *cfg
	genCfg.Training.BatchTokens = genCfg.SeqLen
	if err := configureCharFeaturesForConfigPath(&genCfg, configPath, safetensorsLoad); err != nil {
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
	loadedWeights, err := loadSafetensorsWeights(safetensorsLoad, shapes)
	if err != nil {
		return fmt.Errorf("load safetensors %q: %w", safetensorsLoad, err)
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
	context, err := generationPromptTokens(prompt, cfg.VocabSize, rng)
	if err != nil {
		return err
	}
	if len(context) > cfg.SeqLen {
		return fmt.Errorf("prompt length %d exceeds seq_len %d", len(context), cfg.SeqLen)
	}

	result, err := generateDiffusionTokens(&genCfg, evaluator, context, maxTokens)
	if err != nil {
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
		ZLossInactive:        true,
		DropoutInactive:      true,
		Objective:            arch.ObjectiveBlockDiffusion,
	}
	return BuildTrainingIRProgramFromConfig(cfg, state)
}

func generateDiffusionTokens(cfg *ArchConfig, evaluator diffusionGenerationEvaluator, prompt []int, maxTokens int) (diffusionGenerationResult, error) {
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
	if cfg.Training.EffectiveObjective() != arch.ObjectiveBlockDiffusion {
		return diffusionGenerationResult{}, fmt.Errorf("generate-diffusion requires training.objective=%q, got %q", arch.ObjectiveBlockDiffusion, cfg.Training.EffectiveObjective())
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
			if _, err := evaluator.EvaluateObjectiveGPU(batch, 1, cfg.SeqLen); err != nil {
				return nil, err
			}
			logits, err := evaluator.ReadOutput("logits", []int{cfg.SeqLen, cfg.VocabSize})
			if err != nil {
				return nil, fmt.Errorf("read logits: %w", err)
			}
			if len(logits) != cfg.SeqLen*cfg.VocabSize {
				return nil, fmt.Errorf("logits length mismatch: got=%d want=%d", len(logits), cfg.SeqLen*cfg.VocabSize)
			}
			return diffusionPredictionsFromLogits(logits, input.unresolved, cfg.VocabSize)
		})
		if err != nil {
			return diffusionGenerationResult{}, fmt.Errorf("diffusion block [%d,%d): %w", blockStart, blockEnd, err)
		}

		tokens = blockResult.tokens
		result.commits = append(result.commits, blockResult.commits...)
		result.steps += blockResult.steps
		result.blocks++
		result.tokens = append([]int(nil), tokens...)
		generated += blockLen
	}
	return result, nil
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
