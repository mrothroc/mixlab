// Package main implements the mixlab command-line modes.
package train

import (
	"fmt"
	"runtime"
	"strings"

	"github.com/mrothroc/mixlab/data"
)

func runEvalMode(configPath, trainPattern, safetensorsLoad string) error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	if configPath == "" {
		return fmt.Errorf("-config is required for eval mode; pass a JSON config file, e.g.: mixlab -mode eval -config examples/plain_3L.json -safetensors-load weights.st -train 'data/train_*.bin'")
	}
	if safetensorsLoad == "" {
		return fmt.Errorf("-safetensors-load is required for eval mode")
	}
	if trainPattern == "" {
		return fmt.Errorf("-train is required for eval mode; pass a glob pattern for data shards, e.g.: -train 'data/train_*.bin'")
	}

	cfg, err := LoadArchConfig(configPath)
	if err != nil {
		return err
	}
	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		return fmt.Errorf("build IR program: %w", err)
	}
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		return fmt.Errorf("compute weight shapes: %w", err)
	}
	loadedWeights, err := loadSafetensorsWeights(safetensorsLoad, shapes)
	if err != nil {
		return fmt.Errorf("load safetensors %q: %w", safetensorsLoad, err)
	}
	trainer, err := initGPUTrainer(prog, cfg, loadedWeights)
	if err != nil {
		return fmt.Errorf("init GPU trainer: %w", err)
	}
	defer trainer.CloseTrainer()

	batchTokens := cfg.Training.BatchTokens
	seqLen := cfg.SeqLen
	if batchTokens%seqLen != 0 {
		return fmt.Errorf("batch_tokens (%d) must be divisible by seq_len (%d)", batchTokens, seqLen)
	}
	batchSize := batchTokens / seqLen

	const defaultValBatchCount = 10
	valPattern := strings.Replace(trainPattern, "train", "val", 1)
	valSet, err := data.NewValSet(valPattern, cfg.Training.Seed, defaultValBatchCount, batchTokens, seqLen, effectiveShuffleChunkTokens(cfg))
	if err != nil {
		return fmt.Errorf("load val set %q: %w", valPattern, err)
	}

	tttSteps := cfg.Training.TTTSteps
	tttMode := cfg.Training.TTTMode
	tttLR := float32(cfg.Training.TTTLR)
	tttRank := cfg.Training.TTTRank
	valLoss, err := meanValidationLossWithTTT(valSet, trainer, batchSize, seqLen, tttMode, tttSteps, tttLR, tttRank)
	if err != nil {
		return fmt.Errorf("evaluate validation loss: %w", err)
	}

	fmt.Printf("loaded config %q: model_dim=%d vocab_size=%d seq_len=%d blocks=%d\n",
		cfg.Name, cfg.ModelDim, cfg.VocabSize, cfg.SeqLen, len(cfg.Blocks))
	fmt.Printf("  [%s] loaded %d weights from %s\n", cfg.Name, len(loadedWeights), safetensorsLoad)
	if tttSteps > 0 {
		if tttMode == "lora" {
			fmt.Printf("  [%s] validation loss=%.6f (LoRA-TTT steps=%d lr=%g rank=%d)\n", cfg.Name, valLoss, tttSteps, tttLR, tttRank)
		} else {
			fmt.Printf("  [%s] validation loss=%.6f (score-first TTT steps=%d lr=%g)\n", cfg.Name, valLoss, tttSteps, tttLR)
		}
	} else {
		fmt.Printf("  [%s] validation loss=%.6f\n", cfg.Name, valLoss)
	}
	return nil
}
