// Package main implements the mixlab command-line modes.
package train

import (
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"runtime"

	"github.com/mrothroc/mixlab/data"
)

func runHiddenstats(configPath, trainPattern, safetensorsLoad, outputPath string) error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	if configPath == "" {
		return fmt.Errorf("-config is required for hiddenstats mode")
	}
	if safetensorsLoad == "" {
		return fmt.Errorf("-safetensors-load is required for hiddenstats mode")
	}
	if trainPattern == "" {
		return fmt.Errorf("-train is required for hiddenstats mode")
	}
	if outputPath == "" {
		return fmt.Errorf("-output is required for hiddenstats mode")
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

	loader, err := data.NewLoader(trainPattern, cfg.Training.Seed, effectiveShuffleChunkTokens(cfg))
	if err != nil {
		return err
	}
	xTok, yTok, err := loader.NextBatch(batchTokens, seqLen)
	if err != nil {
		return fmt.Errorf("load batch: %w", err)
	}
	if _, err := trainer.EvaluateGPU(xTok, yTok, batchSize, seqLen); err != nil {
		return fmt.Errorf("forward pass: %w", err)
	}

	hiddenShape := hiddenStatsOutputShape(batchSize, seqLen, cfg.ModelDim)
	hidden, err := readTrainerOutput(trainer, "x_hidden", hiddenShape)
	if err != nil {
		return fmt.Errorf("read x_hidden: %w", err)
	}
	if err := writeFloat32Binary(outputPath, hidden); err != nil {
		return fmt.Errorf("write hidden stats %q: %w", outputPath, err)
	}

	info, err := os.Stat(outputPath)
	if err != nil {
		return fmt.Errorf("stat hidden stats %q: %w", outputPath, err)
	}
	fmt.Printf("hidden state dims: [%d, %d, %d]\n", batchSize, seqLen, cfg.ModelDim)
	fmt.Printf("wrote %s (%d bytes)\n", outputPath, info.Size())
	return nil
}

func hiddenStatsOutputShape(batchSize, seqLen, modelDim int) []int {
	return []int{batchSize, seqLen, modelDim}
}

func writeFloat32Binary(path string, values []float32) error {
	if dir := filepath.Dir(path); dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, values); err != nil {
		_ = f.Close()
		return err
	}
	return f.Close()
}
