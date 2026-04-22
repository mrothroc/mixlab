// Package train implements the mixlab command-line modes.
package train

import (
	"fmt"

	"github.com/mrothroc/mixlab/arch"
)

const bytesPerMiB = 1024 * 1024

func runCount(configPath string) error {
	if configPath == "" {
		return fmt.Errorf("-config is required for count mode; pass a JSON config file, e.g.: mixlab -mode count -config examples/plain_3L.json")
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

	totalParams := 0
	for _, shape := range shapes {
		totalParams += shapeElementCount(shape.Shape)
	}

	float32MiB := float64(totalParams*4) / bytesPerMiB
	int8MiB := float64(totalParams) / bytesPerMiB

	fmt.Printf("Total parameters: %d\n", totalParams)
	fmt.Printf("Total size (float32, in MB): %.2f\n", float32MiB)
	fmt.Printf("Total size (int8 quantized, in MB): %.2f\n", int8MiB)
	fmt.Printf("Number of blocks: %d\n", countConfigBlocks(cfg))
	fmt.Printf("Number of IR ops: %d\n", len(prog.Ops))
	flops := arch.EstimateFLOPs(cfg)
	fmt.Printf("Forward FLOPs: %s\n", formatFLOPs(flops.ForwardFLOPs))
	fmt.Printf("Training step FLOPs: %s\n", formatFLOPs(flops.TrainingFLOPs))
	fmt.Printf("FLOPs per token: %s\n", formatFLOPs(flops.FLOPsPerToken))
	return nil
}

func formatFLOPs(f int64) string {
	v := float64(f)
	switch {
	case v >= 1e12:
		return fmt.Sprintf("%.2f TFLOP", v/1e12)
	case v >= 1e9:
		return fmt.Sprintf("%.2f GFLOP", v/1e9)
	case v >= 1e6:
		return fmt.Sprintf("%.2f MFLOP", v/1e6)
	default:
		return fmt.Sprintf("%d FLOP", f)
	}
}

func shapeElementCount(shape []int) int {
	n := 1
	for _, dim := range shape {
		n *= dim
	}
	return n
}

func countConfigBlocks(cfg *ArchConfig) int {
	if cfg == nil {
		return 0
	}
	return len(cfg.Blocks)
}
