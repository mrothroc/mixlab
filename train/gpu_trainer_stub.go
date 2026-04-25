//go:build !mlx || !cgo || (!darwin && !linux)

package train

import (
	"fmt"

	ir "github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/gpu"
)

// initMLXGPUTrainer is a stub that returns an error on non-MLX platforms.
func initMLXGPUTrainer(
	_ *ir.Program,
	_ *ArchConfig,
	_ [][]float32,
	_ func(gpu.TrainerOptimizerSpec, []WeightShape) (gpu.TrainerOptimizerSpec, error),
) (GPUTrainer, error) {
	return nil, fmt.Errorf("MLX backend unavailable; rebuild with: CGO_ENABLED=1 go build -tags mlx -o mixlab ./cmd/mixlab (macOS: brew install mlx; Linux: see docker/README.md)")
}
