//go:build !mlx || !cgo || (!darwin && !linux)

package train

import (
	"fmt"

	ir "github.com/mrothroc/mixlab/arch"
)

// initMLXGPUTrainer is a stub that returns an error on non-MLX platforms.
func initMLXGPUTrainer(_ *ir.Program, _ *ArchConfig, _ [][]float32) (GPUTrainer, error) {
	return nil, fmt.Errorf("MLX backend unavailable; rebuild with: CGO_ENABLED=1 go build -tags mlx -o mixlab ./cmd/mixlab (requires macOS with Apple Silicon)")
}
