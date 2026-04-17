package train

import (
	"fmt"

	ir "github.com/mrothroc/mixlab/arch"
)

func initGPUTrainer(prog *ir.Program, cfg *ArchConfig, loadedWeights [][]float32) (GPUTrainer, error) {
	if !mlxAvailable() {
		return nil, fmt.Errorf("GPU training requires MLX backend; rebuild with: CGO_ENABLED=1 go build -tags mlx -o mixlab ./cmd/mixlab (requires macOS with Apple Silicon)")
	}
	return initMLXGPUTrainer(prog, cfg, loadedWeights)
}
