package train

import (
	"fmt"

	ir "github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/gpu"
)

func initGPUTrainer(
	prog *ir.Program,
	cfg *ArchConfig,
	loadedWeights [][]float32,
	optimizerOverride func(gpu.TrainerOptimizerSpec, []WeightShape) (gpu.TrainerOptimizerSpec, error),
) (GPUTrainer, error) {
	if !mlxAvailable() {
		return nil, fmt.Errorf("GPU training requires MLX backend; rebuild with: CGO_ENABLED=1 go build -tags mlx -o mixlab ./cmd/mixlab (macOS: brew install mlx; Linux: see docker/README.md)")
	}
	return initMLXGPUTrainer(prog, cfg, loadedWeights, optimizerOverride)
}

// readTrainerWeights reads weights from a trainer via the weight-reading interface.
// Falls back gracefully if the trainer doesn't support weight reading.
func readTrainerWeights(trainer any) ([][]float32, error) {
	type weightReader interface {
		ReadWeights() ([][]float32, error)
	}
	if wr, ok := trainer.(weightReader); ok {
		return wr.ReadWeights()
	}
	return nil, fmt.Errorf("trainer does not support weight reading; ensure you are using the MLX backend")
}

func readTrainerOutput(trainer GPUTrainer, name string, shape []int) ([]float32, error) {
	type outputReader interface {
		ReadOutput(name string, shape []int) ([]float32, error)
	}
	if or, ok := trainer.(outputReader); ok {
		return or.ReadOutput(name, shape)
	}
	return nil, fmt.Errorf("trainer does not support reading named outputs; ensure you are using the MLX backend")
}
