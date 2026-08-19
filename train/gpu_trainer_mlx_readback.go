//go:build mlx && cgo && (darwin || linux)

package train

import (
	"fmt"

	"github.com/mrothroc/mixlab/gpu"
)

// Weight and output readback accessors for the MLX GPU trainer.

// ReadWeights reads all weight tensors back from the GPU trainer.
func (t *mlxGPUTrainer) ReadWeights() ([][]float32, error) {
	if err := t.FlushGPU(); err != nil {
		return nil, err
	}
	nWeights, err := gpu.TrainerNumWeights(t.handle)
	if err != nil {
		return nil, err
	}
	if nWeights != len(t.shapes) {
		return nil, fmt.Errorf("weight count mismatch: trainer=%d expected=%d", nWeights, len(t.shapes))
	}
	weights := make([][]float32, nWeights)
	for i := 0; i < nWeights; i++ {
		size, err := gpu.TrainerWeightSize(t.handle, i)
		if err != nil {
			return nil, fmt.Errorf("weight %d size: %w", i, err)
		}
		data := make([]float32, size)
		if err := gpu.TrainerReadWeight(t.handle, i, data); err != nil {
			return nil, fmt.Errorf("read weight %d: %w", i, err)
		}
		weights[i] = data
	}
	return weights, nil
}

func (t *mlxGPUTrainer) ReadWeightsGPU(indexes []int) ([][]float32, error) {
	if err := t.FlushGPU(); err != nil {
		return nil, err
	}
	weights := make([][]float32, len(indexes))
	for outIndex, weightIndex := range indexes {
		if weightIndex < 0 || weightIndex >= len(t.shapes) {
			return nil, fmt.Errorf("weight index %d out of range [0,%d)", weightIndex, len(t.shapes))
		}
		size, err := gpu.TrainerWeightSize(t.handle, weightIndex)
		if err != nil {
			return nil, fmt.Errorf("weight %d size: %w", weightIndex, err)
		}
		data := make([]float32, size)
		if err := gpu.TrainerReadWeight(t.handle, weightIndex, data); err != nil {
			return nil, fmt.Errorf("read weight %d: %w", weightIndex, err)
		}
		weights[outIndex] = data
	}
	return weights, nil
}

// ReadOutput reads a named output tensor cached by the last trainer step or eval.
func (t *mlxGPUTrainer) ReadOutput(name string, shape []int) ([]float32, error) {
	if err := t.FlushGPU(); err != nil {
		return nil, err
	}
	return gpu.TrainerReadOutput(t.handle, name, shape)
}

// ReadComponentLossesGPU returns the declared scalar component losses from the
// most recently collected training step without flushing a lookahead step.
func (t *mlxGPUTrainer) ReadComponentLossesGPU() (map[string]float64, error) {
	if !t.captureComponentLosses {
		return nil, fmt.Errorf("component loss capture is not enabled")
	}
	if len(t.componentLossOutputs) == 0 {
		return nil, nil
	}
	result := make(map[string]float64, len(t.componentLossOutputs))
	for _, name := range t.componentLossOutputs {
		out, err := gpu.TrainerReadCachedOutput(t.handle, name, []int{1})
		if err != nil {
			return nil, fmt.Errorf("read cached component loss %q: %w", name, err)
		}
		if len(out) != 1 {
			return nil, fmt.Errorf("cached component loss %q returned %d values, want 1", name, len(out))
		}
		result[name] = float64(out[0])
	}
	return result, nil
}
