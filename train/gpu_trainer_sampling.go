//go:build mlx && cgo && (darwin || linux)

package train

import (
	"fmt"

	"github.com/mrothroc/mixlab/gpu"
)

// Categorical-sampling trainer methods: draw one token per row from a named
// program output, on device, for RTD generator replacement and energy scoring.

func (t *mlxGPUTrainer) SampleObjectiveOutputCategoricalGPU(batch objectiveBatch, batchSize, seqLen int, outputName string, rows, vocab int, temperature float64, seed uint64) ([]int, error) {
	return t.sampleObjectiveOutputCategorical(batch, batchSize, seqLen, outputName, rows, vocab, temperature, seed, true)
}

func (t *mlxGPUTrainer) EvaluateGenerationGPU(xTok, yTok, positions []int, batchSize, seqLen int) ([]float32, error) {
	if len(positions) != batchSize {
		return nil, fmt.Errorf("generation positions=%d, want batch size %d", len(positions), batchSize)
	}
	if err := t.FlushGPU(); err != nil {
		return nil, err
	}
	inputs, err := t.makeObjectiveInputs(objectiveBatch{x: xTok, y: yTok}, batchSize, seqLen)
	if err != nil {
		return nil, err
	}
	positionData := make([]int32, batchSize)
	for i, position := range positions {
		if position < 0 || position >= seqLen {
			return nil, fmt.Errorf("generation_positions[%d]=%d out of range [0,%d)", i, position, seqLen)
		}
		positionData[i] = int32(i*seqLen + position)
	}
	inputs = append(inputs, gpu.TensorInput{
		Name: "generation_positions", DType: gpu.TensorInt32,
		Shape: []int{batchSize}, Data: positionData,
	})
	if _, err := gpu.TrainerEvaluateWithOutputs(t.handle, inputs, []string{"generation_logits"}); err != nil {
		return nil, err
	}
	return gpu.TrainerReadOutput(t.handle, "generation_logits", []int{batchSize, t.vocabSize})
}

func (t *mlxGPUTrainer) SampleObjectiveOutputCategoricalEagerGPU(batch objectiveBatch, batchSize, seqLen int, outputName string, rows, vocab int, temperature float64, seed uint64) ([]int, error) {
	return t.sampleObjectiveOutputCategorical(batch, batchSize, seqLen, outputName, rows, vocab, temperature, seed, false)
}

func (t *mlxGPUTrainer) SampleRTDGeneratorOutputCategoricalGPU(batch objectiveBatch, outputName string, rows, vocab int, temperature float64, seed uint64) ([]int, error) {
	return t.sampleRTDGeneratorOutputCategorical(batch, outputName, rows, vocab, temperature, seed, true)
}

func (t *mlxGPUTrainer) SampleRTDGeneratorOutputCategoricalEagerGPU(batch objectiveBatch, outputName string, rows, vocab int, temperature float64, seed uint64) ([]int, error) {
	return t.sampleRTDGeneratorOutputCategorical(batch, outputName, rows, vocab, temperature, seed, false)
}

func (t *mlxGPUTrainer) sampleObjectiveOutputCategorical(batch objectiveBatch, batchSize, seqLen int, outputName string, rows, vocab int, temperature float64, seed uint64, allowCompile bool) ([]int, error) {
	if batch.batchSizeOverride > 0 {
		batchSize = batch.batchSizeOverride
	}
	if err := t.FlushGPU(); err != nil {
		return nil, err
	}
	inputs, err := t.makeObjectiveInputs(batch, batchSize, seqLen)
	if err != nil {
		return nil, err
	}
	var raw []int32
	if allowCompile {
		raw, err = gpu.TrainerSampleCategoricalOutput(t.handle, inputs, outputName, rows, vocab, float32(temperature), seed)
	} else {
		raw, err = gpu.TrainerSampleCategoricalOutputEager(t.handle, inputs, outputName, rows, vocab, float32(temperature), seed)
	}
	if err != nil {
		return nil, err
	}
	out := make([]int, len(raw))
	for i, v := range raw {
		out[i] = int(v)
	}
	return out, nil
}

func (t *mlxGPUTrainer) sampleRTDGeneratorOutputCategorical(batch objectiveBatch, outputName string, rows, vocab int, temperature float64, seed uint64, allowCompile bool) ([]int, error) {
	if err := t.FlushGPU(); err != nil {
		return nil, err
	}
	inputs, err := t.makeRTDGeneratorInputs(batch)
	if err != nil {
		return nil, err
	}
	var raw []int32
	if allowCompile {
		raw, err = gpu.TrainerSampleCategoricalOutput(t.handle, inputs, outputName, rows, vocab, float32(temperature), seed)
	} else {
		raw, err = gpu.TrainerSampleCategoricalOutputEager(t.handle, inputs, outputName, rows, vocab, float32(temperature), seed)
	}
	if err != nil {
		return nil, err
	}
	out := make([]int, len(raw))
	for i, v := range raw {
		out[i] = int(v)
	}
	return out, nil
}
