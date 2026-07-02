//go:build mlx && cgo && (darwin || linux)

package train

import "github.com/mrothroc/mixlab/gpu"

// Categorical-sampling trainer methods: draw one token per row from a named
// program output, on device, for RTD generator replacement and energy scoring.

func (t *mlxGPUTrainer) SampleObjectiveOutputCategoricalGPU(batch objectiveBatch, batchSize, seqLen int, outputName string, rows, vocab int, temperature float64, seed uint64) ([]int, error) {
	return t.sampleObjectiveOutputCategorical(batch, batchSize, seqLen, outputName, rows, vocab, temperature, seed, true)
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
