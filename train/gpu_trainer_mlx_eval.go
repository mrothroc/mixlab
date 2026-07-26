//go:build mlx && cgo && (darwin || linux)

package train

import (
	"fmt"

	"github.com/mrothroc/mixlab/gpu"
)

// EvaluateGPU runs a forward pass without gradients and returns the loss.
func (t *mlxGPUTrainer) EvaluateGPU(xTok, yTok []int, batchSize, seqLen int) (float32, error) {
	return evaluateTokensViaObjectiveGPU(t, xTok, yTok, batchSize, seqLen)
}

// EvaluateObjectiveGPU runs an objective-batch forward pass without gradients and returns the loss.
func (t *mlxGPUTrainer) EvaluateObjectiveGPU(batch objectiveBatch, batchSize, seqLen int) (float32, error) {
	if batch.batchSizeOverride > 0 {
		batchSize = batch.batchSizeOverride
	}
	if err := t.FlushGPU(); err != nil {
		return 0, err
	}
	inputs, err := t.makeObjectiveInputs(batch, batchSize, seqLen)
	if err != nil {
		return 0, err
	}
	return t.evaluatePreparedInputs(inputs)
}

func (t *mlxGPUTrainer) EvaluateObjectiveGPUWithOutputs(batch objectiveBatch, batchSize, seqLen int, outputNames []string) (float32, error) {
	if batch.batchSizeOverride > 0 {
		batchSize = batch.batchSizeOverride
	}
	if err := t.FlushGPU(); err != nil {
		return 0, err
	}
	inputs, err := t.makeObjectiveInputs(batch, batchSize, seqLen)
	if err != nil {
		return 0, err
	}
	if len(outputNames) == 0 {
		return t.evaluatePreparedInputs(inputs)
	}
	loss, err := gpu.TrainerEvaluateWithOutputs(t.handle, inputs, outputNames)
	if err != nil || t.evalLossOutputName == "" || t.evalLossOutputName == "loss" {
		return loss, err
	}
	out, err := gpu.TrainerReadOutput(t.handle, t.evalLossOutputName, []int{1})
	if err != nil {
		return 0, err
	}
	if len(out) != 1 {
		return 0, fmt.Errorf("eval output %q returned %d values, want 1", t.evalLossOutputName, len(out))
	}
	return out[0], nil
}

func (t *mlxGPUTrainer) CompileStatsGPU() (gpu.TrainerCompileStats, error) {
	return gpu.TrainerCompileStatsSnapshot(t.handle)
}

func (t *mlxGPUTrainer) OptimizerStatsGPU() (gpu.TrainerOptimizerStats, error) {
	return gpu.TrainerOptimizerStatsSnapshot(t.handle)
}

// EvaluateObjectiveTrainingLossGPU evaluates the graph's optimizer loss output
// directly, even when a separate dense eval_loss output is available.
func (t *mlxGPUTrainer) EvaluateObjectiveTrainingLossGPU(batch objectiveBatch, batchSize, seqLen int) (float32, error) {
	if batch.batchSizeOverride > 0 {
		batchSize = batch.batchSizeOverride
	}
	if err := t.FlushGPU(); err != nil {
		return 0, err
	}
	inputs, err := t.makeObjectiveInputs(batch, batchSize, seqLen)
	if err != nil {
		return 0, err
	}
	return gpu.TrainerEvaluate(t.handle, inputs)
}

func (t *mlxGPUTrainer) evaluatePreparedInputs(inputs []gpu.TensorInput) (float32, error) {
	loss, err := gpu.TrainerEvaluate(t.handle, inputs)
	if err != nil || t.evalLossOutputName == "" || t.evalLossOutputName == "loss" {
		return loss, err
	}
	out, err := gpu.TrainerReadOutput(t.handle, t.evalLossOutputName, []int{1})
	if err != nil {
		return 0, err
	}
	if len(out) != 1 {
		return 0, fmt.Errorf("eval output %q returned %d values, want 1", t.evalLossOutputName, len(out))
	}
	return out[0], nil
}

// EvaluatePerTokenGPU runs a forward pass without gradients and returns per-token NLLs.
func (t *mlxGPUTrainer) EvaluatePerTokenGPU(xTok, yTok []int, batchSize, seqLen int) ([]float32, error) {
	if err := t.FlushGPU(); err != nil {
		return nil, err
	}
	inputs, err := t.makeObjectiveInputs(objectiveBatch{x: xTok, y: yTok}, batchSize, seqLen)
	if err != nil {
		return nil, err
	}
	return gpu.TrainerEvaluatePerToken(t.handle, inputs)
}

// EvaluateLoRATTTGPU runs per-batch LoRA TTT without mutating the base trainer weights.
func (t *mlxGPUTrainer) EvaluateLoRATTTGPU(xTok, yTok []int, batchSize, seqLen, tttSteps int, tttLR float32, tttRank int) (float32, error) {
	if err := t.FlushGPU(); err != nil {
		return 0, err
	}
	inputs, err := t.makeObjectiveInputs(objectiveBatch{x: xTok, y: yTok}, batchSize, seqLen)
	if err != nil {
		return 0, err
	}
	return gpu.TrainerEvaluateLoRA(t.handle, inputs, tttRank, tttSteps, tttLR)
}
