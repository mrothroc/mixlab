package train

import (
	"fmt"
	"math"
	"os"
	"strings"

	"github.com/mrothroc/mixlab/data"
	"github.com/mrothroc/mixlab/gpu"
)

const gridMemoryPreflightEnv = "MIXLAB_GRID_MEMORY_PREFLIGHT"

func gridMemoryPreflightEnabled(dedicated bool) (bool, error) {
	switch strings.TrimSpace(os.Getenv(gridMemoryPreflightEnv)) {
	case "":
		return dedicated, nil
	case "1":
		return true, nil
	case "0":
		return false, nil
	default:
		return false, fmt.Errorf("%s must be 0 or 1", gridMemoryPreflightEnv)
	}
}

func gridMemoryDiagnostic(phase string, plan mlxMemoryLimitPlan) string {
	m := gpu.MemoryStatsSnapshot()
	d, _ := gpu.DeviceMemoryInfo()
	line := fmt.Sprintf("grid memory phase=%s active=%s cache=%s peak=%s device_free=%s initial_device_free=%s configured_limit=%s",
		phase, formatMiB(m.ActiveBytes), formatMiB(m.CacheBytes), formatMiB(m.PeakBytes),
		formatOptionalMiB(d.FreeBytes), formatOptionalMiB(plan.DeviceFreeBytes), formatAppliedLimit(plan.ApplyMemoryLimit, plan.MemoryLimitBytes))
	if c, ok := gpu.CUDAMemorySnapshot(); ok {
		line += fmt.Sprintf(" cuda_pool_reserved=%s cuda_pool_used=%s cuda_graph_reserved=%s cuda_graph_used=%s",
			formatMiB(c.PoolReservedBytes), formatMiB(c.PoolUsedBytes), formatMiB(c.GraphReservedBytes), formatMiB(c.GraphUsedBytes))
	}
	return line
}

func annotateGridMemoryError(err error, phase string, cfg *ArchConfig, plan mlxMemoryLimitPlan) error {
	if err == nil || !isMLXOutOfMemoryError(err) {
		return err
	}
	return fmt.Errorf("dense_regression memory failure during %s: device=%q vram=%s batch_size=%d grid=%dx%dx%d; %s. The requested allocation did not fit; the observed peak is not an estimate of total required memory. %s: %w",
		phase, plan.DeviceName, formatOptionalMiB(plan.DeviceMemoryBytes), cfg.Training.BatchSize,
		cfg.InputAdapter.Height, cfg.InputAdapter.Width, cfg.InputAdapter.Channels,
		gridMemoryDiagnostic(phase, plan), gridMemoryHint(plan), err)
}

func gridMemoryHint(plan mlxMemoryLimitPlan) string {
	if !plan.DedicatedDevice {
		return "Inspect MLX allocations and configured memory/cache limits"
	}
	hint := "Inspect CUDA allocator/graph reservations and configured memory/cache limits"
	// Low initial availability is evidence of pre-existing device pressure, not
	// proof that another process caused the failed allocation.
	if plan.DeviceMemoryBytes > 0 && plan.DeviceFreeBytes > 0 && plan.DeviceFreeBytes < plan.DeviceMemoryBytes-plan.DeviceMemoryBytes/5 {
		hint += "; device availability was already below 80% at startup, also check other GPU processes"
	}
	return hint
}

// Exercise both cold and cached steps before full validation. Grid graphs have
// no stochastic ops; the loader/sampler is not advanced. CPU snapshots avoid
// retaining a second copy of model/optimizer state on the GPU during the probe.
func preflightGridTraining(tr GPUTrainer, shapes []WeightShape, batch data.GridBatch, step int, lr float32) error {
	stateReader, ok := tr.(gpuTrainerStateReader)
	if !ok {
		return fmt.Errorf("grid memory preflight requires optimizer snapshot support")
	}
	restorer, ok := tr.(gpuTrainerStateRestorer)
	if !ok {
		return fmt.Errorf("grid memory preflight requires optimizer restore support")
	}
	setter, ok := tr.(gpuTrainingStepSetter)
	if !ok {
		return fmt.Errorf("grid memory preflight requires training-step restore support")
	}
	weights, err := readTrainerWeights(tr)
	if err != nil {
		return err
	}
	if len(weights) != len(shapes) {
		return fmt.Errorf("grid memory preflight weight count mismatch")
	}
	state, err := stateReader.ReadTrainerState()
	if err != nil {
		return err
	}
	for i := 0; i < 3; i++ {
		if err := submitPreparedStepGPU(tr, objectiveBatch{grid: &batch}, batch.BatchSize, 0, lr); err != nil {
			return fmt.Errorf("training probe %d: %w", i+1, err)
		}
		loss, err := tr.CollectLossGPU()
		if err != nil {
			return fmt.Errorf("training probe %d: %w", i+1, err)
		}
		if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
			return fmt.Errorf("training probe %d produced non-finite loss", i+1)
		}
	}
	// On any probe error the caller aborts the run and closes the trainer; never
	// save or continue a partially updated model. Only success restores and runs.
	for i, s := range shapes {
		if err := tr.SetWeightGPU(s.Name, weights[i]); err != nil {
			return fmt.Errorf("restore preflight weight %s: %w", s.Name, err)
		}
	}
	if err := restorer.RestoreTrainerState(state); err != nil {
		return fmt.Errorf("restore preflight optimizer: %w", err)
	}
	return setter.SetTrainingStepGPU(step)
}
