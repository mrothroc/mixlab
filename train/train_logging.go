package train

import (
	"fmt"
	"strings"
	"time"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/gpu"
)

func formatMultiheadHeadsForLog(heads []arch.MultiheadHeadSpec) string {
	parts := make([]string, 0, len(heads))
	for _, h := range heads {
		parts = append(parts, fmt.Sprintf("%s:%s weight=%g output=%s agg=%s",
			h.Name, h.Objective, h.LossWeight, h.OutputHead, h.LayerAggregation))
	}
	return strings.Join(parts, ", ")
}

func handleMLXMemoryControls(name string, step, logEvery, clearEvery int, telemetry *telemetryRuntime) {
	if clearEvery > 0 && (step+1)%clearEvery == 0 {
		gpu.ClearMemoryCache()
	}
	if line := mlxMemoryTelemetryLine(step, logEvery, telemetry, sampleGPUUtilPercent); line != "" {
		fmt.Printf("  [%s] %s\n", name, line)
	}
}

// Environment-only logging uses the same gauges as the HTTP telemetry endpoint.
// Injecting the sampler keeps cadence and unavailable-device tests portable.
func mlxMemoryTelemetryLine(step, logEvery int, telemetry *telemetryRuntime, sampleGPU func() *float64) string {
	if logEvery <= 0 {
		return ""
	}
	if step != 0 && (step+1)%logEvery != 0 {
		return ""
	}
	state := newTelemetryState()
	state.s.Step = step
	if telemetry != nil && telemetry.state != nil {
		state = telemetry.state
	}
	snapshot := state.snapshot(false)
	snapshot.GPUUtilPercent = sampleGPU()
	return formatTelemetryLine(snapshot)
}

func formatMiB(bytes uint64) string {
	return fmt.Sprintf("%.1fMiB", float64(bytes)/(1024.0*1024.0))
}

func formatProgressTiming(elapsed, steadyElapsed time.Duration, stepsForRate, step, totalSteps int) string {
	if step < 1 || totalSteps <= 0 || stepsForRate < 1 || steadyElapsed <= 0 {
		return fmt.Sprintf("(%.1fs)", elapsed.Seconds())
	}
	// ETA uses steady-state rate (post-warmup) so the one-time compile cost
	// doesn't dominate early estimates.
	avgStepDuration := steadyElapsed / time.Duration(stepsForRate)
	remainingSteps := totalSteps - (step + 1)
	if remainingSteps < 0 {
		remainingSteps = 0
	}
	eta := time.Duration(remainingSteps) * avgStepDuration
	return fmt.Sprintf("(%.1fs, ~%s remaining)", elapsed.Seconds(), eta.Round(time.Second))
}
