package train

import (
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/gpu"
)

func handleMLXMemoryControls(name string, step, logEvery, clearEvery int, telemetry *telemetryRuntime) {
	if clearEvery > 0 && (step+1)%clearEvery == 0 {
		gpu.ClearMemoryCache()
	}
	if logEvery <= 0 {
		return
	}
	if step != 0 && (step+1)%logEvery != 0 {
		return
	}
	if telemetry != nil && telemetry.state != nil {
		fmt.Printf("  [%s] %s\n", name, formatTelemetryLine(telemetry.state.snapshot(true)))
		return
	}
	stats := gpu.MemoryStatsSnapshot()
	fmt.Printf("  [%s] [telemetry] step %d gpu_util=n/a mlx_active=%s mlx_cache=%s mlx_peak=%s\n",
		name, step, formatMiB(stats.ActiveBytes), formatMiB(stats.CacheBytes), formatMiB(stats.PeakBytes))
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
