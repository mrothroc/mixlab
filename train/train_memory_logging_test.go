package train

import (
	"strings"
	"testing"
)

func TestMLXMemoryLoggingSamplesGPUWithoutDebugServer(t *testing.T) {
	for _, live := range []bool{false, true} {
		for _, available := range []bool{false, true} {
			var rt *telemetryRuntime
			if live {
				rt = &telemetryRuntime{state: newTelemetryState()}
				rt.state.update(telemetryUpdate{Step: 9, TotalSteps: 20})
			}
			calls := 0
			sample := func() *float64 {
				calls++
				if !available {
					return nil
				}
				v := 99.0
				return &v
			}
			for _, cadence := range []struct{ step, every int }{{0, 0}, {9, -1}, {1, 10}, {8, 10}} {
				if got := mlxMemoryTelemetryLine(cadence.step, cadence.every, rt, sample); got != "" || calls != 0 {
					t.Fatalf("off cadence sampled GPU: %q calls=%d", got, calls)
				}
			}
			for _, step := range []int{0, 9} {
				line := mlxMemoryTelemetryLine(step, 10, rt, sample)
				want := "gpu_util=n/a"
				if available {
					want = "gpu_util=99%"
				}
				for _, field := range []string{want, "mlx_active=", "mlx_cache=", "mlx_peak=", "rss="} {
					if !strings.Contains(line, field) {
						t.Fatalf("missing %q: %s", field, line)
					}
				}
				if live && !strings.Contains(line, "step 9/20") || !live && strings.Contains(line, "/0") {
					t.Fatal(line)
				}
			}
			if calls != 2 {
				t.Fatalf("sampler calls=%d want 2", calls)
			}
		}
	}
}
