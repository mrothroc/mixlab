package train

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"
)

func TestParseIORegGPUUtilPercent(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want *float64
	}{
		{
			name: "single",
			in:   `| |   "PerformanceStatistics" = {"Device Utilization %"=95}`,
			want: ptrFloat64(95),
		},
		{
			name: "multiple_uses_max",
			in:   `"Device Utilization %"=12 "Device Utilization %" = 87`,
			want: ptrFloat64(87),
		},
		{
			name: "decimal",
			in:   `"Device Utilization %"=42.5`,
			want: ptrFloat64(42.5),
		},
		{
			name: "missing",
			in:   `"PerformanceStatistics" = {}`,
			want: nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := parseIORegGPUUtilPercent(tt.in)
			if tt.want == nil {
				if got != nil {
					t.Fatalf("got %v, want nil", *got)
				}
				return
			}
			if got == nil || *got != *tt.want {
				t.Fatalf("got %v, want %v", got, *tt.want)
			}
		})
	}
}

func TestTelemetrySnapshotJSONShape(t *testing.T) {
	state := newTelemetryState()
	componentLosses := map[string]float64{"invariance_loss": 0.25}
	masking := &telemetryMasking{Unit: "whole_word", TargetProb: 0.15, EligibleTokens: 100, BudgetTokens: 15, SelectedTokens: 14, RealizedRate: 0.14, CandidateGroups: 70, SelectedGroups: 10, MeanSelectedGroupSize: 1.4, MaxSelectedGroupSize: 3, UnderfilledRows: 1}
	state.update(telemetryUpdate{
		Model:         "tiny",
		Step:          7,
		TotalSteps:    100,
		Loss:          1.25,
		HasLoss:       true,
		ValLoss:       1.5,
		HasValLoss:    true,
		LR:            0.001,
		Objective:     "causal",
		SeqLen:        16,
		BatchTokens:   64,
		Elapsed:       3 * time.Second,
		SteadyElapsed: 2 * time.Second,
		TokensPerSec:  128,
		Timing: telemetryTiming{
			DataMS:       1,
			GPUMS:        2,
			ValidationMS: 3,
			LogMS:        4,
		},
		HasTiming:             true,
		ComponentLosses:       componentLosses,
		OptimizerSteps:        7,
		SkippedOptimizerSteps: 1,
		ConsecutiveSkipped:    1,
		OptimizerStepSkipped:  true,
		Masking:               masking,
	})
	// telemetry state owns the serialized snapshot rather than retaining a
	// caller-owned map that training code can mutate on the next step.
	componentLosses["invariance_loss"] = 999
	masking.Unit = "mutated"
	snap := state.snapshot(false)
	if snap.Model != "tiny" || snap.Step != 7 || snap.TotalSteps != 100 {
		t.Fatalf("bad run fields: %+v", snap.telemetryRunState)
	}
	if snap.Loss == nil || *snap.Loss != 1.25 {
		t.Fatalf("loss=%v, want 1.25", snap.Loss)
	}
	if snap.ValLoss == nil || *snap.ValLoss != 1.5 {
		t.Fatalf("val_loss=%v, want 1.5", snap.ValLoss)
	}
	if snap.GPUUtilPercent != nil {
		t.Fatalf("snapshot(false) sampled gpu util: %v", *snap.GPUUtilPercent)
	}
	if got := snap.ComponentLosses["invariance_loss"]; got != 0.25 {
		t.Fatalf("component loss=%g, want 0.25", got)
	}
	if snap.OptimizerSteps != 7 || snap.SkippedOptimizerSteps != 1 || snap.ConsecutiveSkipped != 1 || !snap.OptimizerStepSkipped {
		t.Fatalf("optimizer telemetry=%+v", snap.telemetryRunState)
	}
	if snap.Masking == nil || snap.Masking.Unit != "whole_word" || snap.Masking.RealizedRate != 0.14 || snap.Masking.MeanSelectedGroupSize != 1.4 {
		t.Fatalf("masking telemetry=%+v", snap.Masking)
	}
	raw, err := json.Marshal(snap)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	for _, want := range []string{`"model":"tiny"`, `"tokens_per_sec":128`, `"component_losses":{"invariance_loss":0.25}`, `"masking":{"unit":"whole_word"`, `"realized_rate":0.14`, `"optimizer_steps":7`, `"skipped_optimizer_steps":1`, `"consecutive_skipped_optimizer_steps":1`, `"optimizer_step_skipped":true`, `"mlx":`, `"host":`, `"rss_bytes":`} {
		if !strings.Contains(string(raw), want) {
			t.Fatalf("snapshot JSON missing %s: %s", want, raw)
		}
	}
}

func TestTelemetryOmitsInactiveComponentLosses(t *testing.T) {
	state := newTelemetryState()
	state.update(telemetryUpdate{Model: "no-op"})
	raw, err := json.Marshal(state.snapshot(false))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if strings.Contains(string(raw), "component_losses") {
		t.Fatalf("inactive component losses serialized: %s", raw)
	}
	if strings.Contains(string(raw), "masking") {
		t.Fatalf("inactive masking telemetry serialized: %s", raw)
	}
}

func TestMLMMaskStatsTelemetryDerivedValues(t *testing.T) {
	masking := (mlmMaskStats{Unit: "whole_word", TargetProb: 0.15, EligibleTokens: 20, SelectedTokens: 3, SelectedGroups: 2, SelectedGroupTokenTotal: 3}).telemetry()
	if masking == nil || masking.RealizedRate != 0.15 || masking.MeanSelectedGroupSize != 1.5 {
		t.Fatalf("masking=%+v", masking)
	}
}

func TestTelemetryRuntimeWritesJSONL(t *testing.T) {
	path := t.TempDir() + "/telemetry.jsonl"
	rt, err := newTelemetryRuntime("", path)
	if err != nil {
		t.Fatalf("newTelemetryRuntime: %v", err)
	}
	rt.state.update(telemetryUpdate{
		Model:        "jsonl",
		Step:         1,
		TotalSteps:   2,
		Loss:         0.75,
		HasLoss:      true,
		LR:           0.01,
		Objective:    "mlm",
		SeqLen:       8,
		BatchTokens:  16,
		TokensPerSec: 32,
	})
	if err := rt.writeSnapshot(false); err != nil {
		t.Fatalf("writeSnapshot: %v", err)
	}
	rt.Close()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read telemetry: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	if len(lines) != 1 {
		t.Fatalf("lines=%d, want 1: %q", len(lines), raw)
	}
	var snap telemetrySnapshot
	if err := json.Unmarshal([]byte(lines[0]), &snap); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if snap.Model != "jsonl" || snap.Step != 1 || snap.Loss == nil || *snap.Loss != 0.75 {
		t.Fatalf("bad snapshot: %+v", snap)
	}
}

func TestTelemetryDebugServerEndpoints(t *testing.T) {
	rt := &telemetryRuntime{state: newTelemetryState()}
	rt.state.update(telemetryUpdate{Model: "http", Step: 3, TotalSteps: 4})
	mux := rt.debugMux()
	for _, path := range []string{"/debug/mixlab/telemetry", "/debug/vars", "/debug/pprof/"} {
		req := httptest.NewRequest(http.MethodGet, path, nil)
		rec := httptest.NewRecorder()
		mux.ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("GET %s status=%d, want 200", path, rec.Code)
		}
	}
}

func TestTelemetryRuntimeDisabledDoesNotStartServer(t *testing.T) {
	called := false
	err := withTelemetryRuntime(TrainOptions{}, func(opts TrainOptions) error {
		called = true
		if opts.telemetry != nil {
			t.Fatal("telemetry runtime set when no telemetry options were provided")
		}
		return nil
	})
	if err != nil {
		t.Fatalf("withTelemetryRuntime: %v", err)
	}
	if !called {
		t.Fatal("callback was not called")
	}
}

func TestFormatTelemetryLine(t *testing.T) {
	util := 95.0
	line := formatTelemetryLine(telemetrySnapshot{
		telemetryRunState: telemetryRunState{
			Step: 5, TotalSteps: 10, TokensPerSec: 1234,
			OptimizerSteps: 4, SkippedOptimizerSteps: 1, ConsecutiveSkipped: 1, OptimizerStepSkipped: true,
		},
		MLX:            telemetryMLX{ActiveBytes: 1024 * 1024, CacheBytes: 2 * 1024 * 1024, PeakBytes: 3 * 1024 * 1024},
		Host:           telemetryHost{RSSBytes: 4 * 1024 * 1024},
		GPUUtilPercent: &util,
	})
	for _, want := range []string{"[telemetry]", "step 5/10", "tok/s=1234", "gpu_util=95%", "mlx_active=1.0MiB", "rss=4.0MiB", "optimizer_steps=4", "skipped_optimizer_steps=1", "consecutive_skips=1", "optimizer_step_skipped=true"} {
		if !strings.Contains(line, want) {
			t.Fatalf("line missing %q: %s", want, line)
		}
	}
}

func ptrFloat64(v float64) *float64 {
	return &v
}
