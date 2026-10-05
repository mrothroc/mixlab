package train

import (
	"errors"
	"strings"
	"testing"
)

func TestGridMemoryPreflightPolicy(t *testing.T) {
	for _, raw := range []string{"", "0", "1", "bad"} {
		for _, cuda := range []bool{false, true} {
			t.Setenv(gridMemoryPreflightEnv, raw)
			got, err := gridMemoryPreflightEnabled(cuda)
			if (err != nil) != (raw == "bad") {
				t.Fatalf("%q: %v", raw, err)
			}
			if err == nil && got != (raw == "1" || (raw == "" && cuda)) {
				t.Fatalf("%q cuda=%v got=%v", raw, cuda, got)
			}
		}
	}
}

func TestGridMemoryFailureDiagnostic(t *testing.T) {
	cfg := gridPhaseConfig(t)
	p := mlxMemoryLimitPlan{DedicatedDevice: true, DeviceName: "test GPU", DeviceMemoryBytes: 24 << 30, ApplyMemoryLimit: true, MemoryLimitBytes: 22 << 30}
	oom := errors.New("cudaMallocAsync failed: out of memory")
	err := annotateGridMemoryError(oom, "preflight before full validation", cfg, p)
	for _, want := range []string{"preflight before full validation", "test GPU", "vram=", "batch_size=2", "grid=8x8x2", "observed peak is not an estimate"} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("missing %q: %v", want, err)
		}
	}
	if !errors.Is(err, oom) {
		t.Fatal("lost underlying error")
	}
	other := errors.New("invalid shape")
	if annotateGridMemoryError(other, "training", cfg, p) != other {
		t.Fatal("non-memory error changed")
	}
	if annotateGridMemoryError(nil, "training", cfg, p) != nil {
		t.Fatal("nil error changed")
	}
}

func TestGridMemoryHintUsesInitialAvailability(t *testing.T) {
	if got := gridMemoryHint(mlxMemoryLimitPlan{}); strings.Contains(got, "CUDA") {
		t.Fatal(got)
	}
	for _, tc := range []struct {
		free  uint64
		other bool
	}{
		{0, false}, {23 << 30, false}, {20 << 30, false}, {12 << 30, true},
	} {
		p := mlxMemoryLimitPlan{DedicatedDevice: true, DeviceMemoryBytes: 24 << 30, DeviceFreeBytes: tc.free}
		got := gridMemoryHint(p)
		if strings.Contains(got, "other GPU processes") != tc.other {
			t.Fatalf("free=%d: %s", tc.free, got)
		}
		if !strings.Contains(got, "allocator/graph reservations") {
			t.Fatal(got)
		}
	}
}
