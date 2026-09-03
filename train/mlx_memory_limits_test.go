package train

import (
	"bytes"
	"errors"
	"runtime"
	"strings"
	"testing"
)

func TestDefaultMLXMemoryLimits(t *testing.T) {
	tests := []struct {
		name       string
		totalRAM   uint64
		wantMemory uint64
		wantCache  uint64
	}{
		{
			name:       "sixteen_gib",
			totalRAM:   16 * mlxBytesPerGiB,
			wantMemory: 8 * mlxBytesPerGiB,
			wantCache:  2 * mlxBytesPerGiB,
		},
		{
			name:       "sixty_four_gib",
			totalRAM:   64 * mlxBytesPerGiB,
			wantMemory: 48 * mlxBytesPerGiB,
			wantCache:  8 * mlxBytesPerGiB,
		},
		{
			name:       "one_twenty_eight_gib",
			totalRAM:   128 * mlxBytesPerGiB,
			wantMemory: 96 * mlxBytesPerGiB,
			wantCache:  16 * mlxBytesPerGiB,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			gotMemory, gotCache, ok := defaultMLXMemoryLimits(tc.totalRAM)
			if !ok {
				t.Fatal("defaultMLXMemoryLimits returned ok=false")
			}
			if gotMemory != tc.wantMemory || gotCache != tc.wantCache {
				t.Fatalf("limits=(%d,%d), want (%d,%d)", gotMemory, gotCache, tc.wantMemory, tc.wantCache)
			}
		})
	}
}

func TestConfigureMLXMemoryLimitsWritesToSelectedStream(t *testing.T) {
	clearMLXMemoryLimitEnv(t)
	t.Setenv(mlxDisableDefaultMemoryLimitsEnv, "1")
	t.Setenv(mlxCacheLimitMBEnv, "64")
	var out bytes.Buffer
	if _, err := configureMLXMemoryLimitsTo("generate", &out); err != nil {
		t.Fatal(err)
	}
	if got := out.String(); !strings.Contains(got, "[generate] MLX memory limits") || !strings.Contains(got, "cache=64.0MiB") {
		t.Fatalf("memory-limit diagnostic=%q", got)
	}
}

func TestResolveMLXMemoryLimitPlanDefaults(t *testing.T) {
	clearMLXMemoryLimitEnv(t)
	plan, err := resolveMLXMemoryLimitPlan(mlxMemoryCapacity{HostRAMBytes: 64 * mlxBytesPerGiB})
	if err != nil {
		t.Fatalf("resolveMLXMemoryLimitPlan: %v", err)
	}
	if !plan.AutoDefault || !plan.ApplyMemoryLimit || !plan.ApplyCacheLimit {
		t.Fatalf("plan did not apply auto defaults: %+v", plan)
	}
	if plan.MemoryLimitBytes != 48*mlxBytesPerGiB {
		t.Fatalf("memory=%d, want %d", plan.MemoryLimitBytes, 48*mlxBytesPerGiB)
	}
	if plan.CacheLimitBytes != 8*mlxBytesPerGiB {
		t.Fatalf("cache=%d, want %d", plan.CacheLimitBytes, 8*mlxBytesPerGiB)
	}
}

func TestResolveMLXMemoryLimitPlanCUDAUsesDeviceVRAM(t *testing.T) {
	clearMLXMemoryLimitEnv(t)
	const (
		hostRAM      = 512 * mlxBytesPerGiB
		deviceVRAM   = 23034 * mlxBytesPerMiB
		freeVRAM     = 22800 * mlxBytesPerMiB
		currentLimit = 21435 * mlxBytesPerMiB
	)
	plan, err := resolveMLXMemoryLimitPlan(mlxMemoryCapacity{
		HostRAMBytes:          hostRAM,
		DeviceName:            "NVIDIA L4",
		DeviceMemoryBytes:     deviceVRAM,
		DeviceFreeMemoryBytes: freeVRAM,
		CurrentMemoryLimit:    currentLimit,
		DedicatedDeviceMemory: true,
	})
	if err != nil {
		t.Fatalf("resolveMLXMemoryLimitPlan: %v", err)
	}
	if plan.MemoryLimitBytes != currentLimit {
		t.Fatalf("memory=%s, want existing device limit %s", formatMiB(plan.MemoryLimitBytes), formatMiB(currentLimit))
	}
	if plan.MemoryLimitBytes > deviceVRAM {
		t.Fatalf("memory=%s exceeds device VRAM=%s", formatMiB(plan.MemoryLimitBytes), formatMiB(deviceVRAM))
	}
	if plan.MemoryLimitBytes == hostRAM-hostRAM/4 {
		t.Fatalf("memory incorrectly derived from host RAM: %s", formatMiB(plan.MemoryLimitBytes))
	}
	if plan.CacheLimitBytes != freeVRAM/8 {
		t.Fatalf("cache=%s, want %s", formatMiB(plan.CacheLimitBytes), formatMiB(freeVRAM/8))
	}
}

func TestResolveMLXMemoryLimitPlanCUDAReplacesOversizedRuntimeLimit(t *testing.T) {
	clearMLXMemoryLimitEnv(t)
	const deviceVRAM = 24 * mlxBytesPerGiB
	plan, err := resolveMLXMemoryLimitPlan(mlxMemoryCapacity{
		HostRAMBytes:          512 * mlxBytesPerGiB,
		DeviceName:            "NVIDIA GPU",
		DeviceMemoryBytes:     deviceVRAM,
		CurrentMemoryLimit:    384 * mlxBytesPerGiB,
		DedicatedDeviceMemory: true,
	})
	if err != nil {
		t.Fatalf("resolveMLXMemoryLimitPlan: %v", err)
	}
	if want := deviceVRAM * 3 / 4; plan.MemoryLimitBytes != want {
		t.Fatalf("memory=%s, want VRAM fallback %s", formatMiB(plan.MemoryLimitBytes), formatMiB(want))
	}
}

func TestResolveMLXMemoryLimitPlanCUDAHonorsExplicitCacheLimit(t *testing.T) {
	clearMLXMemoryLimitEnv(t)
	t.Setenv(mlxCacheLimitMBEnv, "1024")
	plan, err := resolveMLXMemoryLimitPlan(mlxMemoryCapacity{
		HostRAMBytes:          512 * mlxBytesPerGiB,
		DeviceName:            "NVIDIA L4",
		DeviceMemoryBytes:     24 * mlxBytesPerGiB,
		CurrentMemoryLimit:    21 * mlxBytesPerGiB,
		DedicatedDeviceMemory: true,
	})
	if err != nil {
		t.Fatalf("resolveMLXMemoryLimitPlan: %v", err)
	}
	if !plan.CacheLimitEnv || plan.CacheLimitBytes != 1024*mlxBytesPerMiB {
		t.Fatalf("explicit CUDA cache limit was not preserved: %+v", plan)
	}
}

func TestResolveMLXMemoryLimitPlanCUDAMissingDeviceInfoDoesNotUseHostRAM(t *testing.T) {
	clearMLXMemoryLimitEnv(t)
	plan, err := resolveMLXMemoryLimitPlan(mlxMemoryCapacity{
		HostRAMBytes:          512 * mlxBytesPerGiB,
		DedicatedDeviceMemory: true,
	})
	if err != nil {
		t.Fatalf("resolveMLXMemoryLimitPlan: %v", err)
	}
	if plan.ApplyMemoryLimit || plan.ApplyCacheLimit || plan.AutoDefault {
		t.Fatalf("CUDA plan used host RAM after device discovery failed: %+v", plan)
	}
}

func TestMLXUnappliedLimitNotice(t *testing.T) {
	for _, tc := range []struct {
		name string
		plan mlxMemoryLimitPlan
		want string
	}{
		{
			name: "cuda device memory unreadable",
			plan: mlxMemoryLimitPlan{DedicatedDevice: true},
			want: "device memory unavailable, retaining MLX defaults",
		},
		{
			name: "operator opted out",
			plan: mlxMemoryLimitPlan{DedicatedDevice: true, DefaultDisabled: true},
		},
		{
			name: "device memory known",
			plan: mlxMemoryLimitPlan{DedicatedDevice: true, DeviceMemoryBytes: 24 * mlxBytesPerGiB},
		},
		{
			name: "unified memory needs no notice",
			plan: mlxMemoryLimitPlan{},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := mlxUnappliedLimitNotice(tc.plan); got != tc.want {
				t.Fatalf("notice=%q, want %q", got, tc.want)
			}
		})
	}
}

func TestConfigureMLXMemoryLimitsStaysQuietWhenDefaultsDisabled(t *testing.T) {
	clearMLXMemoryLimitEnv(t)
	t.Setenv(mlxDisableDefaultMemoryLimitsEnv, "1")
	var out bytes.Buffer
	if _, err := configureMLXMemoryLimitsTo("arch", &out); err != nil {
		t.Fatal(err)
	}
	if got := out.String(); got != "" {
		t.Fatalf("opting out of default limits still printed: %q", got)
	}
}

func TestFormatMLXMemoryLimitDiagnosticIncludesCUDADevice(t *testing.T) {
	plan := mlxMemoryLimitPlan{
		HostRAMBytes:      512 * mlxBytesPerGiB,
		DeviceName:        "NVIDIA L4",
		DeviceMemoryBytes: 23034 * mlxBytesPerMiB,
		DeviceFreeBytes:   22800 * mlxBytesPerMiB,
		DedicatedDevice:   true,
		MemoryLimitBytes:  21435 * mlxBytesPerMiB,
		CacheLimitBytes:   2875 * mlxBytesPerMiB,
		ApplyMemoryLimit:  true,
		ApplyCacheLimit:   true,
		AutoDefault:       true,
	}
	got := formatMLXMemoryLimitDiagnostic(plan, plan.MemoryLimitBytes, plan.MemoryLimitBytes)
	for _, want := range []string{
		`device="NVIDIA L4"`,
		"vram=23034.0MiB",
		"free_vram=22800.0MiB",
		"memory=21435.0MiB (auto",
		"cache=2875.0MiB (auto",
		"total_ram=524288.0MiB",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("diagnostic %q does not contain %q", got, want)
		}
	}
}

func TestAnnotateMLXTrainingStepErrorIncludesCUDAContext(t *testing.T) {
	base := errors.New("mlx_ir_trainer_submit_step failed: cudaMallocAsync failed: out of memory")
	plan := mlxMemoryLimitPlan{
		DeviceName:        "NVIDIA L4",
		DeviceMemoryBytes: 23034 * mlxBytesPerMiB,
		DedicatedDevice:   true,
		MemoryLimitBytes:  21435 * mlxBytesPerMiB,
		CacheLimitBytes:   2875 * mlxBytesPerMiB,
		ApplyMemoryLimit:  true,
		ApplyCacheLimit:   true,
	}
	got := annotateMLXTrainingStepError(base, plan, 8, 16000)
	if !errors.Is(got, base) {
		t.Fatal("annotated error does not wrap original error")
	}
	for _, want := range []string{
		`device "NVIDIA L4"`,
		"vram=23034.0MiB",
		"configured_memory_limit=21435.0MiB",
		"configured_cache_limit=2875.0MiB",
		"batch_size=8",
		"seq_len=16000",
		"batch_tokens=128000",
		mlxMemoryLimitMBEnv,
		mlxCacheLimitMBEnv,
	} {
		if !strings.Contains(got.Error(), want) {
			t.Fatalf("annotated error %q does not contain %q", got, want)
		}
	}
}

func TestAnnotateMLXTrainingStepErrorLeavesNonOOMUnchanged(t *testing.T) {
	base := errors.New("invalid trainer input")
	got := annotateMLXTrainingStepError(base, mlxMemoryLimitPlan{DedicatedDevice: true}, 8, 16000)
	if got != base {
		t.Fatalf("non-OOM error changed: got %v, want original", got)
	}
}

func TestResolveMLXMemoryLimitPlanEnvOverrides(t *testing.T) {
	clearMLXMemoryLimitEnv(t)
	t.Setenv(mlxMemoryLimitMBEnv, "4096")
	t.Setenv(mlxCacheLimitMBEnv, "0")
	plan, err := resolveMLXMemoryLimitPlan(mlxMemoryCapacity{HostRAMBytes: 64 * mlxBytesPerGiB})
	if err != nil {
		t.Fatalf("resolveMLXMemoryLimitPlan: %v", err)
	}
	if !plan.MemoryLimitEnv || !plan.CacheLimitEnv {
		t.Fatalf("plan did not mark env overrides: %+v", plan)
	}
	if plan.MemoryLimitBytes != 4096*mlxBytesPerMiB {
		t.Fatalf("memory=%d, want %d", plan.MemoryLimitBytes, 4096*mlxBytesPerMiB)
	}
	if plan.CacheLimitBytes != 0 {
		t.Fatalf("cache=%d, want 0", plan.CacheLimitBytes)
	}
}

func TestResolveMLXMemoryLimitPlanDisableDefaultsKeepsExplicitOverrides(t *testing.T) {
	clearMLXMemoryLimitEnv(t)
	t.Setenv(mlxDisableDefaultMemoryLimitsEnv, "1")
	plan, err := resolveMLXMemoryLimitPlan(mlxMemoryCapacity{HostRAMBytes: 64 * mlxBytesPerGiB})
	if err != nil {
		t.Fatalf("resolveMLXMemoryLimitPlan: %v", err)
	}
	if !plan.DefaultDisabled {
		t.Fatalf("default disabled flag not set: %+v", plan)
	}
	if plan.ApplyMemoryLimit || plan.ApplyCacheLimit {
		t.Fatalf("defaults were applied despite disable env: %+v", plan)
	}

	t.Setenv(mlxCacheLimitMBEnv, "8192")
	plan, err = resolveMLXMemoryLimitPlan(mlxMemoryCapacity{HostRAMBytes: 64 * mlxBytesPerGiB})
	if err != nil {
		t.Fatalf("resolveMLXMemoryLimitPlan with explicit cache: %v", err)
	}
	if !plan.ApplyCacheLimit || plan.CacheLimitBytes != 8192*mlxBytesPerMiB {
		t.Fatalf("explicit cache override not applied: %+v", plan)
	}
	if plan.ApplyMemoryLimit {
		t.Fatalf("memory limit should remain unset with defaults disabled: %+v", plan)
	}
}

func TestResolveMLXMemoryLimitPlanRejectsInvalidEnv(t *testing.T) {
	clearMLXMemoryLimitEnv(t)
	t.Setenv(mlxMemoryLimitMBEnv, "0")
	if _, err := resolveMLXMemoryLimitPlan(mlxMemoryCapacity{HostRAMBytes: 64 * mlxBytesPerGiB}); err == nil {
		t.Fatal("resolveMLXMemoryLimitPlan accepted zero memory limit")
	}

	clearMLXMemoryLimitEnv(t)
	t.Setenv(mlxMemoryLimitMBEnv, "1024")
	t.Setenv(mlxCacheLimitMBEnv, "2048")
	if _, err := resolveMLXMemoryLimitPlan(mlxMemoryCapacity{HostRAMBytes: 64 * mlxBytesPerGiB}); err == nil {
		t.Fatal("resolveMLXMemoryLimitPlan accepted cache limit above memory limit")
	}
}

func TestPhysicalMemoryBytesSmoke(t *testing.T) {
	total, err := physicalMemoryBytes()
	if err != nil {
		if runtime.GOOS != "darwin" && runtime.GOOS != "linux" {
			t.Skipf("physical memory lookup not implemented on %s", runtime.GOOS)
		}
		t.Fatalf("physicalMemoryBytes: %v", err)
	}
	if total < 512*mlxBytesPerMiB {
		t.Fatalf("physicalMemoryBytes=%d, want a plausible host RAM size", total)
	}
}

func clearMLXMemoryLimitEnv(t *testing.T) {
	t.Helper()
	t.Setenv(mlxMemoryLimitMBEnv, "")
	t.Setenv(mlxCacheLimitMBEnv, "")
	t.Setenv(mlxDisableDefaultMemoryLimitsEnv, "")
}
