package train

import (
	"bytes"
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
	if err := configureMLXMemoryLimitsTo("generate", &out); err != nil {
		t.Fatal(err)
	}
	if got := out.String(); !strings.Contains(got, "[generate] MLX memory limits") || !strings.Contains(got, "cache=64.0MiB") {
		t.Fatalf("memory-limit diagnostic=%q", got)
	}
}

func TestResolveMLXMemoryLimitPlanDefaults(t *testing.T) {
	clearMLXMemoryLimitEnv(t)
	plan, err := resolveMLXMemoryLimitPlan(64 * mlxBytesPerGiB)
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

func TestResolveMLXMemoryLimitPlanEnvOverrides(t *testing.T) {
	clearMLXMemoryLimitEnv(t)
	t.Setenv(mlxMemoryLimitMBEnv, "4096")
	t.Setenv(mlxCacheLimitMBEnv, "0")
	plan, err := resolveMLXMemoryLimitPlan(64 * mlxBytesPerGiB)
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
	plan, err := resolveMLXMemoryLimitPlan(64 * mlxBytesPerGiB)
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
	plan, err = resolveMLXMemoryLimitPlan(64 * mlxBytesPerGiB)
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
	if _, err := resolveMLXMemoryLimitPlan(64 * mlxBytesPerGiB); err == nil {
		t.Fatal("resolveMLXMemoryLimitPlan accepted zero memory limit")
	}

	clearMLXMemoryLimitEnv(t)
	t.Setenv(mlxMemoryLimitMBEnv, "1024")
	t.Setenv(mlxCacheLimitMBEnv, "2048")
	if _, err := resolveMLXMemoryLimitPlan(64 * mlxBytesPerGiB); err == nil {
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
