package train

import (
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/mrothroc/mixlab/gpu"
)

const (
	mlxMemLogEveryEnv                = "MIXLAB_MLX_MEM_LOG_EVERY"
	mlxClearCacheEveryEnv            = "MIXLAB_MLX_CLEAR_CACHE_EVERY"
	mlxCacheLimitMBEnv               = "MIXLAB_MLX_CACHE_LIMIT_MB"
	mlxMemoryLimitMBEnv              = "MIXLAB_MLX_MEMORY_LIMIT_MB"
	mlxDisableDefaultMemoryLimitsEnv = "MIXLAB_DISABLE_MLX_MEMORY_LIMITS"

	mlxBytesPerMiB = uint64(1024 * 1024)
	mlxBytesPerGiB = uint64(1024 * 1024 * 1024)
)

type mlxMemoryLimitPlan struct {
	TotalRAMBytes    uint64
	MemoryLimitBytes uint64
	CacheLimitBytes  uint64
	ApplyMemoryLimit bool
	ApplyCacheLimit  bool
	MemoryLimitEnv   bool
	CacheLimitEnv    bool
	AutoDefault      bool
	DefaultDisabled  bool
}

func configureMLXMemoryLimits(name string) error {
	totalRAM, _ := physicalMemoryBytes()
	plan, err := resolveMLXMemoryLimitPlan(totalRAM)
	if err != nil {
		return err
	}
	if !plan.ApplyMemoryLimit && !plan.ApplyCacheLimit {
		return nil
	}
	parts := make([]string, 0, 3)
	if plan.ApplyMemoryLimit {
		prev := gpu.SetMemoryLimit(plan.MemoryLimitBytes)
		parts = append(parts, fmt.Sprintf("memory=%s (%s, previous %s)",
			formatMiB(plan.MemoryLimitBytes), mlxMemoryLimitSource(plan.MemoryLimitEnv), formatMiB(prev)))
	}
	if plan.ApplyCacheLimit {
		prev := gpu.SetMemoryCacheLimit(plan.CacheLimitBytes)
		parts = append(parts, fmt.Sprintf("cache=%s (%s, previous %s)",
			formatMiB(plan.CacheLimitBytes), mlxMemoryLimitSource(plan.CacheLimitEnv), formatMiB(prev)))
	}
	if plan.TotalRAMBytes > 0 {
		parts = append(parts, "total_ram="+formatMiB(plan.TotalRAMBytes))
	}
	fmt.Printf("  [%s] MLX memory limits: %s\n", name, strings.Join(parts, ", "))
	return nil
}

func mlxMemoryLimitSource(fromEnv bool) string {
	if fromEnv {
		return "env"
	}
	return "auto"
}

func resolveMLXMemoryLimitPlan(totalRAM uint64) (mlxMemoryLimitPlan, error) {
	plan := mlxMemoryLimitPlan{TotalRAMBytes: totalRAM}
	if envTruthy(mlxDisableDefaultMemoryLimitsEnv) {
		plan.DefaultDisabled = true
	} else if memoryLimit, cacheLimit, ok := defaultMLXMemoryLimits(totalRAM); ok {
		plan.MemoryLimitBytes = memoryLimit
		plan.CacheLimitBytes = cacheLimit
		plan.ApplyMemoryLimit = true
		plan.ApplyCacheLimit = true
		plan.AutoDefault = true
	}

	if memoryLimit, ok, err := parseMemoryLimitMBEnv(mlxMemoryLimitMBEnv, false); err != nil {
		return mlxMemoryLimitPlan{}, err
	} else if ok {
		plan.MemoryLimitBytes = memoryLimit
		plan.ApplyMemoryLimit = true
		plan.MemoryLimitEnv = true
	}

	if cacheLimit, ok, err := parseMemoryLimitMBEnv(mlxCacheLimitMBEnv, true); err != nil {
		return mlxMemoryLimitPlan{}, err
	} else if ok {
		plan.CacheLimitBytes = cacheLimit
		plan.ApplyCacheLimit = true
		plan.CacheLimitEnv = true
	}

	if plan.ApplyMemoryLimit && plan.ApplyCacheLimit {
		if !plan.CacheLimitEnv && plan.CacheLimitBytes > plan.MemoryLimitBytes/2 {
			plan.CacheLimitBytes = plan.MemoryLimitBytes / 2
		}
		if plan.CacheLimitBytes > plan.MemoryLimitBytes {
			return mlxMemoryLimitPlan{}, fmt.Errorf("%s (%s) must be <= %s (%s)",
				mlxCacheLimitMBEnv, formatMiB(plan.CacheLimitBytes),
				mlxMemoryLimitMBEnv, formatMiB(plan.MemoryLimitBytes))
		}
	}

	return plan, nil
}

func parseMemoryLimitMBEnv(name string, allowZero bool) (uint64, bool, error) {
	raw := strings.TrimSpace(os.Getenv(name))
	if raw == "" {
		return 0, false, nil
	}
	mb, err := strconv.ParseUint(raw, 10, 64)
	if err != nil {
		return 0, false, fmt.Errorf("%s must be a non-negative integer MiB value, got %q", name, raw)
	}
	if mb == 0 && !allowZero {
		return 0, false, fmt.Errorf("%s must be > 0 MiB", name)
	}
	if mb > ^uint64(0)/mlxBytesPerMiB {
		return 0, false, fmt.Errorf("%s is too large: %q MiB", name, raw)
	}
	return mb * mlxBytesPerMiB, true, nil
}

func defaultMLXMemoryLimits(totalRAM uint64) (memoryLimit, cacheLimit uint64, ok bool) {
	if totalRAM == 0 {
		return 0, 0, false
	}
	reserve := totalRAM / 4
	if reserve < 8*mlxBytesPerGiB {
		reserve = 8 * mlxBytesPerGiB
	}
	maxReserve := totalRAM / 2
	if reserve > maxReserve {
		reserve = maxReserve
	}
	if reserve >= totalRAM {
		return 0, 0, false
	}
	memoryLimit = totalRAM - reserve
	cacheLimit = totalRAM / 8
	if cacheLimit < 512*mlxBytesPerMiB {
		cacheLimit = 512 * mlxBytesPerMiB
	}
	if halfMemory := memoryLimit / 2; halfMemory > 0 && cacheLimit > halfMemory {
		cacheLimit = halfMemory
	}
	if cacheLimit == 0 {
		cacheLimit = memoryLimit
	}
	return memoryLimit, cacheLimit, true
}
