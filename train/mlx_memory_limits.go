package train

import (
	"fmt"
	"io"
	"os"
	"runtime"
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

type mlxMemoryCapacity struct {
	HostRAMBytes          uint64
	DeviceName            string
	DeviceMemoryBytes     uint64
	DeviceFreeMemoryBytes uint64
	CurrentMemoryLimit    uint64
	DedicatedDeviceMemory bool
}

type mlxMemoryLimitPlan struct {
	HostRAMBytes      uint64
	DeviceName        string
	DeviceMemoryBytes uint64
	DeviceFreeBytes   uint64
	DedicatedDevice   bool
	MemoryLimitBytes  uint64
	CacheLimitBytes   uint64
	ApplyMemoryLimit  bool
	ApplyCacheLimit   bool
	MemoryLimitEnv    bool
	CacheLimitEnv     bool
	AutoDefault       bool
	DefaultDisabled   bool
}

func configureMLXMemoryLimits(name string) (mlxMemoryLimitPlan, error) {
	return configureMLXMemoryLimitsTo(name, os.Stdout)
}

func configureMLXMemoryLimitsTo(name string, output io.Writer) (mlxMemoryLimitPlan, error) {
	plan, err := resolveMLXMemoryLimitPlan(detectMLXMemoryCapacity())
	if err != nil {
		return mlxMemoryLimitPlan{}, err
	}
	if !plan.ApplyMemoryLimit && !plan.ApplyCacheLimit {
		if notice := mlxUnappliedLimitNotice(plan); notice != "" && output != nil {
			if _, err := fmt.Fprintf(output, "  [%s] MLX memory limits: %s\n", name, notice); err != nil {
				return mlxMemoryLimitPlan{}, err
			}
		}
		return plan, nil
	}
	var previousMemoryLimit uint64
	var previousCacheLimit uint64
	if plan.ApplyMemoryLimit {
		previousMemoryLimit = gpu.SetMemoryLimit(plan.MemoryLimitBytes)
	}
	if plan.ApplyCacheLimit {
		previousCacheLimit = gpu.SetMemoryCacheLimit(plan.CacheLimitBytes)
	}
	if output != nil {
		if _, err := fmt.Fprintf(output, "  [%s] MLX memory limits: %s\n", name,
			formatMLXMemoryLimitDiagnostic(plan, previousMemoryLimit, previousCacheLimit)); err != nil {
			return mlxMemoryLimitPlan{}, err
		}
	}
	return plan, nil
}

// mlxUnappliedLimitNotice explains why no limit was set, for the one case where
// silence misleads. Declining to size a limit from host RAM is correct when a
// dedicated-memory device cannot be read, but it is indistinguishable from the
// feature never running -- on the very platform whose memory failures this
// exists to diagnose. Opting out explicitly needs no narration.
func mlxUnappliedLimitNotice(plan mlxMemoryLimitPlan) string {
	if !plan.DedicatedDevice || plan.DefaultDisabled || plan.DeviceMemoryBytes > 0 {
		return ""
	}
	return "device memory unavailable, retaining MLX defaults"
}

func formatMLXMemoryLimitDiagnostic(plan mlxMemoryLimitPlan, previousMemoryLimit, previousCacheLimit uint64) string {
	parts := make([]string, 0, 7)
	if plan.DeviceName != "" {
		parts = append(parts, fmt.Sprintf("device=%q", plan.DeviceName))
	}
	if plan.DeviceMemoryBytes > 0 {
		label := "device_memory"
		if plan.DedicatedDevice {
			label = "vram"
		}
		parts = append(parts, label+"="+formatMiB(plan.DeviceMemoryBytes))
	}
	if plan.DedicatedDevice && plan.DeviceFreeBytes > 0 {
		parts = append(parts, "free_vram="+formatMiB(plan.DeviceFreeBytes))
	}
	if plan.ApplyMemoryLimit {
		parts = append(parts, fmt.Sprintf("memory=%s (%s, previous %s)",
			formatMiB(plan.MemoryLimitBytes), mlxMemoryLimitSource(plan.MemoryLimitEnv), formatMiB(previousMemoryLimit)))
	}
	if plan.ApplyCacheLimit {
		parts = append(parts, fmt.Sprintf("cache=%s (%s, previous %s)",
			formatMiB(plan.CacheLimitBytes), mlxMemoryLimitSource(plan.CacheLimitEnv), formatMiB(previousCacheLimit)))
	}
	if plan.HostRAMBytes > 0 {
		parts = append(parts, "total_ram="+formatMiB(plan.HostRAMBytes))
	}
	return strings.Join(parts, ", ")
}

func detectMLXMemoryCapacity() mlxMemoryCapacity {
	hostRAM, _ := physicalMemoryBytes()
	capacity := mlxMemoryCapacity{
		HostRAMBytes:          hostRAM,
		DedicatedDeviceMemory: runtime.GOOS == "linux",
	}
	if device, ok := gpu.DeviceMemoryInfo(); ok {
		capacity.DeviceName = device.Name
		capacity.DeviceMemoryBytes = device.TotalBytes
		capacity.DeviceFreeMemoryBytes = device.FreeBytes
		capacity.CurrentMemoryLimit = gpu.CurrentMemoryLimit()
	}
	return capacity
}

func mlxMemoryLimitSource(fromEnv bool) string {
	if fromEnv {
		return "env"
	}
	return "auto"
}

func resolveMLXMemoryLimitPlan(capacity mlxMemoryCapacity) (mlxMemoryLimitPlan, error) {
	plan := mlxMemoryLimitPlan{
		HostRAMBytes:      capacity.HostRAMBytes,
		DeviceName:        capacity.DeviceName,
		DeviceMemoryBytes: capacity.DeviceMemoryBytes,
		DeviceFreeBytes:   capacity.DeviceFreeMemoryBytes,
		DedicatedDevice:   capacity.DedicatedDeviceMemory,
	}
	if envTruthy(mlxDisableDefaultMemoryLimitsEnv) {
		plan.DefaultDisabled = true
	} else if memoryLimit, cacheLimit, ok := defaultMLXMemoryLimitsForCapacity(capacity); ok {
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

func defaultMLXMemoryLimitsForCapacity(capacity mlxMemoryCapacity) (memoryLimit, cacheLimit uint64, ok bool) {
	if !capacity.DedicatedDeviceMemory {
		return defaultMLXMemoryLimits(capacity.HostRAMBytes)
	}
	available := capacity.DeviceMemoryBytes
	if available == 0 {
		// A Linux MLX process is expected to use CUDA. Never substitute host RAM
		// when the runtime cannot report device memory; MLX's own limit is safer.
		return 0, 0, false
	}
	if capacity.DeviceFreeMemoryBytes > 0 && capacity.DeviceFreeMemoryBytes < available {
		available = capacity.DeviceFreeMemoryBytes
	}
	if capacity.CurrentMemoryLimit > 0 && capacity.CurrentMemoryLimit <= available {
		memoryLimit = capacity.CurrentMemoryLimit
	} else {
		memoryLimit = available * 3 / 4
	}
	cacheLimit = available / 8
	if cacheLimit < 512*mlxBytesPerMiB {
		cacheLimit = 512 * mlxBytesPerMiB
	}
	if halfMemory := memoryLimit / 2; halfMemory > 0 && cacheLimit > halfMemory {
		cacheLimit = halfMemory
	}
	if memoryLimit == 0 || cacheLimit == 0 {
		return 0, 0, false
	}
	return memoryLimit, cacheLimit, true
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

func annotateMLXTrainingStepError(err error, plan mlxMemoryLimitPlan, batchSize, seqLen int) error {
	if err == nil || !plan.DedicatedDevice || !isMLXOutOfMemoryError(err) {
		return err
	}
	return fmt.Errorf(
		"CUDA out of memory on device %q (vram=%s, configured_memory_limit=%s, configured_cache_limit=%s, batch_size=%d, seq_len=%d, batch_tokens=%d); reduce batch_tokens or seq_len, or tune %s and %s: %w",
		plan.DeviceName,
		formatOptionalMiB(plan.DeviceMemoryBytes),
		formatAppliedLimit(plan.ApplyMemoryLimit, plan.MemoryLimitBytes),
		formatAppliedLimit(plan.ApplyCacheLimit, plan.CacheLimitBytes),
		batchSize,
		seqLen,
		batchSize*seqLen,
		mlxMemoryLimitMBEnv,
		mlxCacheLimitMBEnv,
		err,
	)
}

func isMLXOutOfMemoryError(err error) bool {
	message := strings.ToLower(err.Error())
	return strings.Contains(message, "out of memory") ||
		strings.Contains(message, "cudamalloc") ||
		strings.Contains(message, "cuda allocation")
}

func formatOptionalMiB(bytes uint64) string {
	if bytes == 0 {
		return "unknown"
	}
	return formatMiB(bytes)
}

func formatAppliedLimit(applied bool, bytes uint64) string {
	if !applied {
		return "runtime-default"
	}
	return formatMiB(bytes)
}
