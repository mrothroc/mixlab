package gpu

// DeviceMemory describes memory exposed by the active MLX GPU. FreeBytes is
// zero on backends that do not publish a live free-memory value.
type DeviceMemory struct {
	Name       string
	TotalBytes uint64
	FreeBytes  uint64
}

// DeviceMemoryInfo returns memory properties for the active MLX GPU.
func DeviceMemoryInfo() (DeviceMemory, bool) {
	total, free, ok := mlxDeviceMemoryInfo()
	if !ok {
		return DeviceMemory{}, false
	}
	return DeviceMemory{Name: DeviceName(), TotalBytes: total, FreeBytes: free}, true
}

func MemoryStatsSnapshot() MemoryStats {
	return mlxMemoryStats()
}

func ClearMemoryCache() {
	mlxClearMemoryCache()
}

func SetMemoryLimit(bytes uint64) uint64 {
	return mlxSetMemoryLimit(bytes)
}

func CurrentMemoryLimit() uint64 {
	return mlxGetMemoryLimit()
}

func SetMemoryCacheLimit(bytes uint64) uint64 {
	return mlxSetMemoryCacheLimit(bytes)
}
