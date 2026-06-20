package gpu

func MemoryStatsSnapshot() MemoryStats {
	return mlxMemoryStats()
}

func ClearMemoryCache() {
	mlxClearMemoryCache()
}

func SetMemoryLimit(bytes uint64) uint64 {
	return mlxSetMemoryLimit(bytes)
}

func SetMemoryCacheLimit(bytes uint64) uint64 {
	return mlxSetMemoryCacheLimit(bytes)
}
