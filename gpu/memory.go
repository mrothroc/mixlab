package gpu

func MemoryStatsSnapshot() MemoryStats {
	return mlxMemoryStats()
}

func ClearMemoryCache() {
	mlxClearMemoryCache()
}

func SetMemoryCacheLimit(bytes uint64) uint64 {
	return mlxSetMemoryCacheLimit(bytes)
}
