package gpu

// CUDAMemory separates runtime reservations from MLX's live-array accounting.
// PoolUsedBytes is part of PoolReservedBytes, not an additional allocation.
// Graph memory is reported separately by CUDA. Snapshots do not synchronize.
type CUDAMemory struct {
	PoolReservedBytes  uint64
	PoolUsedBytes      uint64
	GraphReservedBytes uint64
	GraphUsedBytes     uint64
}

func CUDAMemorySnapshot() (CUDAMemory, bool) { return mlxCUDAMemorySnapshot() }
