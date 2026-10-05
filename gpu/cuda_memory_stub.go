//go:build !mlx || !cgo || (!darwin && !linux)

package gpu

func mlxCUDAMemorySnapshot() (CUDAMemory, bool) { return CUDAMemory{}, false }
