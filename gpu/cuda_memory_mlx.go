//go:build mlx && cgo && (darwin || linux)

package gpu

/*
#include "mlx_bridge.h"
*/
import "C"

func mlxCUDAMemorySnapshot() (CUDAMemory, bool) {
	var p, u, g, v C.uint64_t
	if C.mlx_cuda_memory_snapshot(&p, &u, &g, &v) != 0 {
		return CUDAMemory{}, false
	}
	return CUDAMemory{uint64(p), uint64(u), uint64(g), uint64(v)}, true
}
