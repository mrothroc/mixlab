#include "mlx_bridge.h"

#ifdef __linux__
#include <cuda_runtime.h>
#endif

int mlx_cuda_memory_snapshot(uint64_t* reserved, uint64_t* used,
                             uint64_t* graph_reserved, uint64_t* graph_used) {
  *reserved = *used = *graph_reserved = *graph_used = 0;
#ifdef __linux__
  if (mlx_init() != 0) return -1;
  int device = 0;
  cudaMemPool_t pool;
  uint64_t r = 0, u = 0, gr = 0, gu = 0;
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaDeviceGetDefaultMemPool(&pool, device) != cudaSuccess ||
      cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &r) != cudaSuccess ||
      cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &u) != cudaSuccess ||
      cudaDeviceGetGraphMemAttribute(device, cudaGraphMemAttrReservedMemCurrent, &gr) != cudaSuccess ||
      cudaDeviceGetGraphMemAttribute(device, cudaGraphMemAttrUsedMemCurrent, &gu) != cudaSuccess) {
    return -1;
  }
  *reserved = r;
  *used = u;
  *graph_reserved = gr;
  *graph_used = gu;
  return 0;
#else
  return -1;
#endif
}
