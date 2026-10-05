// Explicit GPU pressure regression for the pinned MLX allocator patch.
// Run on an otherwise idle CUDA host; this intentionally occupies most VRAM.
#include <mlx/backend/cuda/allocator.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

int main(int argc, char** argv) {
  const bool expect_oom = argc == 2 && std::string(argv[1]) == "--expect-oom";
  constexpr size_t MiB = 1ULL << 20;
  auto& allocator = mlx::core::cu::allocator();
  cudaStream_t stream;
  check(cudaStreamCreate(&stream));
  allocator.clear_cache();
  check(cudaDeviceSynchronize());
  size_t available, total;
  check(cudaMemGetInfo(&available, &total));
  if (available < 8ULL << 30) throw std::runtime_error("requires >=8 GiB free VRAM");
  void* workspace = nullptr;
  // Simulate this process's non-MLX backend workspace, not another process.
  check(cudaMalloc(&workspace, available / 3));
  allocator.set_cache_limit(available / 4);
  auto live = allocator.malloc_async(MiB, 0, stream);
  auto* live_ptr = static_cast<mlx::core::cu::CudaBuffer*>(live.ptr())->data;
  check(cudaMemsetAsync(live_ptr, 0x5a, MiB, stream));
  auto cached = allocator.malloc_async(available / 5, 0, stream);
  check(cudaStreamSynchronize(stream));
  allocator.free(cached);
  std::printf("pressure cache=%zu active=%zu free_at_start=%zu\n",
      allocator.get_cache_memory(), allocator.get_active_memory(), available);
  bool oom = false;
  try {
    auto large = allocator.malloc_async(available * 2 / 3 - 512 * MiB, 0, stream);
    check(cudaStreamSynchronize(stream));
    allocator.free(large);
  } catch (const std::exception& e) {
    if (std::string(e.what()).find("out of memory") == std::string::npos) throw;
    oom = true;
    std::printf("allocation: %s\n", e.what());
    (void)cudaGetLastError();
  }
  unsigned char sentinel[16];
  check(cudaMemcpy(sentinel, live_ptr, sizeof(sentinel), cudaMemcpyDeviceToHost));
  if (!std::all_of(std::begin(sentinel), std::end(sentinel), [](auto x) { return x == 0x5a; })) {
    throw std::runtime_error("reclaim modified live memory");
  }
  allocator.free(live);
  allocator.clear_cache();
  cudaMemPool_t pool;
  check(cudaDeviceGetDefaultMemPool(&pool, 0));
  uint64_t reserved, used;
  check(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &reserved));
  check(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used));
  std::printf("oom=%d expected=%d pool_reserved=%llu pool_used=%llu live_unchanged=1\n",
      oom, expect_oom, (unsigned long long)reserved, (unsigned long long)used);
  if (oom != expect_oom) return 1;
  if (!expect_oom && (reserved > MiB || used > MiB)) {
    throw std::runtime_error("clear_cache did not return idle pool backing");
  }
  check(cudaFree(workspace));
  check(cudaDeviceSynchronize());
  if (!expect_oom) {
    // A real capacity failure must still throw, with exactly one retry.
    bool failed = false;
    try { (void)allocator.malloc_async(total + MiB, 0, stream); }
    catch (const std::exception& e) {
      failed = std::string(e.what()).find("out of memory") != std::string::npos;
      (void)cudaGetLastError();
    }
    if (!failed) throw std::runtime_error("oversized allocation did not fail");
  }
  check(cudaStreamDestroy(stream));
}
