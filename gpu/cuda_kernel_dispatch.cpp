#include "cuda_kernel_dispatch.h"

#include "cuda_kernels/registry_generated.h"

#ifdef __linux__
#include <cuda_runtime_api.h>
#endif

#include <optional>
#include <stdexcept>
#include <string>

namespace mx = mlx::core;

namespace mlx_ir {
namespace {

#ifdef __linux__
int current_cuda_sm() {
  int device = 0;
  auto err = cudaGetDevice(&device);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("cudaGetDevice failed: ") + cudaGetErrorString(err));
  }
  cudaDeviceProp prop{};
  err = cudaGetDeviceProperties(&prop, device);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("cudaGetDeviceProperties failed: ") + cudaGetErrorString(err));
  }
  return prop.major * 10 + prop.minor;
}

std::string lookup_precompiled_kernel_blob(const std::string& kernel_name) {
  if (cuda_kernels::kEmbeddedCudaKernelImageCount == 0) {
    throw std::runtime_error("no embedded CUDA kernels were built");
  }
  const int current_sm = current_cuda_sm();
  const cuda_kernels::EmbeddedKernelImage* best = nullptr;
  int smallest_supported_sm = -1;
  for (unsigned int i = 0; i < cuda_kernels::kEmbeddedCudaKernelImageCount; ++i) {
    const auto& candidate = cuda_kernels::kEmbeddedCudaKernelImages[i];
    if (kernel_name != candidate.kernel_name) {
      continue;
    }
    if (smallest_supported_sm < 0 || candidate.sm < smallest_supported_sm) {
      smallest_supported_sm = candidate.sm;
    }
    if (candidate.sm == current_sm) {
      best = &candidate;
      break;
    }
    if (candidate.sm <= current_sm &&
        (best == nullptr || candidate.sm > best->sm)) {
      best = &candidate;
    } else if (best == nullptr) {
      best = &candidate;
    }
  }
  if (best == nullptr) {
    throw std::runtime_error("embedded CUDA kernel not found: " + kernel_name);
  }
  if (best->sm > current_sm) {
    throw std::runtime_error(
        "no compatible embedded CUDA kernel image for " + kernel_name +
        " on sm_" + std::to_string(current_sm) +
        "; smallest supported architecture is sm_" + std::to_string(smallest_supported_sm));
  }
  return std::string(
      reinterpret_cast<const char*>(best->blob),
      static_cast<size_t>(best->blob_len));
}
#endif

} // namespace

std::vector<mx::array> launch_precompiled_cuda_kernel(
    const std::string& kernel_name,
    const std::vector<mx::array>& inputs,
    const std::vector<mx::Shape>& output_shapes,
    const std::vector<mx::Dtype>& output_dtypes,
    const std::vector<mx::fast::ScalarArg>& scalars,
    std::tuple<int, int, int> grid,
    std::tuple<int, int, int> threadgroup,
    mx::Stream stream,
    int shared_memory,
    bool ensure_row_contiguous) {
#ifdef __linux__
  return mx::fast::precompiled_cuda_kernel(
      kernel_name,
      lookup_precompiled_kernel_blob(kernel_name),
      inputs,
      output_shapes,
      output_dtypes,
      scalars,
      grid,
      threadgroup,
      shared_memory,
      std::nullopt,
      ensure_row_contiguous,
      stream);
#else
  (void)kernel_name;
  (void)inputs;
  (void)output_shapes;
  (void)output_dtypes;
  (void)scalars;
  (void)grid;
  (void)threadgroup;
  (void)stream;
  (void)shared_memory;
  (void)ensure_row_contiguous;
  throw std::runtime_error("precompiled CUDA kernels are unavailable on this platform");
#endif
}

} // namespace mlx_ir
