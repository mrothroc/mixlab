#include "cuda_kernel_dispatch.h"

#include "cuda_kernels/registry_generated.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace mx = mlx::core;

namespace mlx_ir {
namespace {

#ifdef __linux__
std::string lookup_precompiled_kernel_blob(const std::string& kernel_name) {
  if (cuda_kernels::kEmbeddedCudaKernelImageCount == 0) {
    throw std::runtime_error("no embedded CUDA kernels were built");
  }
  for (unsigned int i = 0; i < cuda_kernels::kEmbeddedCudaKernelImageCount; ++i) {
    const auto& candidate = cuda_kernels::kEmbeddedCudaKernelImages[i];
    if (kernel_name == candidate.kernel_name) {
      std::cout << "[cuda_kernel_dispatch] loading kernel name=" << kernel_name
                << " fatbin_size=" << candidate.blob_len << std::endl;
      return std::string(
          reinterpret_cast<const char*>(candidate.blob),
          static_cast<size_t>(candidate.blob_len));
    }
  }
  throw std::runtime_error("embedded CUDA kernel not found: " + kernel_name);
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
  std::cout << "[gated_delta_cuda] before cuda_kernel factory" << std::endl;
  auto outputs = mx::fast::precompiled_cuda_kernel(
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
  std::cout << "[gated_delta_cuda] cuda_kernel factory returned" << std::endl;
  std::cout << "[gated_delta_cuda] returning kernel output before eval" << std::endl;
  return outputs;
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
