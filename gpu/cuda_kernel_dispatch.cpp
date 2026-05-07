#include "cuda_kernel_dispatch.h"

#include "cuda_kernels/registry_generated.h"

#include <mlx/stream.h>

#ifdef __linux__
#include <mlx/backend/cuda/allocator.h>
#include <mlx/backend/cuda/device.h>
#include <mlx/backend/cuda/jit_module.h>
#include <mlx/backend/cuda/utils.h>
#endif

#include <cstdlib>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <variant>

namespace mx = mlx::core;

namespace mlx_ir {
namespace {

#ifdef __linux__
bool log_cuda_kernel_debug() {
  const char* override = std::getenv("MIXLAB_GATED_DELTA_CUDA_DEBUG");
  return override != nullptr && std::string(override) == "1";
}

std::mutex g_cuda_kernel_load_mu;
std::unordered_set<std::string> g_precompiled_cuda_load_failures;

std::string cuda_array_shape_string(const mx::array& array) {
  std::ostringstream oss;
  oss << "[";
  for (int i = 0; i < array.ndim(); ++i) {
    if (i > 0) {
      oss << ",";
    }
    oss << array.shape(i);
  }
  oss << "]";
  return oss.str();
}

std::string lookup_precompiled_kernel_blob(const std::string& kernel_name) {
  if (cuda_kernels::kEmbeddedCudaKernelImageCount == 0) {
    throw std::runtime_error("no embedded CUDA kernels were built");
  }
  for (unsigned int i = 0; i < cuda_kernels::kEmbeddedCudaKernelImageCount; ++i) {
    const auto& candidate = cuda_kernels::kEmbeddedCudaKernelImages[i];
    if (kernel_name == candidate.kernel_name) {
      if (log_cuda_kernel_debug()) {
        std::cout << "[cuda_kernel_dispatch] loading kernel name=" << kernel_name
                  << " fatbin_size=" << candidate.blob_len << std::endl;
      }
      return std::string(
          reinterpret_cast<const char*>(candidate.blob),
          static_cast<size_t>(candidate.blob_len));
    }
  }
  throw std::runtime_error("embedded CUDA kernel not found: " + kernel_name);
}

std::string lookup_cuda_kernel_source(const std::string& kernel_name) {
  if (cuda_kernels::kEmbeddedCudaKernelImageCount == 0) {
    throw std::runtime_error("no embedded CUDA kernels were built");
  }
  for (unsigned int i = 0; i < cuda_kernels::kEmbeddedCudaKernelImageCount; ++i) {
    const auto& candidate = cuda_kernels::kEmbeddedCudaKernelImages[i];
    if (kernel_name == candidate.kernel_name) {
      if (candidate.source == nullptr || candidate.source_len == 0) {
        throw std::runtime_error("embedded CUDA source not found: " + kernel_name);
      }
      return std::string(
          reinterpret_cast<const char*>(candidate.source),
          static_cast<size_t>(candidate.source_len));
    }
  }
  throw std::runtime_error("embedded CUDA source not found: " + kernel_name);
}

mx::cu::JitModule& load_precompiled_or_source_cuda_module(
    mx::Stream stream,
    const std::string& kernel_name) {
  {
    std::lock_guard<std::mutex> lock(g_cuda_kernel_load_mu);
    if (g_precompiled_cuda_load_failures.find(kernel_name) != g_precompiled_cuda_load_failures.end()) {
      const auto source = lookup_cuda_kernel_source(kernel_name);
      return mx::cu::get_jit_module(
          stream.device,
          kernel_name + "_source",
          [source, kernel_name]() {
            return std::make_tuple(
                false,
                source,
                std::vector<std::string>{kernel_name});
          },
          false);
    }
  }

  const auto blob = lookup_precompiled_kernel_blob(kernel_name);
  try {
    return mx::cu::get_jit_module(
        stream.device,
        kernel_name,
        [blob, kernel_name]() {
          return std::make_tuple(
              true,
              blob,
              std::vector<std::string>{kernel_name});
        },
        false);
  } catch (const std::exception& e) {
    {
      std::lock_guard<std::mutex> lock(g_cuda_kernel_load_mu);
      g_precompiled_cuda_load_failures.insert(kernel_name);
    }
    const auto source = lookup_cuda_kernel_source(kernel_name);
    std::cerr << "[cuda_kernel_dispatch] precompiled CUDA kernel load failed for "
              << kernel_name << " (" << e.what()
              << "); retrying from embedded source" << std::endl;
    return mx::cu::get_jit_module(
        stream.device,
        kernel_name + "_source",
        [source, kernel_name]() {
          return std::make_tuple(
              false,
              source,
              std::vector<std::string>{kernel_name});
        },
        false);
  }
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
  const bool debug = log_cuda_kernel_debug();
  if (debug) {
    std::cout << "[gated_delta_cuda] before cuda_kernel factory" << std::endl;
  }
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
  if (debug) {
    std::cout << "[gated_delta_cuda] cuda_kernel factory returned" << std::endl;
    std::cout << "[gated_delta_cuda] before synchronize(stream)" << std::endl;
  }
  mx::synchronize(stream);
  if (debug) {
    std::cout << "[gated_delta_cuda] after synchronize(stream)" << std::endl;
    std::cout << "[gated_delta_cuda] returning kernel output before eval" << std::endl;
  }
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

void launch_precompiled_cuda_kernel_into(
    const std::string& kernel_name,
    const std::vector<mx::array>& inputs,
    const std::vector<mx::array*>& outputs,
    const std::vector<mx::fast::ScalarArg>& scalars,
    std::tuple<int, int, int> grid,
    std::tuple<int, int, int> threadgroup,
    mx::Stream stream,
    int shared_memory,
    bool allocate_outputs) {
#ifdef __linux__
  auto& encoder = mx::cu::get_command_encoder(stream);
  if (allocate_outputs) {
    for (size_t output_idx = 0; output_idx < outputs.size(); ++output_idx) {
      auto* out = outputs[output_idx];
      if (out == nullptr) {
        throw std::runtime_error("null output passed to precompiled CUDA kernel");
      }
      try {
        out->set_data(mx::cu::malloc_async(out->nbytes(), encoder));
      } catch (const std::exception& e) {
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        const auto mem_status = cuMemGetInfo(&free_bytes, &total_bytes);
        std::ostringstream oss;
        oss << "CUDA allocation failed for kernel " << kernel_name
            << " output[" << output_idx << "]"
            << " shape=" << cuda_array_shape_string(*out)
            << " nbytes=" << out->nbytes();
        if (mem_status == CUDA_SUCCESS) {
          oss << " cuda_free=" << free_bytes << " cuda_total=" << total_bytes;
        }
        oss << ": " << e.what();
        throw std::runtime_error(oss.str());
      }
    }
  }

  mx::cu::JitModule& mod = load_precompiled_or_source_cuda_module(stream, kernel_name);

  mx::cu::KernelArgs args;
  for (const auto& in : inputs) {
    args.append(in);
  }
  for (auto* out : outputs) {
    args.append(*out);
  }
  for (const auto& scalar : scalars) {
    if (std::holds_alternative<bool>(scalar)) {
      args.append(std::get<bool>(scalar));
    } else if (std::holds_alternative<int>(scalar)) {
      args.append(std::get<int>(scalar));
    } else if (std::holds_alternative<float>(scalar)) {
      args.append(std::get<float>(scalar));
    }
  }

  const auto [tx, ty, tz] = threadgroup;
  const auto [gx, gy, gz] = grid;
  dim3 block(tx, ty, tz);
  dim3 grid_dim(gx, gy, gz);

  for (const auto& in : inputs) {
    encoder.set_input_array(in);
  }
  for (const auto* out : outputs) {
    encoder.set_output_array(*out);
  }
  auto kernel = mod.get_kernel(kernel_name, [shared_memory](CUfunction kernel) {
    if (shared_memory > 0 && shared_memory > 48000) {
      cuFuncSetAttribute(
          kernel,
          CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
          shared_memory);
    }
  });
  encoder.add_kernel_node_raw(
      kernel,
      grid_dim,
      block,
      {},
      static_cast<uint32_t>(shared_memory),
      args.args());
#else
  (void)kernel_name;
  (void)inputs;
  (void)outputs;
  (void)scalars;
  (void)grid;
  (void)threadgroup;
  (void)stream;
  (void)shared_memory;
  (void)allocate_outputs;
  throw std::runtime_error("precompiled CUDA kernels are unavailable on this platform");
#endif
}

} // namespace mlx_ir
