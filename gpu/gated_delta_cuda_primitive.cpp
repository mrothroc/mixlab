#include "gated_delta_cuda_primitive.h"
#include "cuda_kernel_dispatch.h"

#include <mlx/ops.h>
#include <mlx/primitives.h>

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <ios>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace mx = mlx::core;

namespace mlx_ir {
namespace {

std::atomic<int> g_eval_gpu_calls{0};

struct ForceUnbufferedStdout {
  ForceUnbufferedStdout() { std::cout.setf(std::ios::unitbuf); }
} _force_unbuffered_stdout;

struct ForceUnbufferedStderr {
  ForceUnbufferedStderr() { std::cerr.setf(std::ios::unitbuf); }
} _force_unbuffered_stderr;

bool use_experimental_gated_delta_cuda_kernel() {
  const char* override = std::getenv("MIXLAB_GATED_DELTA_USE_CUDA_KERNEL");
  return override != nullptr && std::string(override) == "1";
}

bool log_gated_delta_cuda_debug() {
  const char* override = std::getenv("MIXLAB_GATED_DELTA_CUDA_DEBUG");
  return override != nullptr && std::string(override) == "1";
}

class SolveStrictlyLowerCUDAPrimitive : public mx::UnaryPrimitive {
 public:
  SolveStrictlyLowerCUDAPrimitive(
      mx::Stream stream,
      int matrix_count,
      int chunk_size)
      : mx::UnaryPrimitive(stream),
        matrix_count_(matrix_count),
        chunk_size_(chunk_size) {}

  void eval_cpu(const std::vector<mx::array>&, mx::array&) override {
    throw std::runtime_error("SolveStrictlyLowerCUDAPrimitive is CUDA-only");
  }

  void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override {
    const bool debug = log_gated_delta_cuda_debug();
    if (debug) {
      const int call_n = ++g_eval_gpu_calls;
      std::cout << "[gated_delta_cuda] eval_gpu CALL #" << call_n << " env="
                << (std::getenv("MIXLAB_GATED_DELTA_USE_CUDA_KERNEL")
                        ? std::getenv("MIXLAB_GATED_DELTA_USE_CUDA_KERNEL")
                        : "(unset)")
                << std::endl;
    }
    if (inputs.size() != 1) {
      throw std::runtime_error("SolveStrictlyLowerCUDAPrimitive expects 1 input");
    }
    if (debug) {
      std::cout << "[gated_delta_cuda] input[0].shape=[";
      for (size_t i = 0; i < inputs[0].shape().size(); ++i) {
        std::cout << inputs[0].shape()[i];
        if (i + 1 < inputs[0].shape().size()) {
          std::cout << ",";
        }
      }
      std::cout << "] dtype=" << inputs[0].dtype() << std::endl;
    }
    if (!use_experimental_gated_delta_cuda_kernel()) {
      throw std::runtime_error(
          "SolveStrictlyLowerCUDAPrimitive requires MIXLAB_GATED_DELTA_USE_CUDA_KERNEL=1");
    }
    if (inputs[0].dtype() != mx::float32 || out.dtype() != mx::float32) {
      throw std::runtime_error("SolveStrictlyLowerCUDAPrimitive requires float32 tensors");
    }

    if (debug) {
      std::cout << "[gated_delta_cuda] solve_strictly_lower_cuda enter matrix_count="
                << matrix_count_ << " chunk_size=" << chunk_size_ << std::endl;
    }
    const int threads = chunk_size_ < 128 ? chunk_size_ : 128;
    const int blocks = (chunk_size_ + threads - 1) / threads;
    if (debug) {
      std::cout << "[gated_delta_cuda] before precompiled_cuda_kernel grid=("
                << blocks << "," << matrix_count_ << ",1) block=("
                << threads << ",1,1)" << std::endl;
    }
    launch_precompiled_cuda_kernel_into(
        "gated_delta_chunk_solve",
        {inputs[0]},
        {&out},
        {chunk_size_},
        std::make_tuple(blocks, matrix_count_, 1),
        std::make_tuple(threads, 1, 1),
        stream());
    if (debug) {
      std::cout << "[gated_delta_cuda] eval_gpu RETURNING out.shape=[";
      for (size_t i = 0; i < out.shape().size(); ++i) {
        std::cout << out.shape()[i];
        if (i + 1 < out.shape().size()) {
          std::cout << ",";
        }
      }
      std::cout << "] dtype=" << out.dtype() << std::endl;
    }
  }

  std::vector<mx::array> vjp(
      const std::vector<mx::array>&,
      const std::vector<mx::array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<mx::array>& outputs) override {
    const bool debug = log_gated_delta_cuda_debug();
    if (debug) {
      std::cout << "[gated_delta_cuda] vjp ENTER" << std::endl;
    }
    if (cotangents.size() != 1 || outputs.size() != 1) {
      throw std::runtime_error("SolveStrictlyLowerCUDAPrimitive vjp expects one cotangent and one output");
    }
    auto solve_t = mx::transpose(outputs[0], {0, 2, 1});
    auto grad = mx::matmul(mx::matmul(solve_t, cotangents[0]), solve_t);
    grad = mx::tril(grad, -1);

    std::vector<mx::array> grads;
    grads.reserve(argnums.size());
    for (int argnum : argnums) {
      if (argnum != 0) {
        throw std::runtime_error("SolveStrictlyLowerCUDAPrimitive vjp argnum out of range");
      }
      grads.push_back(grad);
    }
    if (debug) {
      std::cout << "[gated_delta_cuda] vjp RETURN" << std::endl;
    }
    return grads;
  }

  const char* name() const override {
    return "SolveStrictlyLowerCUDAPrimitive";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    auto* rhs = dynamic_cast<const SolveStrictlyLowerCUDAPrimitive*>(&other);
    return rhs != nullptr &&
        matrix_count_ == rhs->matrix_count_ &&
        chunk_size_ == rhs->chunk_size_;
  }

  std::vector<mx::Shape> output_shapes(const std::vector<mx::array>&) override {
    return {mx::Shape{
        static_cast<mx::ShapeElem>(matrix_count_),
        static_cast<mx::ShapeElem>(chunk_size_),
        static_cast<mx::ShapeElem>(chunk_size_)}};
  }

 private:
  int matrix_count_;
  int chunk_size_;
};

} // namespace

mx::array solve_strictly_lower_cuda_primitive(
    const mx::array& raw_attn,
    int matrix_count,
    int chunk_size) {
  auto stream = mx::default_stream(mx::default_device());
  auto primitive = std::make_shared<SolveStrictlyLowerCUDAPrimitive>(
      stream,
      matrix_count,
      chunk_size);

  return mx::array(
      mx::Shape{
          static_cast<mx::ShapeElem>(matrix_count),
          static_cast<mx::ShapeElem>(chunk_size),
          static_cast<mx::ShapeElem>(chunk_size)},
      mx::float32,
      primitive,
      std::vector<mx::array>{raw_attn});
}

} // namespace mlx_ir
