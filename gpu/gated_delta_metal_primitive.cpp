#include "gated_delta_metal_primitive.h"

#include <mlx/allocator.h>
#include <mlx/ops.h>
#include <mlx/primitives.h>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef __APPLE__
#include <mlx/backend/metal/device.h>
#include <mlx/backend/metal/metal.h>
#endif

namespace mx = mlx::core;

namespace mlx_ir {
namespace {

#ifdef __APPLE__

const char* kGatedDeltaMetalSolveSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

kernel void gated_delta_chunk_solve_metal(
    const device float* raw_attn [[buffer(0)]],
    device float* solve_attn [[buffer(1)]],
    constant int& chunk_size [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint col = gid.x;
  const uint matrix = gid.y;
  if (col >= static_cast<uint>(chunk_size)) {
    return;
  }

  const uint n = static_cast<uint>(chunk_size);
  const uint base = matrix * n * n;

  for (uint row = 0; row < col; ++row) {
    solve_attn[base + row * n + col] = 0.0f;
  }
  solve_attn[base + col * n + col] = 1.0f;

  for (uint row = col + 1; row < n; ++row) {
    float acc = raw_attn[base + row * n + col];
    for (uint j = col + 1; j < row; ++j) {
      acc += raw_attn[base + row * n + j] * solve_attn[base + j * n + col];
    }
    solve_attn[base + row * n + col] = acc;
  }
}
)METAL";

class SolveStrictlyLowerMetalPrimitive : public mx::UnaryPrimitive {
 public:
  SolveStrictlyLowerMetalPrimitive(
      mx::Stream stream,
      int matrix_count,
      int chunk_size)
      : mx::UnaryPrimitive(stream),
        matrix_count_(matrix_count),
        chunk_size_(chunk_size) {}

  void eval_cpu(const std::vector<mx::array>&, mx::array&) override {
    throw std::runtime_error("SolveStrictlyLowerMetalPrimitive is Metal-only");
  }

  void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override {
    if (inputs.size() != 1) {
      throw std::runtime_error("SolveStrictlyLowerMetalPrimitive expects 1 input");
    }
    if (inputs[0].dtype() != mx::float32 || out.dtype() != mx::float32) {
      throw std::runtime_error("SolveStrictlyLowerMetalPrimitive requires float32 tensors");
    }
    if (!mx::metal::is_available()) {
      throw std::runtime_error("SolveStrictlyLowerMetalPrimitive requires Metal");
    }

    auto& device = mx::metal::device(stream().device);
    auto& encoder = device.get_command_encoder(stream().index);
    out.set_data(mx::allocator::malloc(out.nbytes()));

    auto* library = device.get_library(
        "mixlab_gated_delta_metal_solve",
        []() { return std::string(kGatedDeltaMetalSolveSource); });
    auto* kernel = device.get_kernel(
        "gated_delta_chunk_solve_metal",
        library);

    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(inputs[0], 0);
    encoder.set_output_array(out, 1);
    encoder.set_bytes(chunk_size_, 2);
    encoder.dispatch_threads(
        MTL::Size::Make(
            static_cast<NS::UInteger>(chunk_size_),
            static_cast<NS::UInteger>(matrix_count_),
            1),
        MTL::Size::Make(
            static_cast<NS::UInteger>(std::min(chunk_size_, 64)),
            1,
            1));
  }

  std::vector<mx::array> vjp(
      const std::vector<mx::array>&,
      const std::vector<mx::array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<mx::array>& outputs) override {
    if (cotangents.size() != 1 || outputs.size() != 1) {
      throw std::runtime_error("SolveStrictlyLowerMetalPrimitive vjp expects one cotangent and one output");
    }
    auto solve_t = mx::transpose(outputs[0], {0, 2, 1});
    auto grad = mx::matmul(mx::matmul(solve_t, cotangents[0]), solve_t);
    grad = mx::tril(grad, -1);

    std::vector<mx::array> grads;
    grads.reserve(argnums.size());
    for (int argnum : argnums) {
      if (argnum != 0) {
        throw std::runtime_error("SolveStrictlyLowerMetalPrimitive vjp argnum out of range");
      }
      grads.push_back(grad);
    }
    return grads;
  }

  const char* name() const override {
    return "SolveStrictlyLowerMetalPrimitive";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    auto* rhs = dynamic_cast<const SolveStrictlyLowerMetalPrimitive*>(&other);
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

#endif

} // namespace

bool gated_delta_metal_primitive_available() {
#ifdef __APPLE__
  return mx::metal::is_available();
#else
  return false;
#endif
}

mx::array solve_strictly_lower_metal_primitive(
    const mx::array& raw_attn,
    int matrix_count,
    int chunk_size) {
#ifdef __APPLE__
  auto stream = mx::default_stream(mx::default_device());
  auto primitive = std::make_shared<SolveStrictlyLowerMetalPrimitive>(
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
#else
  (void)raw_attn;
  (void)matrix_count;
  (void)chunk_size;
  throw std::runtime_error("Metal gated delta solve primitive is unavailable on this platform");
#endif
}

} // namespace mlx_ir
