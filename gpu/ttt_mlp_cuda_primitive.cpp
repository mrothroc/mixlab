#include "ttt_mlp_cuda_primitive.h"
#include "cuda_kernel_dispatch.h"

#include <mlx/device.h>
#include <mlx/ops.h>
#include <mlx/primitives.h>

#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace mx = mlx::core;

namespace mlx_ir {
namespace {

class TTTMLPCausalConvCUDAPrimitive : public mx::Primitive {
 public:
  TTTMLPCausalConvCUDAPrimitive(mx::Stream stream, int batch, int token_count, int width)
      : mx::Primitive(stream), batch_(batch), token_count_(token_count), width_(width) {}

  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("TTTMLPCausalConvCUDAPrimitive is CUDA-only");
  }

  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override {
    if (inputs.size() != 3 || outputs.size() != 1) {
      throw std::runtime_error("TTTMLPCausalConvCUDAPrimitive expects 3 inputs and 1 output");
    }
    for (const auto& input : inputs) {
      if (input.dtype() != mx::float32) {
        throw std::runtime_error("TTTMLPCausalConvCUDAPrimitive requires float32 inputs");
      }
    }
    const int elements = batch_ * token_count_ * width_;
    const int threads = 256;
    const int blocks = (elements + threads - 1) / threads;
    launch_precompiled_cuda_kernel_into(
        "ttt_mlp_causal_conv",
        inputs,
        {&outputs[0]},
        {batch_, token_count_, width_},
        std::make_tuple(blocks, 1, 1),
        std::make_tuple(threads, 1, 1),
        stream());
  }

  const char* name() const override { return "TTTMLPCausalConvCUDAPrimitive"; }

  bool is_equivalent(const mx::Primitive& other) const override {
    auto* rhs = dynamic_cast<const TTTMLPCausalConvCUDAPrimitive*>(&other);
    return rhs != nullptr && batch_ == rhs->batch_ &&
        token_count_ == rhs->token_count_ && width_ == rhs->width_;
  }

  std::vector<mx::Shape> output_shapes(const std::vector<mx::array>&) override {
    return {mx::Shape{
        static_cast<mx::ShapeElem>(batch_),
        static_cast<mx::ShapeElem>(token_count_),
        static_cast<mx::ShapeElem>(width_)}};
  }

 private:
  int batch_;
  int token_count_;
  int width_;
};

} // namespace

bool ttt_mlp_causal_conv_cuda_primitive_available() {
  const char* disabled = std::getenv("MIXLAB_TTT_MLP_DISABLE_CUDA_PRIMITIVE");
  if (disabled != nullptr && std::string(disabled) == "1") {
    return false;
  }
#ifdef __linux__
  return mx::is_available(mx::Device::gpu);
#else
  return false;
#endif
}

mx::array ttt_mlp_causal_conv_cuda(
    const mx::array& x,
    const mx::array& history,
    const mx::array& weight,
    int batch,
    int token_count,
    int width) {
  if (batch <= 0 || token_count <= 0 || width <= 0) {
    throw std::runtime_error("TTT-MLP CUDA causal convolution has invalid shape");
  }
  auto stream = mx::default_stream(mx::default_device());
  auto primitive = std::make_shared<TTTMLPCausalConvCUDAPrimitive>(
      stream, batch, token_count, width);
  return mx::array(
      mx::Shape{
          static_cast<mx::ShapeElem>(batch),
          static_cast<mx::ShapeElem>(token_count),
          static_cast<mx::ShapeElem>(width)},
      mx::float32,
      primitive,
      std::vector<mx::array>{
          mx::contiguous(x),
          mx::contiguous(history),
          mx::contiguous(weight)});
}

} // namespace mlx_ir
