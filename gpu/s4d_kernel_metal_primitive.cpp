#include "s4d_kernel_metal_primitive.h"

#include <mlx/allocator.h>
#include <mlx/device.h>
#include <mlx/ops.h>
#include <mlx/primitives.h>
#include <mlx/transforms.h>
#include <mlx/version.h>

#include <atomic>
#include <cstdlib>
#include <iostream>
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

constexpr int kS4DKernelMetalThreads = 256;

void validate_s4d_kernel_shape(int D, int T, int state_pairs) {
  if (D <= 0 || T <= 0 || state_pairs <= 0) {
    throw std::runtime_error(
        "S4D bidirectional Metal kernel requires positive D,T,state_pairs");
  }
}

std::vector<mx::array> contiguous_s4d_kernel_inputs(
    const std::vector<mx::array>& inputs,
    int D,
    int state_pairs) {
  if (inputs.size() != 8) {
    throw std::runtime_error(
        "S4D bidirectional Metal kernel requires eight inputs");
  }
  const size_t expected = static_cast<size_t>(D * state_pairs);
  std::vector<mx::array> contiguous;
  contiguous.reserve(inputs.size());
  for (const auto& input : inputs) {
    if (input.dtype() != mx::float32 || input.size() != expected) {
      throw std::runtime_error(
          "S4D bidirectional Metal kernel inputs must be float32 [D,state_pairs]");
    }
    contiguous.push_back(mx::contiguous(input));
  }
  return contiguous;
}

void log_s4d_kernel_metal_once() {
  static std::atomic<bool> logged{false};
  if (!logged.exchange(true)) {
    std::cerr
        << "[mlx_ir] S4D bidirectional kernel using native Metal forward/backward primitive"
        << " (set MIXLAB_S4D_DISABLE_METAL_KERNEL_PRIMITIVE=1 to use the MLX fallback)"
        << std::endl;
  }
}

#ifdef __APPLE__

const char* kS4DKernelMetalSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

kernel void s4d_bidirectional_kernel_forward_metal(
    device const float* b_real [[buffer(0)]],
    device const float* b_imag [[buffer(1)]],
    device const float* log_magnitude [[buffer(2)]],
    device const float* phase [[buffer(3)]],
    device const float* c_forward_real [[buffer(4)]],
    device const float* c_forward_imag [[buffer(5)]],
    device const float* c_backward_real [[buffer(6)]],
    device const float* c_backward_imag [[buffer(7)]],
    device float* output [[buffer(8)]],
    constant int& T [[buffer(9)]],
    constant int& D [[buffer(10)]],
    constant int& state_pairs [[buffer(11)]],
    uint2 gid [[thread_position_in_grid]]) {
  const int position = int(gid.x);
  const int d = int(gid.y);
  if (position >= T || d >= D) {
    return;
  }
  float forward_sum = 0.0f;
  float backward_sum = 0.0f;
  for (int n = 0; n < state_pairs; ++n) {
    const int index = d * state_pairs + n;
    const float magnitude = exp(log_magnitude[index] * float(position));
    const float angle = phase[index] * float(position);
    const float power_real = magnitude * cos(angle);
    const float power_imag = magnitude * sin(angle);
    const float forward_gamma_real =
        c_forward_real[index] * b_real[index] -
        c_forward_imag[index] * b_imag[index];
    const float forward_gamma_imag =
        c_forward_real[index] * b_imag[index] +
        c_forward_imag[index] * b_real[index];
    const float backward_gamma_real =
        c_backward_real[index] * b_real[index] -
        c_backward_imag[index] * b_imag[index];
    const float backward_gamma_imag =
        c_backward_real[index] * b_imag[index] +
        c_backward_imag[index] * b_real[index];
    forward_sum +=
        forward_gamma_real * power_real - forward_gamma_imag * power_imag;
    backward_sum +=
        backward_gamma_real * power_real - backward_gamma_imag * power_imag;
  }
  output[d * (2 * T) + position] = 2.0f * forward_sum;
  output[d * (2 * T) + (2 * T - 1 - position)] = 2.0f * backward_sum;
}

kernel void s4d_bidirectional_kernel_backward_metal(
    device const float* b_real [[buffer(0)]],
    device const float* b_imag [[buffer(1)]],
    device const float* log_magnitude [[buffer(2)]],
    device const float* phase [[buffer(3)]],
    device const float* c_forward_real [[buffer(4)]],
    device const float* c_forward_imag [[buffer(5)]],
    device const float* c_backward_real [[buffer(6)]],
    device const float* c_backward_imag [[buffer(7)]],
    device const float* cotangent [[buffer(8)]],
    device float* grad_b_real [[buffer(9)]],
    device float* grad_b_imag [[buffer(10)]],
    device float* grad_log_magnitude [[buffer(11)]],
    device float* grad_phase [[buffer(12)]],
    device float* grad_c_forward_real [[buffer(13)]],
    device float* grad_c_forward_imag [[buffer(14)]],
    device float* grad_c_backward_real [[buffer(15)]],
    device float* grad_c_backward_imag [[buffer(16)]],
    constant int& T [[buffer(17)]],
    constant int& D [[buffer(18)]],
    constant int& state_pairs [[buffer(19)]],
    uint gid [[thread_position_in_grid]]) {
  const int total = D * state_pairs;
  if (int(gid) >= total) {
    return;
  }
  const int index = int(gid);
  const int d = index / state_pairs;
  const float br = b_real[index];
  const float bi = b_imag[index];
  const float cfr = c_forward_real[index];
  const float cfi = c_forward_imag[index];
  const float cbr = c_backward_real[index];
  const float cbi = c_backward_imag[index];
  const float gamma_forward_real = cfr * br - cfi * bi;
  const float gamma_forward_imag = cfr * bi + cfi * br;
  const float gamma_backward_real = cbr * br - cbi * bi;
  const float gamma_backward_imag = cbr * bi + cbi * br;

  float grad_gamma_forward_real = 0.0f;
  float grad_gamma_forward_imag = 0.0f;
  float grad_gamma_backward_real = 0.0f;
  float grad_gamma_backward_imag = 0.0f;
  float grad_log = 0.0f;
  float grad_angle = 0.0f;
  for (int position = 0; position < T; ++position) {
    const float magnitude = exp(log_magnitude[index] * float(position));
    const float angle = phase[index] * float(position);
    const float power_real = magnitude * cos(angle);
    const float power_imag = magnitude * sin(angle);
    const float grad_forward = cotangent[d * (2 * T) + position];
    const float grad_backward =
        cotangent[d * (2 * T) + (2 * T - 1 - position)];
    grad_gamma_forward_real += 2.0f * grad_forward * power_real;
    grad_gamma_forward_imag -= 2.0f * grad_forward * power_imag;
    grad_gamma_backward_real += 2.0f * grad_backward * power_real;
    grad_gamma_backward_imag -= 2.0f * grad_backward * power_imag;
    const float weighted_forward =
        gamma_forward_real * power_real - gamma_forward_imag * power_imag;
    const float weighted_backward =
        gamma_backward_real * power_real - gamma_backward_imag * power_imag;
    grad_log += 2.0f * float(position) *
        (grad_forward * weighted_forward + grad_backward * weighted_backward);
    grad_angle -= 2.0f * float(position) *
        (grad_forward *
             (gamma_forward_real * power_imag + gamma_forward_imag * power_real) +
         grad_backward *
             (gamma_backward_real * power_imag + gamma_backward_imag * power_real));
  }

  grad_b_real[index] =
      grad_gamma_forward_real * cfr + grad_gamma_forward_imag * cfi +
      grad_gamma_backward_real * cbr + grad_gamma_backward_imag * cbi;
  grad_b_imag[index] =
      -grad_gamma_forward_real * cfi + grad_gamma_forward_imag * cfr -
      grad_gamma_backward_real * cbi + grad_gamma_backward_imag * cbr;
  grad_log_magnitude[index] = grad_log;
  grad_phase[index] = grad_angle;
  grad_c_forward_real[index] =
      grad_gamma_forward_real * br + grad_gamma_forward_imag * bi;
  grad_c_forward_imag[index] =
      -grad_gamma_forward_real * bi + grad_gamma_forward_imag * br;
  grad_c_backward_real[index] =
      grad_gamma_backward_real * br + grad_gamma_backward_imag * bi;
  grad_c_backward_imag[index] =
      -grad_gamma_backward_real * bi + grad_gamma_backward_imag * br;
}
)METAL";

class S4DBidirectionalKernelForwardMetalPrimitive : public mx::Primitive {
 public:
  S4DBidirectionalKernelForwardMetalPrimitive(
      mx::Stream stream,
      int D,
      int T,
      int state_pairs)
      : mx::Primitive(stream), D_(D), T_(T), state_pairs_(state_pairs) {}

  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error(
        "S4DBidirectionalKernelForwardMetalPrimitive is Metal-only");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    if (inputs.size() != 8 || outputs.size() != 1) {
      throw std::runtime_error(
          "S4D bidirectional Metal forward has invalid input/output count");
    }
    validate_s4d_kernel_shape(D_, T_, state_pairs_);
    outputs[0].set_data(mx::allocator::malloc(outputs[0].nbytes()));
    auto& device = mx::metal::device(stream().device);
#if MLX_VERSION_NUMERIC >= 31002
    auto& encoder = mx::metal::get_command_encoder(stream());
#else
    auto& encoder = device.get_command_encoder(stream().index);
#endif
    auto* library = device.get_library(
        "mixlab_s4d_bidirectional_kernel",
        []() { return std::string(kS4DKernelMetalSource); });
    auto* kernel = device.get_kernel(
        "s4d_bidirectional_kernel_forward_metal", library);
    encoder.set_compute_pipeline_state(kernel);
    for (int i = 0; i < 8; ++i) {
      encoder.set_input_array(inputs[static_cast<size_t>(i)], i);
    }
    encoder.set_output_array(outputs[0], 8);
    encoder.set_bytes(T_, 9);
    encoder.set_bytes(D_, 10);
    encoder.set_bytes(state_pairs_, 11);
    encoder.dispatch_threads(
        MTL::Size::Make(
            static_cast<NS::UInteger>(T_),
            static_cast<NS::UInteger>(D_),
            1),
        MTL::Size::Make(kS4DKernelMetalThreads, 1, 1));
  }

  const char* name() const override {
    return "S4DBidirectionalKernelForwardMetalPrimitive";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    auto* rhs =
        dynamic_cast<const S4DBidirectionalKernelForwardMetalPrimitive*>(&other);
    return rhs != nullptr &&
        D_ == rhs->D_ && T_ == rhs->T_ && state_pairs_ == rhs->state_pairs_;
  }

  std::vector<mx::Shape> output_shapes(const std::vector<mx::array>&) override {
    return {mx::Shape{
        static_cast<mx::ShapeElem>(D_),
        static_cast<mx::ShapeElem>(2 * T_)}};
  }

 private:
  int D_;
  int T_;
  int state_pairs_;
};

class S4DBidirectionalKernelBackwardMetalPrimitive : public mx::Primitive {
 public:
  S4DBidirectionalKernelBackwardMetalPrimitive(
      mx::Stream stream,
      int D,
      int T,
      int state_pairs)
      : mx::Primitive(stream), D_(D), T_(T), state_pairs_(state_pairs) {}

  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error(
        "S4DBidirectionalKernelBackwardMetalPrimitive is Metal-only");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    if (inputs.size() != 9 || outputs.size() != 8) {
      throw std::runtime_error(
          "S4D bidirectional Metal backward has invalid input/output count");
    }
    validate_s4d_kernel_shape(D_, T_, state_pairs_);
    for (auto& output : outputs) {
      output.set_data(mx::allocator::malloc(output.nbytes()));
    }
    auto& device = mx::metal::device(stream().device);
#if MLX_VERSION_NUMERIC >= 31002
    auto& encoder = mx::metal::get_command_encoder(stream());
#else
    auto& encoder = device.get_command_encoder(stream().index);
#endif
    auto* library = device.get_library(
        "mixlab_s4d_bidirectional_kernel",
        []() { return std::string(kS4DKernelMetalSource); });
    auto* kernel = device.get_kernel(
        "s4d_bidirectional_kernel_backward_metal", library);
    encoder.set_compute_pipeline_state(kernel);
    for (int i = 0; i < 9; ++i) {
      encoder.set_input_array(inputs[static_cast<size_t>(i)], i);
    }
    for (int i = 0; i < 8; ++i) {
      encoder.set_output_array(outputs[static_cast<size_t>(i)], 9 + i);
    }
    encoder.set_bytes(T_, 17);
    encoder.set_bytes(D_, 18);
    encoder.set_bytes(state_pairs_, 19);
    encoder.dispatch_threads(
        MTL::Size::Make(
            static_cast<NS::UInteger>(D_ * state_pairs_), 1, 1),
        MTL::Size::Make(kS4DKernelMetalThreads, 1, 1));
  }

  const char* name() const override {
    return "S4DBidirectionalKernelBackwardMetalPrimitive";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    auto* rhs =
        dynamic_cast<const S4DBidirectionalKernelBackwardMetalPrimitive*>(&other);
    return rhs != nullptr &&
        D_ == rhs->D_ && T_ == rhs->T_ && state_pairs_ == rhs->state_pairs_;
  }

  std::vector<mx::Shape> output_shapes(const std::vector<mx::array>&) override {
    std::vector<mx::Shape> shapes(
        8,
        mx::Shape{
            static_cast<mx::ShapeElem>(D_),
            static_cast<mx::ShapeElem>(state_pairs_)});
    return shapes;
  }

 private:
  int D_;
  int T_;
  int state_pairs_;
};

mx::array s4d_bidirectional_kernel_metal_forward(
    const std::vector<mx::array>& raw_inputs,
    int D,
    int T,
    int state_pairs) {
  auto inputs = contiguous_s4d_kernel_inputs(raw_inputs, D, state_pairs);
  auto stream = mx::default_stream(mx::default_device());
  auto primitive = std::make_shared<S4DBidirectionalKernelForwardMetalPrimitive>(
      stream, D, T, state_pairs);
  return mx::array(
      {D, 2 * T},
      mx::float32,
      primitive,
      inputs);
}

std::vector<mx::array> s4d_bidirectional_kernel_metal_vjp(
    const std::vector<mx::array>& raw_inputs,
    const std::vector<mx::array>& cotangents,
    int D,
    int T,
    int state_pairs) {
  if (cotangents.size() != 1) {
    throw std::runtime_error(
        "S4D bidirectional Metal backward expects one cotangent");
  }
  auto inputs = contiguous_s4d_kernel_inputs(raw_inputs, D, state_pairs);
  inputs.push_back(mx::contiguous(cotangents[0]));
  auto stream = mx::default_stream(mx::default_device());
  auto primitive = std::make_shared<S4DBidirectionalKernelBackwardMetalPrimitive>(
      stream, D, T, state_pairs);
  std::vector<mx::Shape> shapes(
      8,
      mx::Shape{
          static_cast<mx::ShapeElem>(D),
          static_cast<mx::ShapeElem>(state_pairs)});
  return mx::array::make_arrays(
      shapes,
      std::vector<mx::Dtype>(8, mx::float32),
      primitive,
      inputs);
}

#endif

} // namespace

bool s4d_bidirectional_kernel_metal_primitive_available() {
  const char* disabled =
      std::getenv("MIXLAB_S4D_DISABLE_METAL_KERNEL_PRIMITIVE");
  if (disabled != nullptr && std::string(disabled) == "1") {
    return false;
  }
#ifdef __APPLE__
  return mx::metal::is_available();
#else
  return false;
#endif
}

mx::array s4d_bidirectional_kernel_metal_primitive(
    const mx::array& b_real,
    const mx::array& b_imag,
    const mx::array& log_magnitude,
    const mx::array& phase,
    const mx::array& c_forward_real,
    const mx::array& c_forward_imag,
    const mx::array& c_backward_real,
    const mx::array& c_backward_imag,
    int D,
    int T,
    int state_pairs) {
#ifdef __APPLE__
  validate_s4d_kernel_shape(D, T, state_pairs);
  if (!s4d_bidirectional_kernel_metal_primitive_available()) {
    throw std::runtime_error(
        "S4D bidirectional Metal kernel primitive is unavailable");
  }
  log_s4d_kernel_metal_once();
  auto kernel = mx::custom_vjp(
      [D, T, state_pairs](const std::vector<mx::array>& inputs) {
        return std::vector<mx::array>{
            s4d_bidirectional_kernel_metal_forward(
                inputs, D, T, state_pairs)};
      },
      [D, T, state_pairs](
          const std::vector<mx::array>& inputs,
          const std::vector<mx::array>& cotangents,
          const std::vector<mx::array>&) {
        return s4d_bidirectional_kernel_metal_vjp(
            inputs, cotangents, D, T, state_pairs);
      });
  return kernel({
      b_real,
      b_imag,
      log_magnitude,
      phase,
      c_forward_real,
      c_forward_imag,
      c_backward_real,
      c_backward_imag})[0];
#else
  (void)b_real;
  (void)b_imag;
  (void)log_magnitude;
  (void)phase;
  (void)c_forward_real;
  (void)c_forward_imag;
  (void)c_backward_real;
  (void)c_backward_imag;
  (void)D;
  (void)T;
  (void)state_pairs;
  throw std::runtime_error(
      "S4D bidirectional Metal kernel primitive is unavailable");
#endif
}

} // namespace mlx_ir
