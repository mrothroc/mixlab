#include "mamba3_cuda_primitive.h"
#include "cuda_kernel_dispatch.h"

#include <mlx/device.h>
#include <mlx/ops.h>
#include <mlx/primitives.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace mx = mlx::core;

namespace mlx_ir {
namespace {

constexpr int kMamba3CUDAThreads = 32;
constexpr int kMamba3DefaultBackwardWindow = 64;

bool env_is_one(const char* name) {
  const char* raw = std::getenv(name);
  return raw != nullptr && std::string(raw) == "1";
}

int mamba3_backward_window_size(int T) {
  const char* raw = std::getenv("MIXLAB_MAMBA3_BWD_WINDOW");
  if (raw == nullptr || raw[0] == '\0') {
    return std::min(T, kMamba3DefaultBackwardWindow);
  }
  char* end = nullptr;
  long parsed = std::strtol(raw, &end, 10);
  if (end == raw || (end && *end != '\0') || parsed <= 0) {
    return std::min(T, kMamba3DefaultBackwardWindow);
  }
  return std::max(
      1,
      std::min(
          std::min(T, kMamba3DefaultBackwardWindow),
          static_cast<int>(parsed)));
}

void validate_mamba3_cuda_shape(int B, int T, int D, int N, int G) {
  if (B <= 0 || T <= 0 || D <= 0 || N <= 0 || G <= 0) {
    throw std::runtime_error("Mamba3 CUDA primitive requires positive B,T,D,N,G");
  }
  if ((D % G) != 0) {
    throw std::runtime_error("Mamba3 CUDA primitive requires D divisible by G");
  }
  if ((N % 2) != 0) {
    throw std::runtime_error("Mamba3 CUDA primitive requires even state_size");
  }
  if ((N / 2) > kMamba3CUDAThreads) {
    throw std::runtime_error("Mamba3 CUDA primitive supports state_size <= 64");
  }
}

std::vector<mx::array> contiguous_mamba3_inputs(
    const mx::array& x_flat,
    const mx::array& dt_flat,
    const mx::array& lambda_flat,
    const mx::array& theta_flat,
    const mx::array& a_log,
    const mx::array& b_proj_flat,
    const mx::array& c_proj_flat) {
  return {
      mx::contiguous(x_flat),
      mx::contiguous(dt_flat),
      mx::contiguous(lambda_flat),
      mx::contiguous(theta_flat),
      mx::contiguous(a_log),
      mx::contiguous(b_proj_flat),
      mx::contiguous(c_proj_flat)};
}

void require_float32_inputs(const std::vector<mx::array>& inputs, const char* name) {
  for (const auto& input : inputs) {
    if (input.dtype() != mx::float32) {
      throw std::runtime_error(std::string(name) + " requires float32 inputs");
    }
  }
}

void require_float32_outputs(const std::vector<mx::array>& outputs, const char* name) {
  for (const auto& output : outputs) {
    if (output.dtype() != mx::float32) {
      throw std::runtime_error(std::string(name) + " requires float32 outputs");
    }
  }
}

void log_mamba3_cuda_once() {
  static std::atomic<bool> logged{false};
  if (!logged.exchange(true)) {
    std::cerr << "[mlx_ir] canonical Mamba3 scan using fused CUDA primitive"
              << " (set MIXLAB_MAMBA3_DISABLE_CUDA_PRIMITIVE=1 only for small debug fallback runs)"
              << std::endl;
  }
}

class Mamba3SelectiveScanCUDAForwardPrimitive : public mx::UnaryPrimitive {
 public:
  Mamba3SelectiveScanCUDAForwardPrimitive(
      mx::Stream stream,
      int B,
      int T,
      int D,
      int N,
      int G)
      : mx::UnaryPrimitive(stream), B_(B), T_(T), D_(D), N_(N), G_(G) {}

  void eval_cpu(const std::vector<mx::array>&, mx::array&) override {
    throw std::runtime_error("Mamba3SelectiveScanCUDAForwardPrimitive is CUDA-only");
  }

  void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override {
    if (inputs.size() != 7) {
      throw std::runtime_error("Mamba3SelectiveScanCUDAForwardPrimitive expects 7 inputs");
    }
    require_float32_inputs(inputs, "Mamba3SelectiveScanCUDAForwardPrimitive");
    if (out.dtype() != mx::float32) {
      throw std::runtime_error("Mamba3SelectiveScanCUDAForwardPrimitive requires float32 output");
    }
    validate_mamba3_cuda_shape(B_, T_, D_, N_, G_);

    launch_precompiled_cuda_kernel_into(
        "mamba3_selective_scan_fwd",
        inputs,
        {&out},
        {B_, T_, D_, N_, G_},
        std::make_tuple(D_, B_, 1),
        std::make_tuple(kMamba3CUDAThreads, 1, 1),
        stream());
  }

  const char* name() const override {
    return "Mamba3SelectiveScanCUDAForwardPrimitive";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    auto* rhs = dynamic_cast<const Mamba3SelectiveScanCUDAForwardPrimitive*>(&other);
    return rhs != nullptr &&
        B_ == rhs->B_ &&
        T_ == rhs->T_ &&
        D_ == rhs->D_ &&
        N_ == rhs->N_ &&
        G_ == rhs->G_;
  }

  std::vector<mx::Shape> output_shapes(const std::vector<mx::array>&) override {
    return {mx::Shape{
        static_cast<mx::ShapeElem>(B_ * T_),
        static_cast<mx::ShapeElem>(D_)}};
  }

 private:
  int B_;
  int T_;
  int D_;
  int N_;
  int G_;
};

class Mamba3SelectiveScanCUDABackwardPrimitive : public mx::Primitive {
 public:
  Mamba3SelectiveScanCUDABackwardPrimitive(
      mx::Stream stream,
      int B,
      int T,
      int D,
      int N,
      int G)
      : mx::Primitive(stream), B_(B), T_(T), D_(D), N_(N), G_(G) {}

  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("Mamba3SelectiveScanCUDABackwardPrimitive is CUDA-only");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    if (inputs.size() != 8) {
      throw std::runtime_error("Mamba3SelectiveScanCUDABackwardPrimitive expects 8 inputs");
    }
    if (outputs.size() != 9) {
      throw std::runtime_error("Mamba3SelectiveScanCUDABackwardPrimitive expects 9 outputs");
    }
    require_float32_inputs(inputs, "Mamba3SelectiveScanCUDABackwardPrimitive");
    require_float32_outputs(outputs, "Mamba3SelectiveScanCUDABackwardPrimitive");
    validate_mamba3_cuda_shape(B_, T_, D_, N_, G_);

    const int BT = B_ * T_;
    const int K = N_ / 2;
    const int window_size = mamba3_backward_window_size(T_);
    const int n_windows = (T_ + window_size - 1) / window_size;
    const int checkpoint_rows = B_ * n_windows;
    const int h_checkpoints_size = checkpoint_rows * D_ * N_;
    const int phi_checkpoints_size = checkpoint_rows * D_ * K;
    const int grad_x_size = BT * D_;
    const int grad_dt_size = BT * D_;
    const int grad_lambda_size = BT * D_;
    const int grad_theta_size = BT * D_ * K;
    const int grad_a_log_size = D_ * N_;
    const int grad_b_size = BT * G_ * N_;
    const int grad_c_size = BT * G_ * N_;
    const int max_size = std::max(
        {h_checkpoints_size,
         phi_checkpoints_size,
         grad_x_size,
         grad_dt_size,
         grad_lambda_size,
         grad_theta_size,
         grad_a_log_size,
         grad_b_size,
         grad_c_size});
    const int zero_threads = 256;
    const int zero_blocks = (max_size + zero_threads - 1) / zero_threads;
    launch_precompiled_cuda_kernel_into(
        "mamba3_selective_scan_zero",
        {},
        {&outputs[0], &outputs[1], &outputs[2], &outputs[3],
         &outputs[4], &outputs[5], &outputs[6], &outputs[7], &outputs[8]},
        {h_checkpoints_size,
         phi_checkpoints_size,
         grad_x_size,
         grad_dt_size,
         grad_lambda_size,
         grad_theta_size,
         grad_a_log_size,
         grad_b_size,
         grad_c_size},
        std::make_tuple(zero_blocks, 1, 1),
        std::make_tuple(zero_threads, 1, 1),
        stream());

    std::vector<mx::array> scan_inputs(inputs.begin(), inputs.begin() + 7);
    launch_precompiled_cuda_kernel_into(
        "mamba3_selective_scan_checkpoints",
        scan_inputs,
        {&outputs[0], &outputs[1]},
        {B_, T_, D_, N_, G_, window_size, n_windows},
        std::make_tuple(D_, B_, 1),
        std::make_tuple(kMamba3CUDAThreads, 1, 1),
        stream(),
        0,
        false);

    std::vector<mx::array> backward_inputs = inputs;
    backward_inputs.push_back(outputs[0]);
    backward_inputs.push_back(outputs[1]);
    launch_precompiled_cuda_kernel_into(
        "mamba3_selective_scan_bwd",
        backward_inputs,
        {&outputs[2], &outputs[3], &outputs[4], &outputs[5],
         &outputs[6], &outputs[7], &outputs[8]},
        {B_, T_, D_, N_, G_, window_size, n_windows},
        std::make_tuple(D_, B_, 1),
        std::make_tuple(kMamba3CUDAThreads, 1, 1),
        stream(),
        0,
        false);
  }

  const char* name() const override {
    return "Mamba3SelectiveScanCUDABackwardPrimitive";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    auto* rhs = dynamic_cast<const Mamba3SelectiveScanCUDABackwardPrimitive*>(&other);
    return rhs != nullptr &&
        B_ == rhs->B_ &&
        T_ == rhs->T_ &&
        D_ == rhs->D_ &&
        N_ == rhs->N_ &&
        G_ == rhs->G_;
  }

  std::vector<mx::Shape> output_shapes(const std::vector<mx::array>&) override {
    const int BT = B_ * T_;
    const int K = N_ / 2;
    const int window_size = mamba3_backward_window_size(T_);
    const int n_windows = (T_ + window_size - 1) / window_size;
    const int checkpoint_rows = B_ * n_windows;
    return {
        mx::Shape{
            static_cast<mx::ShapeElem>(checkpoint_rows),
            static_cast<mx::ShapeElem>(D_),
            static_cast<mx::ShapeElem>(N_)},
        mx::Shape{
            static_cast<mx::ShapeElem>(checkpoint_rows),
            static_cast<mx::ShapeElem>(D_ * K)},
        mx::Shape{
            static_cast<mx::ShapeElem>(BT),
            static_cast<mx::ShapeElem>(D_)},
        mx::Shape{
            static_cast<mx::ShapeElem>(BT),
            static_cast<mx::ShapeElem>(D_)},
        mx::Shape{
            static_cast<mx::ShapeElem>(BT),
            static_cast<mx::ShapeElem>(D_)},
        mx::Shape{
            static_cast<mx::ShapeElem>(BT),
            static_cast<mx::ShapeElem>(D_ * K)},
        mx::Shape{
            static_cast<mx::ShapeElem>(D_),
            static_cast<mx::ShapeElem>(N_)},
        mx::Shape{
            static_cast<mx::ShapeElem>(BT),
            static_cast<mx::ShapeElem>(G_ * N_)},
        mx::Shape{
            static_cast<mx::ShapeElem>(BT),
            static_cast<mx::ShapeElem>(G_ * N_)}};
  }

 private:
  int B_;
  int T_;
  int D_;
  int N_;
  int G_;
};

} // namespace

bool mamba3_selective_scan_cuda_primitive_available(int state_size) {
  if (env_is_one("MIXLAB_MAMBA3_DISABLE_CUDA_PRIMITIVE")) {
    return false;
  }
  if (state_size <= 0 || (state_size % 2) != 0 || (state_size / 2) > kMamba3CUDAThreads) {
    return false;
  }
#ifdef __linux__
  return mx::is_available(mx::Device::gpu);
#else
  return false;
#endif
}

mx::array mamba3_selective_scan_cuda_forward(
    const mx::array& x_flat,
    const mx::array& dt_flat,
    const mx::array& lambda_flat,
    const mx::array& theta_flat,
    const mx::array& a_log,
    const mx::array& b_proj_flat,
    const mx::array& c_proj_flat,
    int B,
    int T,
    int D,
    int N,
    int G) {
  validate_mamba3_cuda_shape(B, T, D, N, G);
  log_mamba3_cuda_once();
  auto stream = mx::default_stream(mx::default_device());
  auto primitive = std::make_shared<Mamba3SelectiveScanCUDAForwardPrimitive>(
      stream, B, T, D, N, G);
  return mx::array(
      mx::Shape{
          static_cast<mx::ShapeElem>(B * T),
          static_cast<mx::ShapeElem>(D)},
      mx::float32,
      primitive,
      contiguous_mamba3_inputs(
          x_flat, dt_flat, lambda_flat, theta_flat, a_log, b_proj_flat, c_proj_flat));
}

std::vector<mx::array> mamba3_selective_scan_cuda_vjp(
    const std::vector<mx::array>& args,
    const std::vector<mx::array>& cotangents,
    int B,
    int T,
    int D,
    int N,
    int G) {
  if (args.size() != 7 || cotangents.size() != 1) {
    throw std::runtime_error("mamba3_selective_scan_cuda_vjp expects 7 args and 1 cotangent");
  }
  validate_mamba3_cuda_shape(B, T, D, N, G);
  std::vector<mx::array> inputs = contiguous_mamba3_inputs(
      args[0], args[1], args[2], args[3], args[4], args[5], args[6]);
  inputs.push_back(mx::contiguous(cotangents[0]));

  auto stream = mx::default_stream(mx::default_device());
  auto primitive = std::make_shared<Mamba3SelectiveScanCUDABackwardPrimitive>(
      stream, B, T, D, N, G);
  const int BT = B * T;
  const int K = N / 2;
  const int window_size = mamba3_backward_window_size(T);
  const int n_windows = (T + window_size - 1) / window_size;
  const int checkpoint_rows = B * n_windows;
  std::vector<mx::Shape> shapes = {
      mx::Shape{
          static_cast<mx::ShapeElem>(checkpoint_rows),
          static_cast<mx::ShapeElem>(D),
          static_cast<mx::ShapeElem>(N)},
      mx::Shape{
          static_cast<mx::ShapeElem>(checkpoint_rows),
          static_cast<mx::ShapeElem>(D * K)},
      mx::Shape{
          static_cast<mx::ShapeElem>(BT),
          static_cast<mx::ShapeElem>(D)},
      mx::Shape{
          static_cast<mx::ShapeElem>(BT),
          static_cast<mx::ShapeElem>(D)},
      mx::Shape{
          static_cast<mx::ShapeElem>(BT),
          static_cast<mx::ShapeElem>(D)},
      mx::Shape{
          static_cast<mx::ShapeElem>(BT),
          static_cast<mx::ShapeElem>(D * K)},
      mx::Shape{
          static_cast<mx::ShapeElem>(D),
          static_cast<mx::ShapeElem>(N)},
      mx::Shape{
          static_cast<mx::ShapeElem>(BT),
          static_cast<mx::ShapeElem>(G * N)},
      mx::Shape{
          static_cast<mx::ShapeElem>(BT),
          static_cast<mx::ShapeElem>(G * N)}};
  std::vector<mx::Dtype> dtypes(shapes.size(), mx::float32);
  auto outputs = mx::array::make_arrays(shapes, dtypes, primitive, inputs);
  return {
      outputs[2],
      outputs[3],
      outputs[4],
      outputs[5],
      outputs[6],
      outputs[7],
      outputs[8]};
}

} // namespace mlx_ir
