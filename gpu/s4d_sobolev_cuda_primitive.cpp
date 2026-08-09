#include "s4d_sobolev_cuda_primitive.h"
#include "cuda_kernel_dispatch.h"

#include <mlx/device.h>
#include <mlx/ops.h>
#include <mlx/primitives.h>

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

constexpr int kS4DSobolevCUDAThreads = 256;

bool env_is_one(const char* name) {
  const char* raw = std::getenv(name);
  return raw != nullptr && std::string(raw) == "1";
}

void validate_s4d_sobolev_inputs(
    const mx::array& product,
    const mx::array& beta,
    int fft_len) {
  if (fft_len <= 0) {
    throw std::runtime_error("S4D Sobolev CUDA primitive requires fft_len > 0");
  }
  if (product.dtype() != mx::complex64 || product.ndim() != 3) {
    throw std::runtime_error(
        "S4D Sobolev CUDA primitive requires complex64 product [B,bins,D]");
  }
  if (beta.dtype() != mx::float32 || beta.ndim() != 1) {
    throw std::runtime_error(
        "S4D Sobolev CUDA primitive requires float32 beta [D]");
  }
  if (product.shape(0) <= 0 || product.shape(2) <= 0) {
    throw std::runtime_error("S4D Sobolev CUDA primitive requires positive B and D");
  }
  if (product.shape(1) != fft_len / 2 + 1) {
    throw std::runtime_error(
        "S4D Sobolev CUDA primitive product bins do not match fft_len");
  }
  if (product.shape(2) != beta.shape(0)) {
    throw std::runtime_error(
        "S4D Sobolev CUDA primitive product width does not match beta");
  }
}

void log_s4d_sobolev_cuda_once() {
  static std::atomic<bool> logged{false};
  if (!logged.exchange(true)) {
    std::cerr << "[mlx_ir] S4D Sobolev filter using CUDA forward/backward primitive"
              << " (set MIXLAB_S4D_SOBOLEV_DISABLE_CUDA_PRIMITIVE=1 only for forward differential checks)"
              << std::endl;
  }
}

class S4DSobolevBackwardCUDAPrimitive : public mx::Primitive {
 public:
  S4DSobolevBackwardCUDAPrimitive(
      mx::Stream stream,
      int batch,
      int bins,
      int width,
      int fft_len)
      : mx::Primitive(stream),
        batch_(batch),
        bins_(bins),
        width_(width),
        fft_len_(fft_len) {}

  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("S4DSobolevBackwardCUDAPrimitive is CUDA-only");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    if (inputs.size() != 3 || outputs.size() != 2) {
      throw std::runtime_error(
          "S4DSobolevBackwardCUDAPrimitive expects 3 inputs and 2 outputs");
    }
    validate_s4d_sobolev_inputs(inputs[0], inputs[1], fft_len_);
    if (inputs[0].shape(0) != batch_ || inputs[0].shape(1) != bins_ ||
        inputs[0].shape(2) != width_) {
      throw std::runtime_error(
          "S4DSobolevBackwardCUDAPrimitive input shape changed after construction");
    }
    if (inputs[2].dtype() != mx::complex64 || inputs[2].shape() != inputs[0].shape()) {
      throw std::runtime_error(
          "S4DSobolevBackwardCUDAPrimitive cotangent must match product");
    }
    if (outputs[0].dtype() != mx::complex64 || outputs[1].dtype() != mx::float32) {
      throw std::runtime_error(
          "S4DSobolevBackwardCUDAPrimitive has invalid output dtypes");
    }

    const int elements = batch_ * bins_ * width_;
    const int element_blocks = (elements + kS4DSobolevCUDAThreads - 1) /
        kS4DSobolevCUDAThreads;
    launch_precompiled_cuda_kernel_into(
        "s4d_sobolev_filter_vjp_product",
        {inputs[2], inputs[1]},
        {&outputs[0]},
        {batch_, bins_, width_, fft_len_},
        std::make_tuple(element_blocks, 1, 1),
        std::make_tuple(kS4DSobolevCUDAThreads, 1, 1),
        stream());
    launch_precompiled_cuda_kernel_into(
        "s4d_sobolev_filter_vjp_beta",
        {inputs[0], inputs[2], inputs[1]},
        {&outputs[1]},
        {batch_, bins_, width_, fft_len_},
        std::make_tuple(width_, 1, 1),
        std::make_tuple(kS4DSobolevCUDAThreads, 1, 1),
        stream(),
        kS4DSobolevCUDAThreads * static_cast<int>(sizeof(double)));
  }

  const char* name() const override {
    return "S4DSobolevBackwardCUDAPrimitive";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    auto* rhs = dynamic_cast<const S4DSobolevBackwardCUDAPrimitive*>(&other);
    return rhs != nullptr && batch_ == rhs->batch_ && bins_ == rhs->bins_ &&
        width_ == rhs->width_ && fft_len_ == rhs->fft_len_;
  }

  std::vector<mx::Shape> output_shapes(const std::vector<mx::array>&) override {
    return {
        mx::Shape{
            static_cast<mx::ShapeElem>(batch_),
            static_cast<mx::ShapeElem>(bins_),
            static_cast<mx::ShapeElem>(width_)},
        mx::Shape{static_cast<mx::ShapeElem>(width_)}};
  }

 private:
  int batch_;
  int bins_;
  int width_;
  int fft_len_;
};

std::vector<mx::array> s4d_sobolev_cuda_vjp(
    const mx::array& product,
    const mx::array& beta,
    const mx::array& cotangent,
    int fft_len) {
  validate_s4d_sobolev_inputs(product, beta, fft_len);
  auto stream = mx::default_stream(mx::default_device());
  auto primitive = std::make_shared<S4DSobolevBackwardCUDAPrimitive>(
      stream,
      product.shape(0),
      product.shape(1),
      product.shape(2),
      fft_len);
  std::vector<mx::Shape> shapes{product.shape(), beta.shape()};
  std::vector<mx::Dtype> dtypes{mx::complex64, mx::float32};
  return mx::array::make_arrays(
      shapes,
      dtypes,
      primitive,
      {mx::contiguous(product), mx::contiguous(beta), mx::contiguous(cotangent)});
}

class S4DSobolevForwardCUDAPrimitive : public mx::Primitive {
 public:
  S4DSobolevForwardCUDAPrimitive(
      mx::Stream stream,
      int batch,
      int bins,
      int width,
      int fft_len)
      : mx::Primitive(stream),
        batch_(batch),
        bins_(bins),
        width_(width),
        fft_len_(fft_len) {}

  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("S4DSobolevForwardCUDAPrimitive is CUDA-only");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    if (inputs.size() != 2 || outputs.size() != 1) {
      throw std::runtime_error(
          "S4DSobolevForwardCUDAPrimitive expects 2 inputs and 1 output");
    }
    validate_s4d_sobolev_inputs(inputs[0], inputs[1], fft_len_);
    if (inputs[0].shape(0) != batch_ || inputs[0].shape(1) != bins_ ||
        inputs[0].shape(2) != width_) {
      throw std::runtime_error(
          "S4DSobolevForwardCUDAPrimitive input shape changed after construction");
    }
    if (outputs[0].dtype() != mx::complex64) {
      throw std::runtime_error(
          "S4DSobolevForwardCUDAPrimitive requires complex64 output");
    }
    const int elements = batch_ * bins_ * width_;
    const int blocks = (elements + kS4DSobolevCUDAThreads - 1) /
        kS4DSobolevCUDAThreads;
    launch_precompiled_cuda_kernel_into(
        "s4d_sobolev_filter_forward",
        inputs,
        {&outputs[0]},
        {batch_, bins_, width_, fft_len_},
        std::make_tuple(blocks, 1, 1),
        std::make_tuple(kS4DSobolevCUDAThreads, 1, 1),
        stream());
  }

  std::vector<mx::array> vjp(
      const std::vector<mx::array>& primals,
      const std::vector<mx::array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<mx::array>&) override {
    if (primals.size() != 2 || cotangents.size() != 1) {
      throw std::runtime_error(
          "S4DSobolevForwardCUDAPrimitive vjp expects 2 primals and 1 cotangent");
    }
    auto all_grads = s4d_sobolev_cuda_vjp(
        primals[0], primals[1], cotangents[0], fft_len_);
    std::vector<mx::array> grads;
    grads.reserve(argnums.size());
    for (int argnum : argnums) {
      if (argnum < 0 || argnum > 1) {
        throw std::runtime_error(
            "S4DSobolevForwardCUDAPrimitive vjp argnum out of range");
      }
      grads.push_back(all_grads[argnum]);
    }
    return grads;
  }

  const char* name() const override {
    return "S4DSobolevForwardCUDAPrimitive";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    auto* rhs = dynamic_cast<const S4DSobolevForwardCUDAPrimitive*>(&other);
    return rhs != nullptr && batch_ == rhs->batch_ && bins_ == rhs->bins_ &&
        width_ == rhs->width_ && fft_len_ == rhs->fft_len_;
  }

  std::vector<mx::Shape> output_shapes(const std::vector<mx::array>&) override {
    return {mx::Shape{
        static_cast<mx::ShapeElem>(batch_),
        static_cast<mx::ShapeElem>(bins_),
        static_cast<mx::ShapeElem>(width_)}};
  }

 private:
  int batch_;
  int bins_;
  int width_;
  int fft_len_;
};

} // namespace

bool s4d_sobolev_cuda_primitive_available() {
  if (env_is_one("MIXLAB_S4D_SOBOLEV_DISABLE_CUDA_PRIMITIVE")) {
    return false;
  }
#ifdef __linux__
  return mx::is_available(mx::Device::gpu) &&
      precompiled_cuda_kernel_available("s4d_sobolev_filter_forward") &&
      precompiled_cuda_kernel_available("s4d_sobolev_filter_vjp_product") &&
      precompiled_cuda_kernel_available("s4d_sobolev_filter_vjp_beta");
#else
  return false;
#endif
}

mx::array s4d_sobolev_filter_cuda_primitive(
    const mx::array& product,
    const mx::array& beta,
    int fft_len) {
  validate_s4d_sobolev_inputs(product, beta, fft_len);
  if (!s4d_sobolev_cuda_primitive_available()) {
    throw std::runtime_error("S4D Sobolev CUDA primitive is unavailable");
  }
  log_s4d_sobolev_cuda_once();
  auto stream = mx::default_stream(mx::default_device());
  auto primitive = std::make_shared<S4DSobolevForwardCUDAPrimitive>(
      stream,
      product.shape(0),
      product.shape(1),
      product.shape(2),
      fft_len);
  return mx::array(
      product.shape(),
      mx::complex64,
      primitive,
      std::vector<mx::array>{mx::contiguous(product), mx::contiguous(beta)});
}

} // namespace mlx_ir
