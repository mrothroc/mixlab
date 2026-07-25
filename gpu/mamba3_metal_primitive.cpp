#include "mamba3_metal_primitive.h"
#include <mlx/allocator.h>
#include <mlx/device.h>
#include <mlx/ops.h>
#include <mlx/primitives.h>
#include <mlx/version.h>
#include <algorithm>
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
constexpr int kMamba3MetalThreads = 32;
constexpr int kMamba3MaxBackwardWindow = 64;
constexpr int kMamba3DefaultBackwardWindow = 32;
bool env_is_one(const char* name) {
  const char* raw = std::getenv(name);
  return raw != nullptr && std::string(raw) == "1";
}
int mamba3_metal_backward_window_size(int T) {
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
          std::min(T, kMamba3MaxBackwardWindow),
          static_cast<int>(parsed)));
}
void validate_mamba3_metal_shape(int B, int T, int D, int N, int G) {
  if (B <= 0 || T <= 0 || D <= 0 || N <= 0 || G <= 0) {
    throw std::runtime_error("Mamba3 Metal primitive requires positive B,T,D,N,G");
  }
  if ((D % G) != 0) {
    throw std::runtime_error("Mamba3 Metal primitive requires D divisible by G");
  }
  if ((N % 2) != 0) {
    throw std::runtime_error("Mamba3 Metal primitive requires even state_size");
  }
  if ((N / 2) > kMamba3MetalThreads) {
    throw std::runtime_error("Mamba3 Metal primitive supports state_size <= 64");
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
void log_mamba3_metal_once() {
  static std::atomic<bool> logged{false};
  if (!logged.exchange(true)) {
    std::cerr << "[mlx_ir] canonical Mamba3 scan using native Metal primitive"
              << " (set MIXLAB_MAMBA3_DISABLE_METAL_PRIMITIVE=1 to use the MLX fallback)"
              << std::endl;
  }
}
#ifdef __APPLE__
const char* kMamba3MetalSource = R"METAL(
#include <metal_stdlib>
using namespace metal;
#define MAMBA3_MAX_BWD_WINDOW 64
inline float mamba3_sigmoid(float x) {
  return 1.0f / (1.0f + exp(-x));
}
inline float mamba3_softplus(float x) {
  if (x > 20.0f) {
    return x;
  }
  if (x < -20.0f) {
    return exp(x);
  }
  return log(1.0f + exp(x));
}
inline int mamba3_state_idx(int row, int d, int n, int D, int N) {
  return (row * D + d) * N + n;
}
inline int mamba3_channel_idx(int row, int d, int D) {
  return row * D + d;
}
inline int mamba3_theta_idx(int row, int d, int k, int D, int K) {
  return row * (D * K) + d * K + k;
}
inline int mamba3_group_idx(int row, int g, int n, int G, int N) {
  return row * (G * N) + g * N + n;
}
inline void mamba3_rotate_pair(
    float even,
    float odd,
    float phi,
    thread float& rot_even,
    thread float& rot_odd) {
  const float c = cos(phi);
  const float s = sin(phi);
  rot_even = c * even + s * odd;
  rot_odd = -s * even + c * odd;
}

inline void mamba3_atomic_add(device float* address, float value) {
  device atomic_uint* atomic_address =
      reinterpret_cast<device atomic_uint*>(address);
  uint expected = atomic_load_explicit(atomic_address, memory_order_relaxed);
  while (true) {
    const uint desired = as_type<uint>(as_type<float>(expected) + value);
    if (atomic_compare_exchange_weak_explicit(
            atomic_address,
            &expected,
            desired,
            memory_order_relaxed,
            memory_order_relaxed)) {
      return;
    }
  }
}

kernel void mamba3_selective_scan_forward_metal(
    const device float* x_flat [[buffer(0)]],
    const device float* dt_flat [[buffer(1)]],
    const device float* lambda_flat [[buffer(2)]],
    const device float* theta_flat [[buffer(3)]],
    const device float* a_log [[buffer(4)]],
    const device float* b_proj_flat [[buffer(5)]],
    const device float* c_proj_flat [[buffer(6)]],
    device float* y_flat [[buffer(7)]],
    constant int& B [[buffer(8)]],
    constant int& T [[buffer(9)]],
    constant int& D [[buffer(10)]],
    constant int& N [[buffer(11)]],
    constant int& G [[buffer(12)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
  const int d = static_cast<int>(group.x);
  const int b = static_cast<int>(group.y);
  const int k = static_cast<int>(tid);
  const int K = N / 2;
  const bool active = b < B && d < D && k < K;
  const int channels_per_group = D / G;
  const int g = d / channels_per_group;
  const int n0 = 2 * k;
  const int n1 = n0 + 1;

  float phi = 0.0f;
  float h0 = 0.0f;
  float h1 = 0.0f;
  float prev_b0 = 0.0f;
  float prev_b1 = 0.0f;
  float prev_x = 0.0f;
  float A0 = 0.0f;
  float A1 = 0.0f;
  if (active) {
    A0 = -exp(a_log[d * N + n0]);
    A1 = -exp(a_log[d * N + n1]);
  }

  for (int t = 0; t < T; ++t) {
    float partial = 0.0f;
    if (active) {
      const int row = b * T + t;
      const int xd = mamba3_channel_idx(row, d, D);
      const float x = x_flat[xd];
      const float dt = mamba3_softplus(dt_flat[xd]);
      const float lambda = mamba3_sigmoid(lambda_flat[xd]);
      phi += dt * theta_flat[mamba3_theta_idx(row, d, k, D, K)];

      float b0;
      float b1;
      float c0;
      float c1;
      mamba3_rotate_pair(
          b_proj_flat[mamba3_group_idx(row, g, n0, G, N)],
          b_proj_flat[mamba3_group_idx(row, g, n1, G, N)],
          phi,
          b0,
          b1);
      mamba3_rotate_pair(
          c_proj_flat[mamba3_group_idx(row, g, n0, G, N)],
          c_proj_flat[mamba3_group_idx(row, g, n1, G, N)],
          phi,
          c0,
          c1);

      const float alpha0 = exp(dt * A0);
      const float alpha1 = exp(dt * A1);
      const float beta0 = (1.0f - lambda) * dt * alpha0;
      const float beta1 = (1.0f - lambda) * dt * alpha1;
      const float gamma = lambda * dt;
      h0 = alpha0 * h0 + gamma * b0 * x +
          (t > 0 ? beta0 * prev_b0 * prev_x : 0.0f);
      h1 = alpha1 * h1 + gamma * b1 * x +
          (t > 0 ? beta1 * prev_b1 * prev_x : 0.0f);
      partial = h0 * c0 + h1 * c1;
      prev_b0 = b0;
      prev_b1 = b1;
      prev_x = x;
    }

    const float total = simd_sum(partial);
    if (tid == 0 && b < B && d < D) {
      y_flat[mamba3_channel_idx(b * T + t, d, D)] = total;
    }
  }
}

kernel void mamba3_zero_backward_metal(
    device float* grad_a_log [[buffer(0)]],
    device float* grad_b [[buffer(1)]],
    device float* grad_c [[buffer(2)]],
    constant int& grad_a_size [[buffer(3)]],
    constant int& grad_b_size [[buffer(4)]],
    constant int& grad_c_size [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid < static_cast<uint>(grad_a_size)) {
    grad_a_log[gid] = 0.0f;
  }
  if (gid < static_cast<uint>(grad_b_size)) {
    grad_b[gid] = 0.0f;
  }
  if (gid < static_cast<uint>(grad_c_size)) {
    grad_c[gid] = 0.0f;
  }
}

kernel void mamba3_backward_checkpoints_metal(
    const device float* x_flat [[buffer(0)]],
    const device float* dt_flat [[buffer(1)]],
    const device float* lambda_flat [[buffer(2)]],
    const device float* theta_flat [[buffer(3)]],
    const device float* a_log [[buffer(4)]],
    const device float* b_proj_flat [[buffer(5)]],
    device float* h_checkpoints [[buffer(8)]],
    device float* phi_checkpoints [[buffer(9)]],
    constant int& B [[buffer(17)]],
    constant int& T [[buffer(18)]],
    constant int& D [[buffer(19)]],
    constant int& N [[buffer(20)]],
    constant int& G [[buffer(21)]],
    constant int& window_size [[buffer(22)]],
    constant int& n_windows [[buffer(23)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
  const int d = static_cast<int>(group.x);
  const int b = static_cast<int>(group.y);
  const int k = static_cast<int>(tid);
  const int K = N / 2;
  if (b >= B || d >= D || k >= K) {
    return;
  }

  const int channels_per_group = D / G;
  const int g = d / channels_per_group;
  const int n0 = 2 * k;
  const int n1 = n0 + 1;
  float phi = 0.0f;
  float h0 = 0.0f;
  float h1 = 0.0f;
  float prev_b0 = 0.0f;
  float prev_b1 = 0.0f;
  float prev_x = 0.0f;
  const float A0 = -exp(a_log[d * N + n0]);
  const float A1 = -exp(a_log[d * N + n1]);

  for (int t = 0; t < T; ++t) {
    if ((t % window_size) == 0) {
      const int w = t / window_size;
      const int h_row = b * n_windows + w;
      const int phi_row = b * (n_windows + 1) + w;
      h_checkpoints[mamba3_state_idx(h_row, d, n0, D, N)] = h0;
      h_checkpoints[mamba3_state_idx(h_row, d, n1, D, N)] = h1;
      phi_checkpoints[mamba3_theta_idx(phi_row, d, k, D, K)] = phi;
    }

    const int row = b * T + t;
    const int xd = mamba3_channel_idx(row, d, D);
    const float x = x_flat[xd];
    const float dt = mamba3_softplus(dt_flat[xd]);
    const float lambda = mamba3_sigmoid(lambda_flat[xd]);
    phi += dt * theta_flat[mamba3_theta_idx(row, d, k, D, K)];

    float b0;
    float b1;
    mamba3_rotate_pair(
        b_proj_flat[mamba3_group_idx(row, g, n0, G, N)],
        b_proj_flat[mamba3_group_idx(row, g, n1, G, N)],
        phi,
        b0,
        b1);

    const float alpha0 = exp(dt * A0);
    const float alpha1 = exp(dt * A1);
    const float beta0 = (1.0f - lambda) * dt * alpha0;
    const float beta1 = (1.0f - lambda) * dt * alpha1;
    const float gamma = lambda * dt;
    h0 = alpha0 * h0 + gamma * b0 * x +
        (t > 0 ? beta0 * prev_b0 * prev_x : 0.0f);
    h1 = alpha1 * h1 + gamma * b1 * x +
        (t > 0 ? beta1 * prev_b1 * prev_x : 0.0f);
    prev_b0 = b0;
    prev_b1 = b1;
    prev_x = x;
  }

  const int final_phi_row = b * (n_windows + 1) + n_windows;
  phi_checkpoints[mamba3_theta_idx(final_phi_row, d, k, D, K)] = phi;
}

kernel void mamba3_selective_scan_backward_metal(
    const device float* x_flat [[buffer(0)]],
    const device float* dt_flat [[buffer(1)]],
    const device float* lambda_flat [[buffer(2)]],
    const device float* theta_flat [[buffer(3)]],
    const device float* a_log [[buffer(4)]],
    const device float* b_proj_flat [[buffer(5)]],
    const device float* c_proj_flat [[buffer(6)]],
    const device float* dy_flat [[buffer(7)]],
    const device float* h_checkpoints [[buffer(8)]],
    const device float* phi_checkpoints [[buffer(9)]],
    device float* grad_x [[buffer(10)]],
    device float* grad_dt [[buffer(11)]],
    device float* grad_lambda [[buffer(12)]],
    device float* grad_theta [[buffer(13)]],
    device float* grad_a_log [[buffer(14)]],
    device float* grad_b [[buffer(15)]],
    device float* grad_c [[buffer(16)]],
    constant int& B [[buffer(17)]],
    constant int& T [[buffer(18)]],
    constant int& D [[buffer(19)]],
    constant int& N [[buffer(20)]],
    constant int& G [[buffer(21)]],
    constant int& window_size [[buffer(22)]],
    constant int& n_windows [[buffer(23)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
  const int d = static_cast<int>(group.x);
  const int b = static_cast<int>(group.y);
  const int k = static_cast<int>(tid);
  const int K = N / 2;
  const bool active = b < B && d < D && k < K;
  const int channels_per_group = D / G;
  const int g = d / channels_per_group;
  const int n0 = 2 * k;
  const int n1 = n0 + 1;

  float A0 = 0.0f;
  float A1 = 0.0f;
  float upstream_next0 = 0.0f;
  float upstream_next1 = 0.0f;
  float alpha_next0 = 0.0f;
  float alpha_next1 = 0.0f;
  float beta_next0 = 0.0f;
  float beta_next1 = 0.0f;
  float phi_carry = 0.0f;
  float grad_a0_total = 0.0f;
  float grad_a1_total = 0.0f;
  float h_before0[MAMBA3_MAX_BWD_WINDOW];
  float h_before1[MAMBA3_MAX_BWD_WINDOW];
  if (active) {
    A0 = -exp(a_log[d * N + n0]);
    A1 = -exp(a_log[d * N + n1]);
  }

  for (int window = n_windows - 1; window >= 0; --window) {
    const int start = window * window_size;
    const int limit = min(start + window_size, T);
    const int end = limit - 1;
    float h_after0 = 0.0f;
    float h_after1 = 0.0f;
    float phi = 0.0f;

    if (active) {
      const int h_row = b * n_windows + window;
      const int phi_row = b * (n_windows + 1) + window;
      h_after0 = h_checkpoints[mamba3_state_idx(h_row, d, n0, D, N)];
      h_after1 = h_checkpoints[mamba3_state_idx(h_row, d, n1, D, N)];
      phi = phi_checkpoints[mamba3_theta_idx(phi_row, d, k, D, K)];
      float prev_b0 = 0.0f;
      float prev_b1 = 0.0f;
      float prev_x = 0.0f;
      if (start > 0) {
        const int prev_row = b * T + start - 1;
        mamba3_rotate_pair(
            b_proj_flat[mamba3_group_idx(prev_row, g, n0, G, N)],
            b_proj_flat[mamba3_group_idx(prev_row, g, n1, G, N)],
            phi,
            prev_b0,
            prev_b1);
        prev_x = x_flat[mamba3_channel_idx(prev_row, d, D)];
      }

      for (int t = start; t < limit; ++t) {
        const int local_t = t - start;
        const int row = b * T + t;
        const int xd = mamba3_channel_idx(row, d, D);
        const float x = x_flat[xd];
        const float dt = mamba3_softplus(dt_flat[xd]);
        const float lambda = mamba3_sigmoid(lambda_flat[xd]);
        phi += dt * theta_flat[mamba3_theta_idx(row, d, k, D, K)];
        float b0;
        float b1;
        mamba3_rotate_pair(
            b_proj_flat[mamba3_group_idx(row, g, n0, G, N)],
            b_proj_flat[mamba3_group_idx(row, g, n1, G, N)],
            phi,
            b0,
            b1);
        const float alpha0 = exp(dt * A0);
        const float alpha1 = exp(dt * A1);
        const float beta0 = (1.0f - lambda) * dt * alpha0;
        const float beta1 = (1.0f - lambda) * dt * alpha1;
        const float gamma = lambda * dt;
        h_before0[local_t] = h_after0;
        h_before1[local_t] = h_after1;
        h_after0 = alpha0 * h_after0 + gamma * b0 * x +
            (t > 0 ? beta0 * prev_b0 * prev_x : 0.0f);
        h_after1 = alpha1 * h_after1 + gamma * b1 * x +
            (t > 0 ? beta1 * prev_b1 * prev_x : 0.0f);
        prev_b0 = b0;
        prev_b1 = b1;
        prev_x = x;
      }
    }

    for (int t = end; t >= start; --t) {
      const int row = b * T + t;
      const int xd = mamba3_channel_idx(row, d, D);
      float grad_x_pair = 0.0f;
      float grad_dt_pair = 0.0f;
      float grad_lambda_pair = 0.0f;
      float dt_raw = 0.0f;
      float lambda = 0.0f;

      if (active) {
        const int local_t = t - start;
        const float x = x_flat[xd];
        dt_raw = dt_flat[xd];
        const float dt = mamba3_softplus(dt_raw);
        lambda = mamba3_sigmoid(lambda_flat[xd]);
        const float theta = theta_flat[mamba3_theta_idx(row, d, k, D, K)];
        const float phi_prev = phi - dt * theta;
        const float dy = dy_flat[xd];

        float b0;
        float b1;
        float c0;
        float c1;
        mamba3_rotate_pair(
            b_proj_flat[mamba3_group_idx(row, g, n0, G, N)],
            b_proj_flat[mamba3_group_idx(row, g, n1, G, N)],
            phi,
            b0,
            b1);
        mamba3_rotate_pair(
            c_proj_flat[mamba3_group_idx(row, g, n0, G, N)],
            c_proj_flat[mamba3_group_idx(row, g, n1, G, N)],
            phi,
            c0,
            c1);

        const float alpha0 = exp(dt * A0);
        const float alpha1 = exp(dt * A1);
        const float beta0 = (1.0f - lambda) * dt * alpha0;
        const float beta1 = (1.0f - lambda) * dt * alpha1;
        const float gamma = lambda * dt;
        float prev_input0 = 0.0f;
        float prev_input1 = 0.0f;
        if (t > 0) {
          const int prev_row = b * T + t - 1;
          float prev_b0;
          float prev_b1;
          mamba3_rotate_pair(
              b_proj_flat[mamba3_group_idx(prev_row, g, n0, G, N)],
              b_proj_flat[mamba3_group_idx(prev_row, g, n1, G, N)],
              phi_prev,
              prev_b0,
              prev_b1);
          const float prev_x = x_flat[mamba3_channel_idx(prev_row, d, D)];
          prev_input0 = prev_b0 * prev_x;
          prev_input1 = prev_b1 * prev_x;
        }

        const float current_input0 = b0 * x;
        const float current_input1 = b1 * x;
        const float upstream0 = dy * c0 + alpha_next0 * upstream_next0;
        const float upstream1 = dy * c1 + alpha_next1 * upstream_next1;
        const float grad_c0 = dy * h_after0;
        const float grad_c1 = dy * h_after1;
        const float grad_b0 =
            gamma * x * upstream0 + beta_next0 * x * upstream_next0;
        const float grad_b1 =
            gamma * x * upstream1 + beta_next1 * x * upstream_next1;

        grad_x_pair =
            gamma * (b0 * upstream0 + b1 * upstream1) +
            beta_next0 * b0 * upstream_next0 +
            beta_next1 * b1 * upstream_next1;
        grad_dt_pair =
            (A0 * alpha0 * h_before0[local_t] +
             (1.0f - lambda) * (alpha0 + dt * A0 * alpha0) * prev_input0 +
             lambda * current_input0) *
                upstream0 +
            (A1 * alpha1 * h_before1[local_t] +
             (1.0f - lambda) * (alpha1 + dt * A1 * alpha1) * prev_input1 +
             lambda * current_input1) *
                upstream1;
        grad_a0_total +=
            (dt * alpha0 * A0 * h_before0[local_t] +
             (1.0f - lambda) * dt * dt * alpha0 * A0 * prev_input0) *
            upstream0;
        grad_a1_total +=
            (dt * alpha1 * A1 * h_before1[local_t] +
             (1.0f - lambda) * dt * dt * alpha1 * A1 * prev_input1) *
            upstream1;
        grad_lambda_pair =
            (-dt * alpha0 * prev_input0 + dt * current_input0) * upstream0 +
            (-dt * alpha1 * prev_input1 + dt * current_input1) * upstream1;

        const float cphi = cos(phi);
        const float sphi = sin(phi);
        mamba3_atomic_add(
            &grad_b[mamba3_group_idx(row, g, n0, G, N)],
            cphi * grad_b0 - sphi * grad_b1);
        mamba3_atomic_add(
            &grad_b[mamba3_group_idx(row, g, n1, G, N)],
            sphi * grad_b0 + cphi * grad_b1);
        mamba3_atomic_add(
            &grad_c[mamba3_group_idx(row, g, n0, G, N)],
            cphi * grad_c0 - sphi * grad_c1);
        mamba3_atomic_add(
            &grad_c[mamba3_group_idx(row, g, n1, G, N)],
            sphi * grad_c0 + cphi * grad_c1);

        const float grad_phi =
            b1 * grad_b0 - b0 * grad_b1 +
            c1 * grad_c0 - c0 * grad_c1;
        phi_carry += grad_phi;
        grad_theta[mamba3_theta_idx(row, d, k, D, K)] = phi_carry * dt;
        grad_dt_pair += phi_carry * theta;

        upstream_next0 = upstream0;
        upstream_next1 = upstream1;
        alpha_next0 = alpha0;
        alpha_next1 = alpha1;
        beta_next0 = beta0;
        beta_next1 = beta1;
        h_after0 = h_before0[local_t];
        h_after1 = h_before1[local_t];
        phi = phi_prev;
      }

      const float grad_x_total = simd_sum(grad_x_pair);
      const float grad_dt_total = simd_sum(grad_dt_pair);
      const float grad_lambda_total = simd_sum(grad_lambda_pair);
      if (tid == 0 && b < B && d < D) {
        grad_x[xd] = grad_x_total;
        grad_dt[xd] = grad_dt_total * mamba3_sigmoid(dt_raw);
        grad_lambda[xd] =
            grad_lambda_total * lambda * (1.0f - lambda);
      }
    }
  }

  if (active) {
    mamba3_atomic_add(&grad_a_log[d * N + n0], grad_a0_total);
    mamba3_atomic_add(&grad_a_log[d * N + n1], grad_a1_total);
  }
}
)METAL";

class Mamba3SelectiveScanMetalForwardPrimitive : public mx::Primitive {
 public:
  Mamba3SelectiveScanMetalForwardPrimitive(
      mx::Stream stream,
      int B,
      int T,
      int D,
      int N,
      int G)
      : mx::Primitive(stream), B_(B), T_(T), D_(D), N_(N), G_(G) {}

  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("Mamba3SelectiveScanMetalForwardPrimitive is Metal-only");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    if (inputs.size() != 7 || outputs.size() != 1) {
      throw std::runtime_error(
          "Mamba3SelectiveScanMetalForwardPrimitive expects 7 inputs and 1 output");
    }
    require_float32_inputs(inputs, "Mamba3SelectiveScanMetalForwardPrimitive");
    require_float32_outputs(outputs, "Mamba3SelectiveScanMetalForwardPrimitive");
    validate_mamba3_metal_shape(B_, T_, D_, N_, G_);

    auto& device = mx::metal::device(stream().device);
#if MLX_VERSION_NUMERIC >= 31002
    auto& encoder = mx::metal::get_command_encoder(stream());
#else
    auto& encoder = device.get_command_encoder(stream().index);
#endif
    outputs[0].set_data(mx::allocator::malloc(outputs[0].nbytes()));
    auto* library = device.get_library(
        "mixlab_mamba3_metal",
        []() { return std::string(kMamba3MetalSource); });
    auto* kernel = device.get_kernel(
        "mamba3_selective_scan_forward_metal",
        library);
    encoder.set_compute_pipeline_state(kernel);
    for (int i = 0; i < 7; ++i) {
      encoder.set_input_array(inputs[static_cast<size_t>(i)], i);
    }
    encoder.set_output_array(outputs[0], 7);
    encoder.set_bytes(B_, 8);
    encoder.set_bytes(T_, 9);
    encoder.set_bytes(D_, 10);
    encoder.set_bytes(N_, 11);
    encoder.set_bytes(G_, 12);
    encoder.dispatch_threadgroups(
        MTL::Size::Make(
            static_cast<NS::UInteger>(D_),
            static_cast<NS::UInteger>(B_),
            1),
        MTL::Size::Make(kMamba3MetalThreads, 1, 1));
  }

  const char* name() const override {
    return "Mamba3SelectiveScanMetalForwardPrimitive";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    auto* rhs = dynamic_cast<const Mamba3SelectiveScanMetalForwardPrimitive*>(&other);
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

class Mamba3SelectiveScanMetalBackwardPrimitive : public mx::Primitive {
 public:
  Mamba3SelectiveScanMetalBackwardPrimitive(
      mx::Stream stream,
      int B,
      int T,
      int D,
      int N,
      int G)
      : mx::Primitive(stream), B_(B), T_(T), D_(D), N_(N), G_(G) {}

  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("Mamba3SelectiveScanMetalBackwardPrimitive is Metal-only");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    if (inputs.size() != 8 || outputs.size() != 9) {
      throw std::runtime_error(
          "Mamba3SelectiveScanMetalBackwardPrimitive expects 8 inputs and 9 outputs");
    }
    require_float32_inputs(inputs, "Mamba3SelectiveScanMetalBackwardPrimitive");
    require_float32_outputs(outputs, "Mamba3SelectiveScanMetalBackwardPrimitive");
    validate_mamba3_metal_shape(B_, T_, D_, N_, G_);

    const int window_size = mamba3_metal_backward_window_size(T_);
    const int n_windows = (T_ + window_size - 1) / window_size;
    auto& device = mx::metal::device(stream().device);
#if MLX_VERSION_NUMERIC >= 31002
    auto& encoder = mx::metal::get_command_encoder(stream());
#else
    auto& encoder = device.get_command_encoder(stream().index);
#endif
    for (auto& output : outputs) {
      output.set_data(mx::allocator::malloc(output.nbytes()));
    }
    auto* library = device.get_library(
        "mixlab_mamba3_metal",
        []() { return std::string(kMamba3MetalSource); });

    auto* zero_kernel = device.get_kernel(
        "mamba3_zero_backward_metal",
        library);
    encoder.set_compute_pipeline_state(zero_kernel);
    encoder.set_output_array(outputs[6], 0);
    encoder.set_output_array(outputs[7], 1);
    encoder.set_output_array(outputs[8], 2);
    const int grad_a_size = D_ * N_;
    const int grad_b_size = B_ * T_ * G_ * N_;
    const int grad_c_size = grad_b_size;
    encoder.set_bytes(grad_a_size, 3);
    encoder.set_bytes(grad_b_size, 4);
    encoder.set_bytes(grad_c_size, 5);
    const int max_zero_size = std::max({grad_a_size, grad_b_size, grad_c_size});
    encoder.dispatch_threads(
        MTL::Size::Make(static_cast<NS::UInteger>(max_zero_size), 1, 1),
        MTL::Size::Make(256, 1, 1));

    auto bind_backward_arrays = [&]() {
      for (int i = 0; i < 8; ++i) {
        encoder.set_input_array(inputs[static_cast<size_t>(i)], i);
      }
      for (int i = 0; i < 9; ++i) {
        encoder.set_output_array(outputs[static_cast<size_t>(i)], 8 + i);
      }
      encoder.set_bytes(B_, 17);
      encoder.set_bytes(T_, 18);
      encoder.set_bytes(D_, 19);
      encoder.set_bytes(N_, 20);
      encoder.set_bytes(G_, 21);
      encoder.set_bytes(window_size, 22);
      encoder.set_bytes(n_windows, 23);
    };

    auto* checkpoint_kernel = device.get_kernel(
        "mamba3_backward_checkpoints_metal",
        library);
    encoder.set_compute_pipeline_state(checkpoint_kernel);
    bind_backward_arrays();
    encoder.dispatch_threadgroups(
        MTL::Size::Make(
            static_cast<NS::UInteger>(D_),
            static_cast<NS::UInteger>(B_),
            1),
        MTL::Size::Make(kMamba3MetalThreads, 1, 1));

    auto* backward_kernel = device.get_kernel(
        "mamba3_selective_scan_backward_metal",
        library);
    encoder.set_compute_pipeline_state(backward_kernel);
    bind_backward_arrays();
    encoder.dispatch_threadgroups(
        MTL::Size::Make(
            static_cast<NS::UInteger>(D_),
            static_cast<NS::UInteger>(B_),
            1),
        MTL::Size::Make(kMamba3MetalThreads, 1, 1));
  }

  const char* name() const override {
    return "Mamba3SelectiveScanMetalBackwardPrimitive";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    auto* rhs = dynamic_cast<const Mamba3SelectiveScanMetalBackwardPrimitive*>(&other);
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
    const int window_size = mamba3_metal_backward_window_size(T_);
    const int n_windows = (T_ + window_size - 1) / window_size;
    return {
        mx::Shape{
            static_cast<mx::ShapeElem>(B_ * n_windows),
            static_cast<mx::ShapeElem>(D_),
            static_cast<mx::ShapeElem>(N_)},
        mx::Shape{
            static_cast<mx::ShapeElem>(B_ * (n_windows + 1)),
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

#endif

} // namespace

bool mamba3_selective_scan_metal_primitive_available(int state_size) {
  if (env_is_one("MIXLAB_MAMBA3_DISABLE_METAL_PRIMITIVE")) {
    return false;
  }
  if (state_size <= 0 ||
      (state_size % 2) != 0 ||
      (state_size / 2) > kMamba3MetalThreads) {
    return false;
  }
#ifdef __APPLE__
  return mx::metal::is_available();
#else
  return false;
#endif
}

mx::array mamba3_selective_scan_metal_forward(
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
#ifdef __APPLE__
  validate_mamba3_metal_shape(B, T, D, N, G);
  log_mamba3_metal_once();
  auto stream = mx::default_stream(mx::default_device());
  auto primitive = std::make_shared<Mamba3SelectiveScanMetalForwardPrimitive>(
      stream, B, T, D, N, G);
  auto outputs = mx::array::make_arrays(
      {mx::Shape{
          static_cast<mx::ShapeElem>(B * T),
          static_cast<mx::ShapeElem>(D)}},
      {mx::float32},
      primitive,
      contiguous_mamba3_inputs(
          x_flat,
          dt_flat,
          lambda_flat,
          theta_flat,
          a_log,
          b_proj_flat,
          c_proj_flat));
  return outputs[0];
#else
  (void)x_flat;
  (void)dt_flat;
  (void)lambda_flat;
  (void)theta_flat;
  (void)a_log;
  (void)b_proj_flat;
  (void)c_proj_flat;
  (void)B;
  (void)T;
  (void)D;
  (void)N;
  (void)G;
  throw std::runtime_error("Mamba3 Metal primitive is unavailable");
#endif
}

std::vector<mx::array> mamba3_selective_scan_metal_vjp(
    const std::vector<mx::array>& args,
    const std::vector<mx::array>& cotangents,
    int B,
    int T,
    int D,
    int N,
    int G) {
#ifdef __APPLE__
  if (args.size() != 7 || cotangents.size() != 1) {
    throw std::runtime_error(
        "mamba3_selective_scan_metal_vjp expects 7 args and 1 cotangent");
  }
  validate_mamba3_metal_shape(B, T, D, N, G);
  auto inputs = contiguous_mamba3_inputs(
      args[0], args[1], args[2], args[3], args[4], args[5], args[6]);
  inputs.push_back(mx::contiguous(cotangents[0]));
  auto stream = mx::default_stream(mx::default_device());
  auto primitive = std::make_shared<Mamba3SelectiveScanMetalBackwardPrimitive>(
      stream, B, T, D, N, G);
  const int BT = B * T;
  const int K = N / 2;
  const int window_size = mamba3_metal_backward_window_size(T);
  const int n_windows = (T + window_size - 1) / window_size;
  std::vector<mx::Shape> shapes = {
      mx::Shape{
          static_cast<mx::ShapeElem>(B * n_windows),
          static_cast<mx::ShapeElem>(D),
          static_cast<mx::ShapeElem>(N)},
      mx::Shape{
          static_cast<mx::ShapeElem>(B * (n_windows + 1)),
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
#else
  (void)args;
  (void)cotangents;
  (void)B;
  (void)T;
  (void)D;
  (void)N;
  (void)G;
  throw std::runtime_error("Mamba3 Metal primitive is unavailable");
#endif
}

} // namespace mlx_ir
