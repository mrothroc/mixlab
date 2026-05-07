#pragma once

namespace {

__device__ inline float mamba3_sigmoid(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

__device__ inline float mamba3_softplus(float x) {
  if (x > 20.0f) {
    return x;
  }
  if (x < -20.0f) {
    return __expf(x);
  }
  return __logf(1.0f + __expf(x));
}

__device__ inline float mamba3_exp(float x) {
  return __expf(x);
}

__device__ inline float mamba3_cos(float x) {
  return __cosf(x);
}

__device__ inline float mamba3_sin(float x) {
  return __sinf(x);
}

__device__ inline void mamba3_rotate_pair(
    float even,
    float odd,
    float phi,
    float* rot_even,
    float* rot_odd) {
  const float c = mamba3_cos(phi);
  const float s = mamba3_sin(phi);
  *rot_even = c * even + s * odd;
  *rot_odd = -s * even + c * odd;
}

__device__ inline int mamba3_state_idx(int row, int d, int n, int D, int N) {
  return (row * D + d) * N + n;
}

__device__ inline int mamba3_channel_idx(int row, int d, int D) {
  return row * D + d;
}

__device__ inline int mamba3_theta_idx(int row, int d, int k, int D, int K) {
  return row * (D * K) + d * K + k;
}

__device__ inline int mamba3_group_idx(int row, int g, int n, int G, int N) {
  return row * (G * N) + g * N + n;
}

} // namespace
