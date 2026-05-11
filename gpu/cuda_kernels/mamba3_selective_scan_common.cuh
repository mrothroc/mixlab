#pragma once

#define MAMBA3_MAX_BWD_WINDOW 64

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

__device__ inline void mamba3_sincos(float x, float* s, float* c) {
  __sincosf(x, s, c);
}

__device__ inline void mamba3_rotate_pair_cs(
    float even,
    float odd,
    float c,
    float s,
    float* rot_even,
    float* rot_odd) {
  *rot_even = c * even + s * odd;
  *rot_odd = -s * even + c * odd;
}

__device__ inline void mamba3_rotate_pair(
    float even,
    float odd,
    float phi,
    float* rot_even,
    float* rot_odd) {
  float s;
  float c;
  mamba3_sincos(phi, &s, &c);
  mamba3_rotate_pair_cs(even, odd, c, s, rot_even, rot_odd);
}

__device__ inline float mamba3_warp_sum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
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
