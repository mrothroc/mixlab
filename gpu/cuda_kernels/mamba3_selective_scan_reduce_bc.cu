#include "mamba3_selective_scan_common.cuh"

extern "C" __global__ void mamba3_selective_scan_reduce_bc(
    const float* grad_b_by_channel,
    const float* grad_c_by_channel,
    float* grad_b,
    float* grad_c,
    int B,
    int T,
    int D,
    int N,
    int G) {
  const int row = static_cast<int>(blockIdx.x);
  const int g = static_cast<int>(blockIdx.y);
  const int n = static_cast<int>(blockIdx.z);
  const int tid = static_cast<int>(threadIdx.x);
  const int channels_per_group = D / G;

  __shared__ float b_partials[128];
  __shared__ float c_partials[128];

  float b_sum = 0.0f;
  float c_sum = 0.0f;
  if (row < B * T && g < G && n < N) {
    for (int local_d = tid; local_d < channels_per_group; local_d += 128) {
      const int d = g * channels_per_group + local_d;
      b_sum += grad_b_by_channel[mamba3_state_idx(row, d, n, D, N)];
      c_sum += grad_c_by_channel[mamba3_state_idx(row, d, n, D, N)];
    }
  }
  b_partials[tid] = b_sum;
  c_partials[tid] = c_sum;
  __syncthreads();

  for (int stride = 64; stride > 0; stride >>= 1) {
    if (tid < stride) {
      b_partials[tid] += b_partials[tid + stride];
      c_partials[tid] += c_partials[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0 && row < B * T && g < G && n < N) {
    grad_b[mamba3_group_idx(row, g, n, G, N)] = b_partials[0];
    grad_c[mamba3_group_idx(row, g, n, G, N)] = c_partials[0];
  }
}
