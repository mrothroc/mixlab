#include "mamba3_selective_scan_common.cuh"

extern "C" __global__ void mamba3_selective_scan_fwd_v3_phi_chunks(
    const float* dt_flat,
    const float* theta_flat,
    float* phi_offsets,
    int B,
    int T,
    int D,
    int N,
    int G,
    int chunk_size,
    int n_chunks) {
  (void)G;
  const int d = static_cast<int>(blockIdx.x);
  const int b = static_cast<int>(blockIdx.y);
  const int chunk = static_cast<int>(blockIdx.z);
  const int k = static_cast<int>(threadIdx.x);
  const int K = N / 2;
  if (b >= B || d >= D || chunk >= n_chunks || k >= K) {
    return;
  }

  const int start = chunk * chunk_size;
  const int limit = (start + chunk_size < T) ? start + chunk_size : T;
  float delta = 0.0f;
  for (int t = start; t < limit; ++t) {
    const int row = b * T + t;
    const int xd = mamba3_channel_idx(row, d, D);
    delta += mamba3_softplus(dt_flat[xd]) * theta_flat[mamba3_theta_idx(row, d, k, D, K)];
  }

  const int chunk_row = b * n_chunks + chunk;
  phi_offsets[mamba3_theta_idx(chunk_row, d, k, D, K)] = delta;
}
