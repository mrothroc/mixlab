#include "mamba3_selective_scan_common.cuh"

extern "C" __global__ void mamba3_selective_scan_fwd_v2_phi(
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
  const int k = static_cast<int>(threadIdx.x);
  const int K = N / 2;
  if (b >= B || d >= D || k >= K) {
    return;
  }

  float phi = 0.0f;
  for (int chunk = 0; chunk < n_chunks; ++chunk) {
    const int chunk_row = b * n_chunks + chunk;
    phi_offsets[mamba3_theta_idx(chunk_row, d, k, D, K)] = phi;
    const int start = chunk * chunk_size;
    const int limit = (start + chunk_size < T) ? start + chunk_size : T;
    for (int t = start; t < limit; ++t) {
      const int row = b * T + t;
      const int xd = mamba3_channel_idx(row, d, D);
      phi += mamba3_softplus(dt_flat[xd]) * theta_flat[mamba3_theta_idx(row, d, k, D, K)];
    }
  }
}
