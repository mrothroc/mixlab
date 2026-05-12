#include "mamba3_selective_scan_common.cuh"

extern "C" __global__ void mamba3_selective_scan_fwd_v3_phi_prefix(
    const float* phi_deltas,
    float* phi_offsets,
    int B,
    int T,
    int D,
    int N,
    int G,
    int chunk_size,
    int n_chunks) {
  (void)T;
  (void)G;
  (void)chunk_size;
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
    const int idx = mamba3_theta_idx(chunk_row, d, k, D, K);
    const float delta = phi_deltas[idx];
    phi_offsets[idx] = phi;
    phi += delta;
  }
}
