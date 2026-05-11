#include "mamba3_selective_scan_common.cuh"

extern "C" __global__ void mamba3_selective_scan_fwd_v2_hstarts(
    const float* chunk_transforms,
    float* h_starts,
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

  const int n0 = 2 * k;
  const int n1 = n0 + 1;
  const int transform_N = 2 * N;
  float h0 = 0.0f;
  float h1 = 0.0f;
  for (int chunk = 0; chunk < n_chunks; ++chunk) {
    const int chunk_row = b * n_chunks + chunk;
    h_starts[mamba3_state_idx(chunk_row, d, n0, D, N)] = h0;
    h_starts[mamba3_state_idx(chunk_row, d, n1, D, N)] = h1;
    const float p0 = chunk_transforms[mamba3_state_idx(chunk_row, d, n0, D, transform_N)];
    const float p1 = chunk_transforms[mamba3_state_idx(chunk_row, d, n1, D, transform_N)];
    const float q0 = chunk_transforms[mamba3_state_idx(chunk_row, d, N + n0, D, transform_N)];
    const float q1 = chunk_transforms[mamba3_state_idx(chunk_row, d, N + n1, D, transform_N)];
    h0 = p0 * h0 + q0;
    h1 = p1 * h1 + q1;
  }
}
