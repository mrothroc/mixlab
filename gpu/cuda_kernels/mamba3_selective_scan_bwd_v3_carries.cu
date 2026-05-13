#include "mamba3_selective_scan_common.cuh"

extern "C" __global__ void mamba3_selective_scan_bwd_v3_carries(
    const float* summaries,
    float* carries,
    int B,
    int T,
    int D,
    int N,
    int G,
    int window_size,
    int n_windows) {
  (void)T;
  (void)G;
  (void)window_size;
  const int d = static_cast<int>(blockIdx.x);
  const int b = static_cast<int>(blockIdx.y);
  const int k = static_cast<int>(threadIdx.x);
  const int K = N / 2;
  if (b >= B || d >= D || k >= K) {
    return;
  }

  float u0 = 0.0f;
  float u1 = 0.0f;
  float phi_carry = 0.0f;

  const int final_row = b * (n_windows + 1) + n_windows;
  carries[mamba3_pair_slot_idx(final_row, d, k, 0, D, K, MAMBA3_BWD_CARRY_SLOTS)] = 0.0f;
  carries[mamba3_pair_slot_idx(final_row, d, k, 1, D, K, MAMBA3_BWD_CARRY_SLOTS)] = 0.0f;
  carries[mamba3_pair_slot_idx(final_row, d, k, 2, D, K, MAMBA3_BWD_CARRY_SLOTS)] = 0.0f;

  for (int window = n_windows - 1; window >= 0; --window) {
    const int summary_row = b * n_windows + window;
    const float p0 =
        summaries[mamba3_pair_slot_idx(summary_row, d, k, 0, D, K, MAMBA3_BWD_SUMMARY_SLOTS)];
    const float p1 =
        summaries[mamba3_pair_slot_idx(summary_row, d, k, 1, D, K, MAMBA3_BWD_SUMMARY_SLOTS)];
    const float q0 =
        summaries[mamba3_pair_slot_idx(summary_row, d, k, 2, D, K, MAMBA3_BWD_SUMMARY_SLOTS)];
    const float q1 =
        summaries[mamba3_pair_slot_idx(summary_row, d, k, 3, D, K, MAMBA3_BWD_SUMMARY_SLOTS)];
    const float phi_const =
        summaries[mamba3_pair_slot_idx(summary_row, d, k, 4, D, K, MAMBA3_BWD_SUMMARY_SLOTS)];
    const float phi_u0 =
        summaries[mamba3_pair_slot_idx(summary_row, d, k, 5, D, K, MAMBA3_BWD_SUMMARY_SLOTS)];
    const float phi_u1 =
        summaries[mamba3_pair_slot_idx(summary_row, d, k, 6, D, K, MAMBA3_BWD_SUMMARY_SLOTS)];

    phi_carry += phi_const + phi_u0 * u0 + phi_u1 * u1;
    u0 = q0 + p0 * u0;
    u1 = q1 + p1 * u1;

    const int carry_row = b * (n_windows + 1) + window;
    carries[mamba3_pair_slot_idx(carry_row, d, k, 0, D, K, MAMBA3_BWD_CARRY_SLOTS)] = u0;
    carries[mamba3_pair_slot_idx(carry_row, d, k, 1, D, K, MAMBA3_BWD_CARRY_SLOTS)] = u1;
    carries[mamba3_pair_slot_idx(carry_row, d, k, 2, D, K, MAMBA3_BWD_CARRY_SLOTS)] = phi_carry;
  }
}
