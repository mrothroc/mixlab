#include "mamba3_selective_scan_common.cuh"

extern "C" __global__ void mamba3_selective_scan_fwd_v2_transform(
    const float* x_flat,
    const float* dt_flat,
    const float* lambda_flat,
    const float* theta_flat,
    const float* a_log,
    const float* b_proj_flat,
    const float* phi_offsets,
    float* chunk_transforms,
    int B,
    int T,
    int D,
    int N,
    int G,
    int chunk_size,
    int n_chunks) {
  const int d = static_cast<int>(blockIdx.x);
  const int b = static_cast<int>(blockIdx.y);
  const int chunk = static_cast<int>(blockIdx.z);
  const int k = static_cast<int>(threadIdx.x);
  const int K = N / 2;
  if (b >= B || d >= D || chunk >= n_chunks || k >= K) {
    return;
  }

  const int channels_per_group = D / G;
  const int g = d / channels_per_group;
  const int n0 = 2 * k;
  const int n1 = n0 + 1;
  const int start = chunk * chunk_size;
  const int limit = (start + chunk_size < T) ? start + chunk_size : T;
  const int chunk_row = b * n_chunks + chunk;
  const float A0 = -mamba3_exp(a_log[d * N + n0]);
  const float A1 = -mamba3_exp(a_log[d * N + n1]);

  float phi = phi_offsets[mamba3_theta_idx(chunk_row, d, k, D, K)];
  float prev_b0 = 0.0f;
  float prev_b1 = 0.0f;
  float prev_x = 0.0f;
  if (start > 0) {
    const int prev_row = b * T + start - 1;
    float sphi;
    float cphi;
    mamba3_sincos(phi, &sphi, &cphi);
    mamba3_rotate_pair_cs(
        b_proj_flat[mamba3_group_idx(prev_row, g, n0, G, N)],
        b_proj_flat[mamba3_group_idx(prev_row, g, n1, G, N)],
        cphi,
        sphi,
        &prev_b0,
        &prev_b1);
    prev_x = x_flat[mamba3_channel_idx(prev_row, d, D)];
  }

  float p0 = 1.0f;
  float p1 = 1.0f;
  float q0 = 0.0f;
  float q1 = 0.0f;
  for (int t = start; t < limit; ++t) {
    const int row = b * T + t;
    const int xd = mamba3_channel_idx(row, d, D);
    const float x = x_flat[xd];
    const float dt = mamba3_softplus(dt_flat[xd]);
    const float lambda = mamba3_sigmoid(lambda_flat[xd]);
    const float theta = theta_flat[mamba3_theta_idx(row, d, k, D, K)];
    phi += dt * theta;

    float sphi;
    float cphi;
    mamba3_sincos(phi, &sphi, &cphi);
    float b0;
    float b1;
    mamba3_rotate_pair_cs(
        b_proj_flat[mamba3_group_idx(row, g, n0, G, N)],
        b_proj_flat[mamba3_group_idx(row, g, n1, G, N)],
        cphi,
        sphi,
        &b0,
        &b1);

    const float alpha0 = mamba3_exp(dt * A0);
    const float alpha1 = mamba3_exp(dt * A1);
    const float beta0 = (1.0f - lambda) * dt * alpha0;
    const float beta1 = (1.0f - lambda) * dt * alpha1;
    const float gamma = lambda * dt;
    const float input0 = gamma * b0 * x + (t > 0 ? beta0 * prev_b0 * prev_x : 0.0f);
    const float input1 = gamma * b1 * x + (t > 0 ? beta1 * prev_b1 * prev_x : 0.0f);

    p0 = alpha0 * p0;
    p1 = alpha1 * p1;
    q0 = alpha0 * q0 + input0;
    q1 = alpha1 * q1 + input1;
    prev_b0 = b0;
    prev_b1 = b1;
    prev_x = x;
  }

  const int transform_N = 2 * N;
  chunk_transforms[mamba3_state_idx(chunk_row, d, n0, D, transform_N)] = p0;
  chunk_transforms[mamba3_state_idx(chunk_row, d, n1, D, transform_N)] = p1;
  chunk_transforms[mamba3_state_idx(chunk_row, d, N + n0, D, transform_N)] = q0;
  chunk_transforms[mamba3_state_idx(chunk_row, d, N + n1, D, transform_N)] = q1;
}
