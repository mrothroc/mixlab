#include "mamba3_selective_scan_common.cuh"

extern "C" __global__ void mamba3_selective_scan_state(
    const float* x_flat,
    const float* dt_flat,
    const float* lambda_flat,
    const float* theta_flat,
    const float* a_log,
    const float* b_proj_flat,
    const float* c_proj_flat,
    float* h_state,
    int B,
    int T,
    int D,
    int N,
    int G) {
  (void)c_proj_flat;
  const int d = static_cast<int>(blockIdx.x);
  const int b = static_cast<int>(blockIdx.y);
  const int k = static_cast<int>(threadIdx.x);
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

  for (int t = 0; t < T; ++t) {
    const int row = b * T + t;
    const int xd = mamba3_channel_idx(row, d, D);
    const float x = x_flat[xd];
    const float dt = mamba3_softplus(dt_flat[xd]);
    const float lambda = mamba3_sigmoid(lambda_flat[xd]);
    const float theta = theta_flat[mamba3_theta_idx(row, d, k, D, K)];
    phi += dt * theta;

    float b0;
    float b1;
    mamba3_rotate_pair(
        b_proj_flat[mamba3_group_idx(row, g, n0, G, N)],
        b_proj_flat[mamba3_group_idx(row, g, n1, G, N)],
        phi,
        &b0,
        &b1);

    const float A0 = -expf(a_log[d * N + n0]);
    const float A1 = -expf(a_log[d * N + n1]);
    const float alpha0 = expf(dt * A0);
    const float alpha1 = expf(dt * A1);
    const float beta0 = (1.0f - lambda) * dt * alpha0;
    const float beta1 = (1.0f - lambda) * dt * alpha1;
    const float gamma = lambda * dt;

    h0 = alpha0 * h0 + gamma * b0 * x + (t > 0 ? beta0 * prev_b0 * prev_x : 0.0f);
    h1 = alpha1 * h1 + gamma * b1 * x + (t > 0 ? beta1 * prev_b1 * prev_x : 0.0f);
    h_state[mamba3_state_idx(row, d, n0, D, N)] = h0;
    h_state[mamba3_state_idx(row, d, n1, D, N)] = h1;
    prev_b0 = b0;
    prev_b1 = b1;
    prev_x = x;
  }
}
