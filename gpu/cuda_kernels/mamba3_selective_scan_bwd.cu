#include "mamba3_selective_scan_common.cuh"

extern "C" __global__ void mamba3_selective_scan_bwd(
    const float* x_flat,
    const float* dt_flat,
    const float* lambda_flat,
    const float* theta_flat,
    const float* a_log,
    const float* b_proj_flat,
    const float* c_proj_flat,
    const float* dy_flat,
    const float* h_state,
    float* grad_x,
    float* grad_dt,
    float* grad_lambda,
    float* grad_theta,
    float* grad_a_log,
    float* grad_b,
    float* grad_c,
    int B,
    int T,
    int D,
    int N,
    int G) {
  const int d = static_cast<int>(blockIdx.x);
  const int b = static_cast<int>(blockIdx.y);
  const int k = static_cast<int>(threadIdx.x);
  const int K = N / 2;
  const bool active = b < B && d < D && k < K;
  const int channels_per_group = D / G;
  const int g = d / channels_per_group;
  const int n0 = 2 * k;
  const int n1 = n0 + 1;

  __shared__ float grad_x_partials[32];
  __shared__ float grad_dt_partials[32];
  __shared__ float grad_lambda_partials[32];

  float phi = 0.0f;
  if (active) {
    for (int t = 0; t < T; ++t) {
      const int row = b * T + t;
      const int xd = mamba3_channel_idx(row, d, D);
      const float dt = mamba3_softplus(dt_flat[xd]);
      phi += dt * theta_flat[mamba3_theta_idx(row, d, k, D, K)];
    }
  }

  float upstream_next0 = 0.0f;
  float upstream_next1 = 0.0f;
  float alpha_next0 = 0.0f;
  float alpha_next1 = 0.0f;
  float beta_next0 = 0.0f;
  float beta_next1 = 0.0f;
  float phi_carry = 0.0f;

  for (int t = T - 1; t >= 0; --t) {
    const int row = b * T + t;
    const int xd = mamba3_channel_idx(row, d, D);
    float grad_x_pair = 0.0f;
    float grad_dt_pair = 0.0f;
    float grad_lambda_pair = 0.0f;
    float dt_raw = 0.0f;
    float lambda = 0.0f;

    if (active) {
      const float x = x_flat[xd];
      dt_raw = dt_flat[xd];
      const float dt = mamba3_softplus(dt_raw);
      lambda = mamba3_sigmoid(lambda_flat[xd]);
      const float theta = theta_flat[mamba3_theta_idx(row, d, k, D, K)];
      const float phi_prev = phi - dt * theta;
      const float dy = dy_flat[xd];

      float b0;
      float b1;
      float c0;
      float c1;
      mamba3_rotate_pair(
          b_proj_flat[mamba3_group_idx(row, g, n0, G, N)],
          b_proj_flat[mamba3_group_idx(row, g, n1, G, N)],
          phi,
          &b0,
          &b1);
      mamba3_rotate_pair(
          c_proj_flat[mamba3_group_idx(row, g, n0, G, N)],
          c_proj_flat[mamba3_group_idx(row, g, n1, G, N)],
          phi,
          &c0,
          &c1);

      const float A0 = -expf(a_log[d * N + n0]);
      const float A1 = -expf(a_log[d * N + n1]);
      const float alpha0 = expf(dt * A0);
      const float alpha1 = expf(dt * A1);
      const float beta0 = (1.0f - lambda) * dt * alpha0;
      const float beta1 = (1.0f - lambda) * dt * alpha1;
      const float gamma = lambda * dt;

      const float h0 = h_state[mamba3_state_idx(row, d, n0, D, N)];
      const float h1 = h_state[mamba3_state_idx(row, d, n1, D, N)];
      const float h_before0 =
          t > 0 ? h_state[mamba3_state_idx(row - 1, d, n0, D, N)] : 0.0f;
      const float h_before1 =
          t > 0 ? h_state[mamba3_state_idx(row - 1, d, n1, D, N)] : 0.0f;

      const float upstream0 = dy * c0 + alpha_next0 * upstream_next0;
      const float upstream1 = dy * c1 + alpha_next1 * upstream_next1;
      const float grad_c0 = dy * h0;
      const float grad_c1 = dy * h1;
      const float grad_b0 = gamma * x * upstream0 + beta_next0 * x * upstream_next0;
      const float grad_b1 = gamma * x * upstream1 + beta_next1 * x * upstream_next1;

      grad_x_pair =
          gamma * (b0 * upstream0 + b1 * upstream1) +
          beta_next0 * b0 * upstream_next0 +
          beta_next1 * b1 * upstream_next1;

      float prev_input0 = 0.0f;
      float prev_input1 = 0.0f;
      if (t > 0) {
        const int prev_row = row - 1;
        float prev_b0;
        float prev_b1;
        mamba3_rotate_pair(
            b_proj_flat[mamba3_group_idx(prev_row, g, n0, G, N)],
            b_proj_flat[mamba3_group_idx(prev_row, g, n1, G, N)],
            phi_prev,
            &prev_b0,
            &prev_b1);
        const float prev_x = x_flat[mamba3_channel_idx(prev_row, d, D)];
        prev_input0 = prev_b0 * prev_x;
        prev_input1 = prev_b1 * prev_x;
      }
      const float current_input0 = b0 * x;
      const float current_input1 = b1 * x;

      grad_dt_pair +=
          (A0 * alpha0 * h_before0 +
           (1.0f - lambda) * (alpha0 + dt * A0 * alpha0) * prev_input0 +
           lambda * current_input0) *
          upstream0;
      grad_dt_pair +=
          (A1 * alpha1 * h_before1 +
           (1.0f - lambda) * (alpha1 + dt * A1 * alpha1) * prev_input1 +
           lambda * current_input1) *
          upstream1;

      atomicAdd(
          &grad_a_log[d * N + n0],
          (dt * alpha0 * A0 * h_before0 +
           (1.0f - lambda) * dt * dt * alpha0 * A0 * prev_input0) *
              upstream0);
      atomicAdd(
          &grad_a_log[d * N + n1],
          (dt * alpha1 * A1 * h_before1 +
           (1.0f - lambda) * dt * dt * alpha1 * A1 * prev_input1) *
              upstream1);

      grad_lambda_pair +=
          (-dt * alpha0 * prev_input0 + dt * current_input0) * upstream0;
      grad_lambda_pair +=
          (-dt * alpha1 * prev_input1 + dt * current_input1) * upstream1;

      const float cphi = cosf(phi);
      const float sphi = sinf(phi);
      atomicAdd(&grad_b[mamba3_group_idx(row, g, n0, G, N)], cphi * grad_b0 - sphi * grad_b1);
      atomicAdd(&grad_b[mamba3_group_idx(row, g, n1, G, N)], sphi * grad_b0 + cphi * grad_b1);
      atomicAdd(&grad_c[mamba3_group_idx(row, g, n0, G, N)], cphi * grad_c0 - sphi * grad_c1);
      atomicAdd(&grad_c[mamba3_group_idx(row, g, n1, G, N)], sphi * grad_c0 + cphi * grad_c1);

      const float grad_phi = b1 * grad_b0 - b0 * grad_b1 + c1 * grad_c0 - c0 * grad_c1;
      phi_carry += grad_phi;
      grad_theta[mamba3_theta_idx(row, d, k, D, K)] = phi_carry * dt;
      grad_dt_pair += phi_carry * theta;

      upstream_next0 = upstream0;
      upstream_next1 = upstream1;
      alpha_next0 = alpha0;
      alpha_next1 = alpha1;
      beta_next0 = beta0;
      beta_next1 = beta1;
      phi = phi_prev;
    }

    grad_x_partials[k] = grad_x_pair;
    grad_dt_partials[k] = grad_dt_pair;
    grad_lambda_partials[k] = grad_lambda_pair;
    __syncthreads();
    for (int stride = 16; stride > 0; stride >>= 1) {
      if (k < stride) {
        grad_x_partials[k] += grad_x_partials[k + stride];
        grad_dt_partials[k] += grad_dt_partials[k + stride];
        grad_lambda_partials[k] += grad_lambda_partials[k + stride];
      }
      __syncthreads();
    }
    if (k == 0 && b < B && d < D) {
      grad_x[xd] = grad_x_partials[0];
      grad_dt[xd] = grad_dt_partials[0] * mamba3_sigmoid(dt_raw);
      grad_lambda[xd] = grad_lambda_partials[0] * lambda * (1.0f - lambda);
    }
    __syncthreads();
  }
}
