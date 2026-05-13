#include "mamba3_selective_scan_common.cuh"

extern "C" __global__ void mamba3_selective_scan_bwd_v3_emit(
    const float* x_flat,
    const float* dt_flat,
    const float* lambda_flat,
    const float* theta_flat,
    const float* a_log,
    const float* b_proj_flat,
    const float* c_proj_flat,
    const float* dy_flat,
    const float* h_checkpoints,
    const float* phi_checkpoints,
    const float* carries,
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
    int G,
    int window_size,
    int n_windows) {
  const int d = static_cast<int>(blockIdx.x);
  const int b = static_cast<int>(blockIdx.y);
  const int window = static_cast<int>(blockIdx.z);
  const int k = static_cast<int>(threadIdx.x);
  const int K = N / 2;
  const bool active = b < B && d < D && window < n_windows && k < K;
  const int channels_per_group = D / G;
  const int g = d / channels_per_group;
  const int n0 = 2 * k;
  const int n1 = n0 + 1;
  const int window_start = window * window_size;
  const int window_limit = (window_start + window_size < T) ? window_start + window_size : T;
  const int window_end = window_limit - 1;

  float A0 = 0.0f;
  float A1 = 0.0f;
  float upstream_next0 = 0.0f;
  float upstream_next1 = 0.0f;
  float alpha_next0 = 0.0f;
  float alpha_next1 = 0.0f;
  float beta_next0 = 0.0f;
  float beta_next1 = 0.0f;
  float phi_carry = 0.0f;
  float h_before0_window[MAMBA3_MAX_BWD_WINDOW];
  float h_before1_window[MAMBA3_MAX_BWD_WINDOW];
  float h_after0 = 0.0f;
  float h_after1 = 0.0f;
  float phi = 0.0f;

  if (active) {
    A0 = -mamba3_exp(a_log[d * N + n0]);
    A1 = -mamba3_exp(a_log[d * N + n1]);
    const int next_carry_row = b * (n_windows + 1) + window + 1;
    upstream_next0 =
        carries[mamba3_pair_slot_idx(next_carry_row, d, k, 0, D, K, MAMBA3_BWD_CARRY_SLOTS)];
    upstream_next1 =
        carries[mamba3_pair_slot_idx(next_carry_row, d, k, 1, D, K, MAMBA3_BWD_CARRY_SLOTS)];
    phi_carry =
        carries[mamba3_pair_slot_idx(next_carry_row, d, k, 2, D, K, MAMBA3_BWD_CARRY_SLOTS)];
    if (window_limit < T) {
      const int next_row = b * T + window_limit;
      const int next_xd = mamba3_channel_idx(next_row, d, D);
      const float next_dt = mamba3_softplus(dt_flat[next_xd]);
      const float next_lambda = mamba3_sigmoid(lambda_flat[next_xd]);
      alpha_next0 = mamba3_exp(next_dt * A0);
      alpha_next1 = mamba3_exp(next_dt * A1);
      beta_next0 = (1.0f - next_lambda) * next_dt * alpha_next0;
      beta_next1 = (1.0f - next_lambda) * next_dt * alpha_next1;
    }

    const int h_checkpoint_row = b * n_windows + window;
    const int phi_start_row = b * (n_windows + 1) + window;
    const int phi_end_row = phi_start_row + 1;
    h_after0 = h_checkpoints[mamba3_state_idx(h_checkpoint_row, d, n0, D, N)];
    h_after1 = h_checkpoints[mamba3_state_idx(h_checkpoint_row, d, n1, D, N)];
    phi = phi_checkpoints[mamba3_theta_idx(phi_end_row, d, k, D, K)];
    float replay_phi = phi_checkpoints[mamba3_theta_idx(phi_start_row, d, k, D, K)];
    float replay_prev_b0 = 0.0f;
    float replay_prev_b1 = 0.0f;
    float replay_prev_x = 0.0f;
    if (window_start > 0) {
      const int prev_row = b * T + window_start - 1;
      float s_replay_phi;
      float c_replay_phi;
      mamba3_sincos(replay_phi, &s_replay_phi, &c_replay_phi);
      mamba3_rotate_pair_cs(
          b_proj_flat[mamba3_group_idx(prev_row, g, n0, G, N)],
          b_proj_flat[mamba3_group_idx(prev_row, g, n1, G, N)],
          c_replay_phi,
          s_replay_phi,
          &replay_prev_b0,
          &replay_prev_b1);
      replay_prev_x = x_flat[mamba3_channel_idx(prev_row, d, D)];
    }
    for (int replay_t = window_start; replay_t <= window_end; ++replay_t) {
      const int local_t = replay_t - window_start;
      const int replay_row = b * T + replay_t;
      const int replay_xd = mamba3_channel_idx(replay_row, d, D);
      const float replay_x = x_flat[replay_xd];
      const float replay_dt = mamba3_softplus(dt_flat[replay_xd]);
      const float replay_lambda = mamba3_sigmoid(lambda_flat[replay_xd]);
      replay_phi += replay_dt * theta_flat[mamba3_theta_idx(replay_row, d, k, D, K)];
      float s_replay_phi;
      float c_replay_phi;
      mamba3_sincos(replay_phi, &s_replay_phi, &c_replay_phi);
      float replay_b0;
      float replay_b1;
      mamba3_rotate_pair_cs(
          b_proj_flat[mamba3_group_idx(replay_row, g, n0, G, N)],
          b_proj_flat[mamba3_group_idx(replay_row, g, n1, G, N)],
          c_replay_phi,
          s_replay_phi,
          &replay_b0,
          &replay_b1);
      const float replay_alpha0 = mamba3_exp(replay_dt * A0);
      const float replay_alpha1 = mamba3_exp(replay_dt * A1);
      const float replay_beta0 = (1.0f - replay_lambda) * replay_dt * replay_alpha0;
      const float replay_beta1 = (1.0f - replay_lambda) * replay_dt * replay_alpha1;
      const float replay_gamma = replay_lambda * replay_dt;
      h_before0_window[local_t] = h_after0;
      h_before1_window[local_t] = h_after1;
      h_after0 = replay_alpha0 * h_after0 + replay_gamma * replay_b0 * replay_x +
          (replay_t > 0 ? replay_beta0 * replay_prev_b0 * replay_prev_x : 0.0f);
      h_after1 = replay_alpha1 * h_after1 + replay_gamma * replay_b1 * replay_x +
          (replay_t > 0 ? replay_beta1 * replay_prev_b1 * replay_prev_x : 0.0f);
      replay_prev_b0 = replay_b0;
      replay_prev_b1 = replay_b1;
      replay_prev_x = replay_x;
    }
  }

  for (int t = window_end; t >= window_start; --t) {
    const int row = b * T + t;
    const int xd = mamba3_channel_idx(row, d, D);
    float grad_x_pair = 0.0f;
    float grad_dt_pair = 0.0f;
    float grad_lambda_pair = 0.0f;
    float dt_raw = 0.0f;
    float lambda = 0.0f;

    if (active) {
      const int local_t = t - window_start;
      const float x = x_flat[xd];
      dt_raw = dt_flat[xd];
      const float dt = mamba3_softplus(dt_raw);
      lambda = mamba3_sigmoid(lambda_flat[xd]);
      const float theta = theta_flat[mamba3_theta_idx(row, d, k, D, K)];
      const float phi_prev = phi - dt * theta;
      const float dy = dy_flat[xd];

      float sphi;
      float cphi;
      mamba3_sincos(phi, &sphi, &cphi);
      float b0;
      float b1;
      float c0;
      float c1;
      mamba3_rotate_pair_cs(
          b_proj_flat[mamba3_group_idx(row, g, n0, G, N)],
          b_proj_flat[mamba3_group_idx(row, g, n1, G, N)],
          cphi,
          sphi,
          &b0,
          &b1);
      mamba3_rotate_pair_cs(
          c_proj_flat[mamba3_group_idx(row, g, n0, G, N)],
          c_proj_flat[mamba3_group_idx(row, g, n1, G, N)],
          cphi,
          sphi,
          &c0,
          &c1);

      const float alpha0 = mamba3_exp(dt * A0);
      const float alpha1 = mamba3_exp(dt * A1);
      const float beta0 = (1.0f - lambda) * dt * alpha0;
      const float beta1 = (1.0f - lambda) * dt * alpha1;
      const float gamma = lambda * dt;

      float prev_input0 = 0.0f;
      float prev_input1 = 0.0f;
      if (t > 0) {
        const int prev_row = b * T + t - 1;
        float sphi_prev;
        float cphi_prev;
        mamba3_sincos(phi_prev, &sphi_prev, &cphi_prev);
        float prev_b0;
        float prev_b1;
        mamba3_rotate_pair_cs(
            b_proj_flat[mamba3_group_idx(prev_row, g, n0, G, N)],
            b_proj_flat[mamba3_group_idx(prev_row, g, n1, G, N)],
            cphi_prev,
            sphi_prev,
            &prev_b0,
            &prev_b1);
        const float prev_x = x_flat[mamba3_channel_idx(prev_row, d, D)];
        prev_input0 = prev_b0 * prev_x;
        prev_input1 = prev_b1 * prev_x;
      }
      const float current_input0 = b0 * x;
      const float current_input1 = b1 * x;
      const float h_before0 = h_before0_window[local_t];
      const float h_before1 = h_before1_window[local_t];

      const float upstream0 = dy * c0 + alpha_next0 * upstream_next0;
      const float upstream1 = dy * c1 + alpha_next1 * upstream_next1;
      const float grad_c0 = dy * h_after0;
      const float grad_c1 = dy * h_after1;
      const float grad_b0 = gamma * x * upstream0 + beta_next0 * x * upstream_next0;
      const float grad_b1 = gamma * x * upstream1 + beta_next1 * x * upstream_next1;

      grad_x_pair =
          gamma * (b0 * upstream0 + b1 * upstream1) +
          beta_next0 * b0 * upstream_next0 +
          beta_next1 * b1 * upstream_next1;

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
      h_after0 = h_before0;
      h_after1 = h_before1;
      phi = phi_prev;
    }

    grad_x_pair = mamba3_warp_sum(grad_x_pair);
    grad_dt_pair = mamba3_warp_sum(grad_dt_pair);
    grad_lambda_pair = mamba3_warp_sum(grad_lambda_pair);
    if (k == 0 && b < B && d < D && window < n_windows) {
      grad_x[xd] = grad_x_pair;
      grad_dt[xd] = grad_dt_pair * mamba3_sigmoid(dt_raw);
      grad_lambda[xd] = grad_lambda_pair * lambda * (1.0f - lambda);
    }
  }
}
