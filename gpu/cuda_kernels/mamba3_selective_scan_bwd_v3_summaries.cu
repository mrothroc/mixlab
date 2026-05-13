#include "mamba3_selective_scan_common.cuh"

extern "C" __global__ void mamba3_selective_scan_bwd_v3_summaries(
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
    float* summaries,
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

  float h_before0_window[MAMBA3_MAX_BWD_WINDOW];
  float h_before1_window[MAMBA3_MAX_BWD_WINDOW];
  float h_after0 = 0.0f;
  float h_after1 = 0.0f;
  float phi = 0.0f;
  float A0 = 0.0f;
  float A1 = 0.0f;
  if (active) {
    A0 = -mamba3_exp(a_log[d * N + n0]);
    A1 = -mamba3_exp(a_log[d * N + n1]);
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

  float alpha_next0 = 0.0f;
  float alpha_next1 = 0.0f;
  float beta_next0 = 0.0f;
  float beta_next1 = 0.0f;
  if (active && window_limit < T) {
    const int next_row = b * T + window_limit;
    const int next_xd = mamba3_channel_idx(next_row, d, D);
    const float next_dt = mamba3_softplus(dt_flat[next_xd]);
    const float next_lambda = mamba3_sigmoid(lambda_flat[next_xd]);
    alpha_next0 = mamba3_exp(next_dt * A0);
    alpha_next1 = mamba3_exp(next_dt * A1);
    beta_next0 = (1.0f - next_lambda) * next_dt * alpha_next0;
    beta_next1 = (1.0f - next_lambda) * next_dt * alpha_next1;
  }

  float q_next0 = 0.0f;
  float q_next1 = 0.0f;
  float p_next0 = 1.0f;
  float p_next1 = 1.0f;
  float phi_const = 0.0f;
  float phi_u0 = 0.0f;
  float phi_u1 = 0.0f;

  for (int t = window_end; t >= window_start; --t) {
    if (active) {
      const int local_t = t - window_start;
      const int row = b * T + t;
      const int xd = mamba3_channel_idx(row, d, D);
      const float x = x_flat[xd];
      const float dt = mamba3_softplus(dt_flat[xd]);
      const float lambda = mamba3_sigmoid(lambda_flat[xd]);
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

      const float u0_const = dy * c0 + alpha_next0 * q_next0;
      const float u1_const = dy * c1 + alpha_next1 * q_next1;
      const float u0_coeff = alpha_next0 * p_next0;
      const float u1_coeff = alpha_next1 * p_next1;

      const float grad_c0 = dy * h_after0;
      const float grad_c1 = dy * h_after1;
      const float grad_b0_const = gamma * x * u0_const + beta_next0 * x * q_next0;
      const float grad_b1_const = gamma * x * u1_const + beta_next1 * x * q_next1;
      const float grad_b0_coeff = gamma * x * u0_coeff + beta_next0 * x * p_next0;
      const float grad_b1_coeff = gamma * x * u1_coeff + beta_next1 * x * p_next1;

      phi_const += b1 * grad_b0_const - b0 * grad_b1_const + c1 * grad_c0 - c0 * grad_c1;
      phi_u0 += b1 * grad_b0_coeff;
      phi_u1 += -b0 * grad_b1_coeff;

      q_next0 = u0_const;
      q_next1 = u1_const;
      p_next0 = u0_coeff;
      p_next1 = u1_coeff;
      alpha_next0 = alpha0;
      alpha_next1 = alpha1;
      beta_next0 = beta0;
      beta_next1 = beta1;
      h_after0 = h_before0_window[local_t];
      h_after1 = h_before1_window[local_t];
      phi = phi_prev;
    }
  }

  if (active) {
    const int summary_row = b * n_windows + window;
    summaries[mamba3_pair_slot_idx(summary_row, d, k, 0, D, K, MAMBA3_BWD_SUMMARY_SLOTS)] = p_next0;
    summaries[mamba3_pair_slot_idx(summary_row, d, k, 1, D, K, MAMBA3_BWD_SUMMARY_SLOTS)] = p_next1;
    summaries[mamba3_pair_slot_idx(summary_row, d, k, 2, D, K, MAMBA3_BWD_SUMMARY_SLOTS)] = q_next0;
    summaries[mamba3_pair_slot_idx(summary_row, d, k, 3, D, K, MAMBA3_BWD_SUMMARY_SLOTS)] = q_next1;
    summaries[mamba3_pair_slot_idx(summary_row, d, k, 4, D, K, MAMBA3_BWD_SUMMARY_SLOTS)] = phi_const;
    summaries[mamba3_pair_slot_idx(summary_row, d, k, 5, D, K, MAMBA3_BWD_SUMMARY_SLOTS)] = phi_u0;
    summaries[mamba3_pair_slot_idx(summary_row, d, k, 6, D, K, MAMBA3_BWD_SUMMARY_SLOTS)] = phi_u1;
  }
}
