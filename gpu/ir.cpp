#include "ir.h"
#include "gated_delta_cuda_primitive.h"
#include "gated_delta_metal_primitive.h"
#include "mamba3_cuda_primitive.h"

#include <mlx/device.h>
#include <mlx/random.h>
#include <mlx/transforms.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mx = mlx::core;

namespace {

constexpr float kGatedDeltaGateFloor = 1e-30f;
constexpr float kGatedDeltaExpClampMin = -80.0f;
constexpr float kGatedDeltaExpClampMax = 0.0f;
// Safety bound: even when fusing, evaluate every K chunks to cap lazy-graph depth on user-overridden small chunk_size with long T.
constexpr int EVAL_EVERY_K_CHUNKS = 16;

using GatedDeltaTimingClock = std::chrono::steady_clock;

struct GatedDeltaTimingTotals {
  long long calls = 0;
  long long chunks = 0;
  double raw_attn_ms = 0.0;
  double solve_ms = 0.0;
  double post_solve_ms = 0.0;
  double causal_attn_ms = 0.0;
  double chunk_loop_ms = 0.0;
};

std::mutex g_gated_delta_timing_mu;
GatedDeltaTimingTotals g_gated_delta_timing;

bool gated_delta_timing_enabled() {
  const char* raw = std::getenv("MIXLAB_GATED_DELTA_TIMING");
  return raw != nullptr && raw[0] != '\0' && raw[0] != '0';
}

double elapsed_ms(
    GatedDeltaTimingClock::time_point start,
    GatedDeltaTimingClock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

void record_gated_delta_timing(
    int n_chunks,
    double raw_attn_ms,
    double solve_ms,
    double post_solve_ms,
    double causal_attn_ms,
    double chunk_loop_ms) {
  if (!gated_delta_timing_enabled()) {
    return;
  }
  std::lock_guard<std::mutex> lock(g_gated_delta_timing_mu);
  g_gated_delta_timing.calls++;
  g_gated_delta_timing.chunks += n_chunks;
  g_gated_delta_timing.raw_attn_ms += raw_attn_ms;
  g_gated_delta_timing.solve_ms += solve_ms;
  g_gated_delta_timing.post_solve_ms += post_solve_ms;
  g_gated_delta_timing.causal_attn_ms += causal_attn_ms;
  g_gated_delta_timing.chunk_loop_ms += chunk_loop_ms;
}

mx::Shape make_shape(const int* vals, int n) {
  mx::Shape s;
  s.reserve(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    s.push_back(static_cast<mx::ShapeElem>(vals[i]));
  }
  return s;
}

mx::array cross_entropy_mean(const mx::array& logits, const mx::array& targets) {
  auto row_max = mx::max(logits, 1, true);
  auto shifted = logits - row_max;
  auto log_norm = mx::log(mx::sum(mx::exp(shifted), 1, true));
  auto log_probs = shifted - log_norm;
  auto idx = mx::reshape(targets, {targets.shape(0), 1});
  auto chosen = mx::take_along_axis(log_probs, idx, 1);
  return -mx::mean(chosen);
}

mx::array cross_entropy_per_token(const mx::array& logits, const mx::array& targets) {
  auto row_max = mx::max(logits, 1, true);
  auto shifted = logits - row_max;
  auto log_norm = mx::log(mx::sum(mx::exp(shifted), 1, true));
  auto log_probs = shifted - log_norm;
  auto idx = mx::reshape(targets, {targets.shape(0), 1});
  auto chosen = mx::take_along_axis(log_probs, idx, 1);
  return -mx::reshape(chosen, {targets.shape(0)});
}

mx::array masked_cross_entropy_mean(
    const mx::array& logits,
    const mx::array& targets,
    const mx::array& loss_mask) {
  if (loss_mask.ndim() != 1 || loss_mask.shape(0) != targets.shape(0)) {
    throw std::runtime_error("loss_mask must be a rank-1 vector matching targets");
  }
  auto nll = cross_entropy_per_token(logits, targets);
  auto mask = mx::astype(
      mx::greater(mx::astype(loss_mask, mx::float32), mx::array(0.0f, mx::float32)),
      mx::float32);
  auto denom = mx::maximum(mx::sum(mask), mx::array(1.0f, mx::float32));
  return mx::sum(nll * mask) / denom;
}

mx::array masked_cross_entropy_per_token(
    const mx::array& logits,
    const mx::array& targets,
    const mx::array& loss_mask) {
  if (loss_mask.ndim() != 1 || loss_mask.shape(0) != targets.shape(0)) {
    throw std::runtime_error("loss_mask must be a rank-1 vector matching targets");
  }
  auto nll = cross_entropy_per_token(logits, targets);
  auto mask = mx::astype(
      mx::greater(mx::astype(loss_mask, mx::float32), mx::array(0.0f, mx::float32)),
      mx::float32);
  return nll * mask;
}

mx::array first_byte_masked_cross_entropy_mean(
    const mx::array& logits,
    const mx::array& targets,
    const mx::array& first_byte_valid) {
  if (first_byte_valid.ndim() != 1 || first_byte_valid.shape(0) != logits.shape(1)) {
    throw std::runtime_error("first-byte mask must be a rank-1 vector matching vocab size");
  }
  auto valid_i32 = mx::astype(first_byte_valid, mx::int32);
  auto valid = mx::greater(valid_i32, mx::array(0, mx::int32));
  auto target_valid = mx::greater(
      mx::take(valid_i32, targets, 0),
      mx::array(0, mx::int32));
  auto masked_logits = mx::where(
      mx::expand_dims(valid, 0),
      logits,
      mx::full_like(logits, -1e9f));
  auto effective_logits = mx::where(mx::expand_dims(target_valid, 1), masked_logits, logits);
  return cross_entropy_mean(effective_logits, targets);
}

mx::array distillation_kl_mean(
    const mx::array& student_logits,
    const mx::array& teacher_probs) {
  if (student_logits.ndim() != 2 || teacher_probs.ndim() != 2 ||
      student_logits.shape(0) != teacher_probs.shape(0) ||
      student_logits.shape(1) != teacher_probs.shape(1)) {
    throw std::runtime_error("teacher_probs must match student logits shape [rows, vocab]");
  }
  auto row_max = mx::max(student_logits, 1, true);
  auto shifted = student_logits - row_max;
  auto log_norm = mx::log(mx::sum(mx::exp(shifted), 1, true));
  auto student_log_probs = shifted - log_norm;
  auto p = mx::astype(teacher_probs, mx::float32);
  auto safe_p = mx::maximum(p, mx::array(1e-20f, mx::float32));
  auto row_kl = mx::sum(p * (mx::log(safe_p) - student_log_probs), 1);
  return mx::mean(row_kl);
}

mx::array as_float32(const mx::array& x) {
  return mx::astype(x, mx::float32);
}

mx::array clamp_float32(const mx::array& x, float lo, float hi) {
  auto lo_arr = mx::array(lo, mx::float32);
  auto hi_arr = mx::array(hi, mx::float32);
  return mx::minimum(mx::maximum(as_float32(x), lo_arr), hi_arr);
}

mx::array stable_exp_nonpos(const mx::array& x) {
  return mx::exp(clamp_float32(x, kGatedDeltaExpClampMin, kGatedDeltaExpClampMax));
}

bool use_chunked_gated_delta_scan_cuda_fast_path() {
  const char* override = std::getenv("MIXLAB_GATED_DELTA_ALLOW_CUDA_CHUNKED");
  if (override != nullptr && std::string(override) == "1") {
    return true;
  }
  return false;
}

bool fuse_gated_delta_chunk_loop() {
  const char* override = std::getenv("MIXLAB_GATED_DELTA_FUSE_CHUNK_LOOP");
  return override != nullptr && std::string(override) == "1";
}

bool cuda_gpu_available() {
#ifdef __linux__
  return mx::is_available(mx::Device::gpu);
#else
  return false;
#endif
}

mx::array running_variance_raw(const mx::array& x_flat, int B, int T, int D, float alpha) {
  auto x = mx::reshape(x_flat, {B, T, D});
  auto out = mx::zeros({B, T, D}, mx::float32);
  const float one_minus_alpha = 1.0f - alpha;
  for (int b = 0; b < B; ++b) {
    auto x_bt = mx::slice(x, {b, 0, 0}, {b + 1, T, D});
    auto x0 = mx::reshape(mx::slice(x_bt, {0, 0, 0}, {1, 1, D}), {D});
    auto mean = x0;
    auto vari = mx::full({D}, 0.01f, mx::float32);
    for (int t = 0; t < T; ++t) {
      auto xt = mx::reshape(mx::slice(x_bt, {0, t, 0}, {1, t + 1, D}), {D});
      auto diff = xt - mean;
      mean = one_minus_alpha * mean + alpha * xt;
      vari = one_minus_alpha * vari + alpha * mx::square(diff);
      out = mx::slice_update(out, mx::reshape(vari, {1, 1, D}), mx::Shape{b, t, 0}, mx::Shape{b + 1, t + 1, D});
    }
  }
  return out;
}

mx::array softplus(const mx::array& x) {
  return mx::log(1.0f + mx::exp(x));
}

mx::array rotate_pairs_rt_all(const mx::array& x_btn, const mx::array& phi_btdk, int B, int T, int D, int N) {
  const int K = N / 2;
  auto even = mx::reshape(mx::slice(x_btn, {0, 0, 0}, {B, T, N}, {1, 1, 2}), {B, T, 1, K});
  auto odd = mx::reshape(mx::slice(x_btn, {0, 0, 1}, {B, T, N}, {1, 1, 2}), {B, T, 1, K});
  auto cos_phi = mx::cos(phi_btdk);
  auto sin_phi = mx::sin(phi_btdk);
  auto rot_even = cos_phi * even + sin_phi * odd;
  auto rot_odd = -sin_phi * even + cos_phi * odd;
  return mx::reshape(mx::stack({rot_even, rot_odd}, 4), {B, T, D, N});
}

mx::array rotate_grouped_pairs_rt_all(
    const mx::array& x_btgn,
    const mx::array& phi_btdk,
    int B,
    int T,
    int D,
    int N,
    int G) {
  const int channels_per_group = D / G;
  std::vector<mx::array> rotated;
  rotated.reserve(static_cast<size_t>(G));
  for (int g = 0; g < G; ++g) {
    auto x_g = mx::reshape(mx::slice(x_btgn, {0, 0, g, 0}, {B, T, g + 1, N}), {B, T, N});
    auto phi_g = mx::slice(
        phi_btdk,
        {0, 0, g * channels_per_group, 0},
        {B, T, (g + 1) * channels_per_group, N / 2});
    rotated.push_back(rotate_pairs_rt_all(x_g, phi_g, B, T, channels_per_group, N));
  }
  if (rotated.size() == 1) {
    return rotated[0];
  }
  return mx::concatenate(rotated, 2);
}

mx::array concat_or_single(const std::vector<mx::array>& arrays, int axis) {
  if (arrays.empty()) {
    throw std::runtime_error("concat_or_single requires at least one array");
  }
  if (arrays.size() == 1) {
    return arrays[0];
  }
  return mx::concatenate(arrays, axis);
}

int mamba3_channel_chunk_size(int B, int T, int D, int N) {
  const char* raw = std::getenv("MIXLAB_MAMBA3_CHANNEL_CHUNK");
  if (!raw || raw[0] == '\0') {
    // Bound [B,T,D,N] scan/VJP intermediates without changing the recurrence.
    // At B=1,T=4096,N=16 this selects 16 channels per chunk.
    constexpr long long kTargetChunkElements = 1LL << 20;
    const long long per_channel_elements = static_cast<long long>(std::max(1, B)) *
        static_cast<long long>(std::max(1, T)) *
        static_cast<long long>(std::max(1, N));
    const long long channels = std::max(1LL, kTargetChunkElements / per_channel_elements);
    return static_cast<int>(std::min<long long>(D, channels));
  }
  char* end = nullptr;
  long parsed = std::strtol(raw, &end, 10);
  if (end == raw || (end && *end != '\0') || parsed <= 0) {
    return 0;
  }
  if (parsed > D) {
    return D;
  }
  return static_cast<int>(parsed);
}

struct AffineScanPair {
  mx::array alpha;
  mx::array input;
};

AffineScanPair affine_scan_hillis_steele(
    const mx::array& alpha,
    const mx::array& input,
    int B,
    int T,
    int D,
    int N) {
  auto scan_alpha = alpha;
  auto scan_input = input;
  for (int step = 1; step < T; step <<= 1) {
    auto head_alpha = mx::slice(scan_alpha, {0, 0, 0, 0}, {B, step, D, N});
    auto head_input = mx::slice(scan_input, {0, 0, 0, 0}, {B, step, D, N});
    auto tail_alpha = mx::slice(scan_alpha, {0, step, 0, 0}, {B, T, D, N});
    auto tail_input = mx::slice(scan_input, {0, step, 0, 0}, {B, T, D, N});
    auto prev_alpha = mx::slice(scan_alpha, {0, 0, 0, 0}, {B, T - step, D, N});
    auto prev_input = mx::slice(scan_input, {0, 0, 0, 0}, {B, T - step, D, N});

    auto updated_alpha = tail_alpha * prev_alpha;
    auto updated_input = tail_alpha * prev_input + tail_input;
    scan_alpha = mx::concatenate({head_alpha, updated_alpha}, 1);
    scan_input = mx::concatenate({head_input, updated_input}, 1);
  }
  return AffineScanPair{scan_alpha, scan_input};
}

AffineScanPair affine_scan_suffix_hillis_steele(
    const mx::array& alpha,
    const mx::array& input,
    int B,
    int T,
    int D,
    int N) {
  auto scan_alpha = alpha;
  auto scan_input = input;
  for (int step = 1; step < T; step <<= 1) {
    auto left_alpha = mx::slice(scan_alpha, {0, 0, 0, 0}, {B, T - step, D, N});
    auto left_input = mx::slice(scan_input, {0, 0, 0, 0}, {B, T - step, D, N});
    auto right_alpha = mx::slice(scan_alpha, {0, step, 0, 0}, {B, T, D, N});
    auto right_input = mx::slice(scan_input, {0, step, 0, 0}, {B, T, D, N});
    auto tail_alpha = mx::slice(scan_alpha, {0, T - step, 0, 0}, {B, T, D, N});
    auto tail_input = mx::slice(scan_input, {0, T - step, 0, 0}, {B, T, D, N});

    auto updated_alpha = left_alpha * right_alpha;
    auto updated_input = left_input + left_alpha * right_input;
    scan_alpha = mx::concatenate({updated_alpha, tail_alpha}, 1);
    scan_input = mx::concatenate({updated_input, tail_input}, 1);
  }
  return AffineScanPair{scan_alpha, scan_input};
}

mx::array affine_scan_chunked(
    const mx::array& alpha,
    const mx::array& input,
    int B,
    int T,
    int D,
    int N,
    int chunk_size) {
  if (chunk_size <= 0 || chunk_size >= T) {
    return affine_scan_hillis_steele(alpha, input, B, T, D, N).input;
  }

  const int chunk = std::max(1, chunk_size);
  const int n_chunks = (T + chunk - 1) / chunk;
  std::vector<mx::array> local_alphas;
  std::vector<mx::array> local_inputs;
  std::vector<mx::array> chunk_alphas;
  std::vector<mx::array> chunk_inputs;
  local_alphas.reserve(static_cast<size_t>(n_chunks));
  local_inputs.reserve(static_cast<size_t>(n_chunks));
  chunk_alphas.reserve(static_cast<size_t>(n_chunks));
  chunk_inputs.reserve(static_cast<size_t>(n_chunks));

  for (int start = 0; start < T; start += chunk) {
    const int end = std::min(T, start + chunk);
    const int len = end - start;
    auto local_alpha = mx::slice(alpha, {0, start, 0, 0}, {B, end, D, N});
    auto local_input = mx::slice(input, {0, start, 0, 0}, {B, end, D, N});
    auto local = affine_scan_hillis_steele(local_alpha, local_input, B, len, D, N);
    local_alphas.push_back(local.alpha);
    local_inputs.push_back(local.input);
    chunk_alphas.push_back(mx::slice(local.alpha, {0, len - 1, 0, 0}, {B, len, D, N}));
    chunk_inputs.push_back(mx::slice(local.input, {0, len - 1, 0, 0}, {B, len, D, N}));
  }

  auto summary_alpha = mx::concatenate(chunk_alphas, 1);
  auto summary_input = mx::concatenate(chunk_inputs, 1);
  auto summary = affine_scan_hillis_steele(summary_alpha, summary_input, B, n_chunks, D, N);

  std::vector<mx::array> adjusted;
  adjusted.reserve(static_cast<size_t>(n_chunks));
  for (int c = 0; c < n_chunks; ++c) {
    mx::array prior = c == 0
        ? mx::zeros({B, 1, D, N}, mx::float32)
        : mx::slice(summary.input, {0, c - 1, 0, 0}, {B, c, D, N});
    adjusted.push_back(local_alphas[static_cast<size_t>(c)] * prior + local_inputs[static_cast<size_t>(c)]);
  }
  return mx::concatenate(adjusted, 1);
}

// Small shape helpers for the closed-form Mamba3 VJP below. They keep the
// t-1/t+1 recurrence terms vectorized instead of building per-token loops.
mx::array shift_time_right_zero3(const mx::array& x, int B, int T, int D) {
  if (T <= 1) {
    return mx::zeros({B, T, D}, mx::float32);
  }
  return mx::concatenate(
      {mx::zeros({B, 1, D}, mx::float32),
       mx::slice(x, {0, 0, 0}, {B, T - 1, D})},
      1);
}

mx::array causal_depthwise_conv1d(
    const mx::array& x_flat,
    const mx::array& weight,
    int B,
    int T,
    int D,
    int K) {
  if (B <= 0 || T <= 0 || D <= 0 || K <= 0) {
    throw std::runtime_error("OP_DEPTHWISE_CONV1D requires positive B,T,D,K");
  }
  auto x = mx::reshape(x_flat, {B, T, D});
  auto out = mx::zeros({B, T, D}, mx::float32);
  for (int k = 0; k < K && k < T; ++k) {
    auto shifted = k == 0
        ? x
        : mx::concatenate(
              {mx::zeros({B, k, D}, mx::float32),
               mx::slice(x, {0, 0, 0}, {B, T - k, D})},
              1);
    auto wk = mx::reshape(mx::slice(weight, {0, k}, {D, k + 1}), {1, 1, D});
    out = out + shifted * wk;
  }
  return mx::reshape(out, {B * T, D});
}

mx::array shift_time_left_zero4(const mx::array& x, int B, int T, int D, int N) {
  if (T <= 1) {
    return mx::zeros({B, T, D, N}, mx::float32);
  }
  return mx::concatenate(
      {mx::slice(x, {0, 1, 0, 0}, {B, T, D, N}),
       mx::zeros({B, 1, D, N}, mx::float32)},
      1);
}

mx::array shift_time_right_zero4(const mx::array& x, int B, int T, int D, int N) {
  if (T <= 1) {
    return mx::zeros({B, T, D, N}, mx::float32);
  }
  return mx::concatenate(
      {mx::zeros({B, 1, D, N}, mx::float32),
       mx::slice(x, {0, 0, 0, 0}, {B, T - 1, D, N})},
      1);
}

std::vector<mx::array> rotate_grouped_pairs_rt_all_vjp(
    const mx::array& grad_b_rot,
    const mx::array& grad_c_rot,
    const mx::array& b_rot,
    const mx::array& c_rot,
    const mx::array& phi,
    int B,
    int T,
    int D,
    int N,
    int G) {
  const int K = N / 2;
  const int channels_per_group = D / G;
  auto gb0 = mx::reshape(mx::slice(grad_b_rot, {0, 0, 0, 0}, {B, T, D, N}, {1, 1, 1, 2}), {B, T, D, K});
  auto gb1 = mx::reshape(mx::slice(grad_b_rot, {0, 0, 0, 1}, {B, T, D, N}, {1, 1, 1, 2}), {B, T, D, K});
  auto gc0 = mx::reshape(mx::slice(grad_c_rot, {0, 0, 0, 0}, {B, T, D, N}, {1, 1, 1, 2}), {B, T, D, K});
  auto gc1 = mx::reshape(mx::slice(grad_c_rot, {0, 0, 0, 1}, {B, T, D, N}, {1, 1, 1, 2}), {B, T, D, K});
  auto b0 = mx::reshape(mx::slice(b_rot, {0, 0, 0, 0}, {B, T, D, N}, {1, 1, 1, 2}), {B, T, D, K});
  auto b1 = mx::reshape(mx::slice(b_rot, {0, 0, 0, 1}, {B, T, D, N}, {1, 1, 1, 2}), {B, T, D, K});
  auto c0 = mx::reshape(mx::slice(c_rot, {0, 0, 0, 0}, {B, T, D, N}, {1, 1, 1, 2}), {B, T, D, K});
  auto c1 = mx::reshape(mx::slice(c_rot, {0, 0, 0, 1}, {B, T, D, N}, {1, 1, 1, 2}), {B, T, D, K});
  auto cos_phi = mx::cos(phi);
  auto sin_phi = mx::sin(phi);

  auto grad_b0_by_channel = cos_phi * gb0 - sin_phi * gb1;
  auto grad_b1_by_channel = sin_phi * gb0 + cos_phi * gb1;
  auto grad_c0_by_channel = cos_phi * gc0 - sin_phi * gc1;
  auto grad_c1_by_channel = sin_phi * gc0 + cos_phi * gc1;
  auto grad_phi = b1 * gb0 - b0 * gb1 + c1 * gc0 - c0 * gc1;

  std::vector<mx::array> grad_b_groups;
  std::vector<mx::array> grad_c_groups;
  grad_b_groups.reserve(static_cast<size_t>(G));
  grad_c_groups.reserve(static_cast<size_t>(G));
  for (int g = 0; g < G; ++g) {
    const int d0 = g * channels_per_group;
    const int d1 = (g + 1) * channels_per_group;
    auto gb0_g = mx::sum(mx::slice(grad_b0_by_channel, {0, 0, d0, 0}, {B, T, d1, K}), 2);
    auto gb1_g = mx::sum(mx::slice(grad_b1_by_channel, {0, 0, d0, 0}, {B, T, d1, K}), 2);
    auto gc0_g = mx::sum(mx::slice(grad_c0_by_channel, {0, 0, d0, 0}, {B, T, d1, K}), 2);
    auto gc1_g = mx::sum(mx::slice(grad_c1_by_channel, {0, 0, d0, 0}, {B, T, d1, K}), 2);
    grad_b_groups.push_back(mx::reshape(mx::stack({gb0_g, gb1_g}, 3), {B, T, 1, N}));
    grad_c_groups.push_back(mx::reshape(mx::stack({gc0_g, gc1_g}, 3), {B, T, 1, N}));
  }
  return {
      mx::reshape(mx::concatenate(grad_b_groups, 2), {B * T, G * N}),
      mx::reshape(mx::concatenate(grad_c_groups, 2), {B * T, G * N}),
      grad_phi};
}

mx::array mamba3_selective_scan_canonical_phase6_impl_channel_chunked(
    const mx::array& x_flat,
    const mx::array& dt_flat,
    const mx::array& lambda_flat,
    const mx::array& theta_flat,
    const mx::array& a_log,
    const mx::array& b_proj_flat,
    const mx::array& c_proj_flat,
    int B,
    int T,
    int D,
    int N,
    int G,
    int scan_chunk_size,
    int channel_chunk_size);

std::vector<mx::array> mamba3_selective_scan_canonical_phase6_vjp_channel_chunked(
    const std::vector<mx::array>& args,
    const std::vector<mx::array>& cotangents,
    int B,
    int T,
    int D,
    int N,
    int G,
    int scan_chunk_size,
    int channel_chunk_size);

// Exact VJP for the canonical Mamba3 scan. This avoids asking MLX autodiff to
// differentiate through the full parallel scan, which was creating an
// oversized compiled CUDA graph at D=448, T=4096, 8 layers.
std::vector<mx::array> mamba3_selective_scan_canonical_phase6_vjp(
    const std::vector<mx::array>& args,
    const std::vector<mx::array>&,
    const std::vector<mx::array>& cotangents,
    int B,
    int T,
    int D,
    int N,
    int G,
    int scan_chunk_size) {
  if (mlx_ir::mamba3_selective_scan_cuda_primitive_available(N)) {
    return mlx_ir::mamba3_selective_scan_cuda_vjp(args, cotangents, B, T, D, N, G);
  }

  const int channel_chunk_size = mamba3_channel_chunk_size(B, T, D, N);
  if (channel_chunk_size > 0 && channel_chunk_size < D) {
    return mamba3_selective_scan_canonical_phase6_vjp_channel_chunked(
        args, cotangents, B, T, D, N, G, scan_chunk_size, channel_chunk_size);
  }

  const auto& x_flat = args[0];
  const auto& dt_flat = args[1];
  const auto& lambda_flat = args[2];
  const auto& theta_flat = args[3];
  const auto& a_log = args[4];
  const auto& b_proj_flat = args[5];
  const auto& c_proj_flat = args[6];

  auto x = mx::reshape(x_flat, {B, T, D});
  auto dt_raw = mx::reshape(dt_flat, {B, T, D});
  auto dt = softplus(dt_raw);
  auto lambda = mx::sigmoid(mx::reshape(lambda_flat, {B, T, D}));
  auto theta = mx::reshape(theta_flat, {B, T, D, N / 2});
  auto A = -mx::exp(a_log);
  auto b_proj = mx::reshape(b_proj_flat, {B, T, G, N});
  auto c_proj = mx::reshape(c_proj_flat, {B, T, G, N});
  auto phi = mx::cumsum(mx::expand_dims(dt, -1) * theta, 1);

  auto A_btdn = mx::reshape(A, {1, 1, D, N});
  auto dt_btd1 = mx::expand_dims(dt, -1);
  auto lambda_btd1 = mx::expand_dims(lambda, -1);
  auto alpha = mx::exp(dt_btd1 * A_btdn);
  auto beta = (1.0f - lambda_btd1) * dt_btd1 * alpha;
  auto gamma = lambda_btd1 * dt_btd1;
  auto b_rot = rotate_grouped_pairs_rt_all(b_proj, phi, B, T, D, N, G);
  auto c_rot = rotate_grouped_pairs_rt_all(c_proj, phi, B, T, D, N, G);
  auto current = gamma * b_rot * mx::expand_dims(x, -1);
  auto previous = beta * shift_time_right_zero4(b_rot, B, T, D, N) * mx::expand_dims(shift_time_right_zero3(x, B, T, D), -1);
  auto h_after = affine_scan_chunked(alpha, current + previous, B, T, D, N, scan_chunk_size);
  auto h_before = shift_time_right_zero4(h_after, B, T, D, N);

  auto dy = mx::reshape(cotangents[0], {B, T, D});
  auto local_dh = mx::expand_dims(dy, -1) * c_rot;
  auto alpha_next = shift_time_left_zero4(alpha, B, T, D, N);
  auto upstream = affine_scan_suffix_hillis_steele(alpha_next, local_dh, B, T, D, N).input;

  auto grad_c_rot = mx::expand_dims(dy, -1) * h_after;
  auto beta_next = shift_time_left_zero4(beta, B, T, D, N);
  auto upstream_next = shift_time_left_zero4(upstream, B, T, D, N);
  auto grad_x = mx::sum(gamma * b_rot * upstream, 3) + mx::sum(beta_next * b_rot * upstream_next, 3);
  auto grad_b_rot = gamma * mx::expand_dims(x, -1) * upstream + beta_next * mx::expand_dims(x, -1) * upstream_next;

  auto prev_x = shift_time_right_zero3(x, B, T, D);
  auto prev_b_rot = shift_time_right_zero4(b_rot, B, T, D, N);
  auto prev_input = prev_b_rot * mx::expand_dims(prev_x, -1);
  auto current_input = b_rot * mx::expand_dims(x, -1);
  auto delta_term =
      A_btdn * alpha * h_before +
      (1.0f - lambda_btd1) * (alpha + dt_btd1 * A_btdn * alpha) * prev_input +
      lambda_btd1 * current_input;
  auto a_log_term =
      dt_btd1 * alpha * A_btdn * h_before +
      (1.0f - lambda_btd1) * dt_btd1 * dt_btd1 * alpha * A_btdn * prev_input;
  auto grad_delta = mx::sum(delta_term * upstream, 3);
  auto grad_a_log = mx::sum(a_log_term * upstream, {0, 1});
  auto grad_lambda =
      mx::sum((-dt_btd1 * alpha * prev_input + dt_btd1 * current_input) * upstream, 3) *
      lambda * (1.0f - lambda);

  auto rot_grads = rotate_grouped_pairs_rt_all_vjp(grad_b_rot, grad_c_rot, b_rot, c_rot, phi, B, T, D, N, G);
  auto grad_phi = rot_grads[2];
  auto phi_carry = mx::cumsum(grad_phi, 1, true);
  auto grad_theta = phi_carry * mx::expand_dims(dt, -1);
  grad_delta = grad_delta + mx::sum(phi_carry * theta, 3);
  auto grad_dt = grad_delta * mx::sigmoid(dt_raw);

  return {
      mx::reshape(grad_x, {B * T, D}),
      mx::reshape(grad_dt, {B * T, D}),
      mx::reshape(grad_lambda, {B * T, D}),
      mx::reshape(grad_theta, {B * T, D * (N / 2)}),
      grad_a_log,
      rot_grads[0],
      rot_grads[1]};
}

std::vector<mx::array> mamba3_selective_scan_canonical_phase6_vjp_channel_chunked(
    const std::vector<mx::array>& args,
    const std::vector<mx::array>& cotangents,
    int B,
    int T,
    int D,
    int N,
    int G,
    int scan_chunk_size,
    int channel_chunk_size) {
  const auto& x_flat = args[0];
  const auto& dt_flat = args[1];
  const auto& lambda_flat = args[2];
  const auto& theta_flat = args[3];
  const auto& a_log = args[4];
  const auto& b_proj_flat = args[5];
  const auto& c_proj_flat = args[6];

  const int K = N / 2;
  const int channels_per_group = D / G;
  auto x = mx::reshape(x_flat, {B, T, D});
  auto dt_raw_all = mx::reshape(dt_flat, {B, T, D});
  auto lambda_raw_all = mx::reshape(lambda_flat, {B, T, D});
  auto theta_all = mx::reshape(theta_flat, {B, T, D, K});
  auto b_proj = mx::reshape(b_proj_flat, {B, T, G, N});
  auto c_proj = mx::reshape(c_proj_flat, {B, T, G, N});
  auto dy_all = mx::reshape(cotangents[0], {B, T, D});

  std::vector<mx::array> grad_x_chunks;
  std::vector<mx::array> grad_dt_chunks;
  std::vector<mx::array> grad_lambda_chunks;
  std::vector<mx::array> grad_theta_chunks;
  std::vector<mx::array> grad_a_log_chunks;
  std::vector<mx::array> grad_b_groups;
  std::vector<mx::array> grad_c_groups;
  grad_x_chunks.reserve(static_cast<size_t>((D + channel_chunk_size - 1) / channel_chunk_size));
  grad_dt_chunks.reserve(grad_x_chunks.capacity());
  grad_lambda_chunks.reserve(grad_x_chunks.capacity());
  grad_theta_chunks.reserve(grad_x_chunks.capacity());
  grad_a_log_chunks.reserve(grad_x_chunks.capacity());
  grad_b_groups.reserve(static_cast<size_t>(G));
  grad_c_groups.reserve(static_cast<size_t>(G));

  for (int g = 0; g < G; ++g) {
    const int group_d0 = g * channels_per_group;
    const int group_d1 = (g + 1) * channels_per_group;
    auto b_g = mx::reshape(mx::slice(b_proj, {0, 0, g, 0}, {B, T, g + 1, N}), {B, T, N});
    auto c_g = mx::reshape(mx::slice(c_proj, {0, 0, g, 0}, {B, T, g + 1, N}), {B, T, N});
    mx::array grad_b_g = mx::zeros({B, T, 1, N}, mx::float32);
    mx::array grad_c_g = mx::zeros({B, T, 1, N}, mx::float32);

    for (int d0 = group_d0; d0 < group_d1; d0 += channel_chunk_size) {
      const int d1 = std::min(group_d1, d0 + channel_chunk_size);
      const int DC = d1 - d0;
      auto x_c = mx::slice(x, {0, 0, d0}, {B, T, d1});
      auto dt_raw = mx::slice(dt_raw_all, {0, 0, d0}, {B, T, d1});
      auto dt = softplus(dt_raw);
      auto lambda = mx::sigmoid(mx::slice(lambda_raw_all, {0, 0, d0}, {B, T, d1}));
      auto theta = mx::slice(theta_all, {0, 0, d0, 0}, {B, T, d1, K});
      auto A_c = mx::slice(a_log, {d0, 0}, {d1, N});
      auto A = -mx::exp(A_c);
      auto dy = mx::slice(dy_all, {0, 0, d0}, {B, T, d1});
      auto phi = mx::cumsum(mx::expand_dims(dt, -1) * theta, 1);

      auto A_btdn = mx::reshape(A, {1, 1, DC, N});
      auto dt_btd1 = mx::expand_dims(dt, -1);
      auto lambda_btd1 = mx::expand_dims(lambda, -1);
      auto alpha = mx::exp(dt_btd1 * A_btdn);
      auto beta = (1.0f - lambda_btd1) * dt_btd1 * alpha;
      auto gamma = lambda_btd1 * dt_btd1;
      auto b_rot = rotate_pairs_rt_all(b_g, phi, B, T, DC, N);
      auto c_rot = rotate_pairs_rt_all(c_g, phi, B, T, DC, N);
      auto current = gamma * b_rot * mx::expand_dims(x_c, -1);
      auto previous = beta * shift_time_right_zero4(b_rot, B, T, DC, N) *
          mx::expand_dims(shift_time_right_zero3(x_c, B, T, DC), -1);
      auto h_after = affine_scan_chunked(alpha, current + previous, B, T, DC, N, scan_chunk_size);
      auto h_before = shift_time_right_zero4(h_after, B, T, DC, N);

      auto local_dh = mx::expand_dims(dy, -1) * c_rot;
      auto alpha_next = shift_time_left_zero4(alpha, B, T, DC, N);
      auto upstream = affine_scan_suffix_hillis_steele(alpha_next, local_dh, B, T, DC, N).input;

      auto grad_c_rot = mx::expand_dims(dy, -1) * h_after;
      auto beta_next = shift_time_left_zero4(beta, B, T, DC, N);
      auto upstream_next = shift_time_left_zero4(upstream, B, T, DC, N);
      auto grad_x = mx::sum(gamma * b_rot * upstream, 3) + mx::sum(beta_next * b_rot * upstream_next, 3);
      auto grad_b_rot = gamma * mx::expand_dims(x_c, -1) * upstream +
          beta_next * mx::expand_dims(x_c, -1) * upstream_next;

      auto prev_x = shift_time_right_zero3(x_c, B, T, DC);
      auto prev_b_rot = shift_time_right_zero4(b_rot, B, T, DC, N);
      auto prev_input = prev_b_rot * mx::expand_dims(prev_x, -1);
      auto current_input = b_rot * mx::expand_dims(x_c, -1);
      auto delta_term =
          A_btdn * alpha * h_before +
          (1.0f - lambda_btd1) * (alpha + dt_btd1 * A_btdn * alpha) * prev_input +
          lambda_btd1 * current_input;
      auto a_log_term =
          dt_btd1 * alpha * A_btdn * h_before +
          (1.0f - lambda_btd1) * dt_btd1 * dt_btd1 * alpha * A_btdn * prev_input;
      auto grad_delta = mx::sum(delta_term * upstream, 3);
      auto grad_a_log = mx::sum(a_log_term * upstream, {0, 1});
      auto grad_lambda =
          mx::sum((-dt_btd1 * alpha * prev_input + dt_btd1 * current_input) * upstream, 3) *
          lambda * (1.0f - lambda);

      auto rot_grads = rotate_grouped_pairs_rt_all_vjp(grad_b_rot, grad_c_rot, b_rot, c_rot, phi, B, T, DC, N, 1);
      auto grad_phi = rot_grads[2];
      auto phi_carry = mx::cumsum(grad_phi, 1, true);
      auto grad_theta = phi_carry * mx::expand_dims(dt, -1);
      grad_delta = grad_delta + mx::sum(phi_carry * theta, 3);
      auto grad_dt = grad_delta * mx::sigmoid(dt_raw);

      grad_x_chunks.push_back(grad_x);
      grad_dt_chunks.push_back(grad_dt);
      grad_lambda_chunks.push_back(grad_lambda);
      grad_theta_chunks.push_back(grad_theta);
      grad_a_log_chunks.push_back(grad_a_log);
      grad_b_g = grad_b_g + mx::reshape(rot_grads[0], {B, T, 1, N});
      grad_c_g = grad_c_g + mx::reshape(rot_grads[1], {B, T, 1, N});
    }
    grad_b_groups.push_back(grad_b_g);
    grad_c_groups.push_back(grad_c_g);
  }

  auto grad_x = concat_or_single(grad_x_chunks, 2);
  auto grad_dt = concat_or_single(grad_dt_chunks, 2);
  auto grad_lambda = concat_or_single(grad_lambda_chunks, 2);
  auto grad_theta = concat_or_single(grad_theta_chunks, 2);
  auto grad_a_log = concat_or_single(grad_a_log_chunks, 0);
  auto grad_b = concat_or_single(grad_b_groups, 2);
  auto grad_c = concat_or_single(grad_c_groups, 2);
  return {
      mx::reshape(grad_x, {B * T, D}),
      mx::reshape(grad_dt, {B * T, D}),
      mx::reshape(grad_lambda, {B * T, D}),
      mx::reshape(grad_theta, {B * T, D * K}),
      grad_a_log,
      mx::reshape(grad_b, {B * T, G * N}),
      mx::reshape(grad_c, {B * T, G * N})};
}

mx::array mamba3_selective_scan_canonical_phase6_impl(
    const mx::array& x_flat,
    const mx::array& dt_flat,
    const mx::array& lambda_flat,
    const mx::array& theta_flat,
    const mx::array& a_log,
    const mx::array& b_proj_flat,
    const mx::array& c_proj_flat,
    int B,
    int T,
    int D,
    int N,
    int G,
    int scan_chunk_size) {
  if (G <= 0) {
    throw std::runtime_error("Mamba3 Phase 6 requires n_groups > 0; got G=" + std::to_string(G));
  }
  if ((D % G) != 0) {
    throw std::runtime_error(
        "Mamba3 Phase 6 requires D divisible by n_groups; got D=" + std::to_string(D) +
        " G=" + std::to_string(G));
  }
  if ((N % 2) != 0) {
    throw std::runtime_error("Mamba3 Phase 6 requires even state_size N for 2x2 rotations; got N=" + std::to_string(N));
  }

  if (mlx_ir::mamba3_selective_scan_cuda_primitive_available(N)) {
    return mlx_ir::mamba3_selective_scan_cuda_forward(
        x_flat, dt_flat, lambda_flat, theta_flat, a_log, b_proj_flat, c_proj_flat,
        B, T, D, N, G);
  }

  const int channel_chunk_size = mamba3_channel_chunk_size(B, T, D, N);
  if (channel_chunk_size > 0 && channel_chunk_size < D) {
    return mamba3_selective_scan_canonical_phase6_impl_channel_chunked(
        x_flat, dt_flat, lambda_flat, theta_flat, a_log, b_proj_flat, c_proj_flat,
        B, T, D, N, G, scan_chunk_size, channel_chunk_size);
  }

  // Canonical Mamba-3 grouped/MIMO scan:
  // Sec. 3.1 Prop. 1 Eq. (5)-(6) gives the exponential-trapezoidal
  // alpha/beta/gamma recurrence, Sec. 3.2 Prop. 2-4 Eq. (9), Eq. (11), and
  // proof Eq. (25) give the real 2x2 / RoPE state rotations, and Sec. 3.3
  // Eq. (12)-(13) plus Appendix C define grouped MIMO B/C. All
  // timestep-parallel terms are hoisted, then h_t = alpha_t*h_{t-1}+input_t
  // is evaluated by a Hillis-Steele associative scan over affine pairs.
  auto x = mx::reshape(x_flat, {B, T, D});
  auto dt = softplus(mx::reshape(dt_flat, {B, T, D}));
  auto lambda = mx::sigmoid(mx::reshape(lambda_flat, {B, T, D}));
  auto theta = mx::reshape(theta_flat, {B, T, D, N / 2});
  auto A = -mx::exp(a_log);
  auto b_proj = mx::reshape(b_proj_flat, {B, T, G, N});
  auto c_proj = mx::reshape(c_proj_flat, {B, T, G, N});

  auto phi = mx::cumsum(mx::expand_dims(dt, -1) * theta, 1);
  auto A_btdn = mx::reshape(A, {1, 1, D, N});
  auto dt_btd1 = mx::expand_dims(dt, -1);
  auto lambda_btd1 = mx::expand_dims(lambda, -1);
  auto alpha_all = mx::exp(dt_btd1 * A_btdn);
  auto beta_all = (1.0f - lambda_btd1) * dt_btd1 * alpha_all;
  auto gamma_all = lambda_btd1 * dt_btd1;

  auto b_rot = rotate_grouped_pairs_rt_all(b_proj, phi, B, T, D, N, G);
  auto c_rot = rotate_grouped_pairs_rt_all(c_proj, phi, B, T, D, N, G);
  auto current_all = gamma_all * b_rot * mx::expand_dims(x, -1);
  auto previous_all = mx::zeros({B, T, D, N}, mx::float32);
  if (T > 1) {
    auto beta_tail = mx::slice(beta_all, {0, 1, 0, 0}, {B, T, D, N});
    auto b_prev = mx::slice(b_rot, {0, 0, 0, 0}, {B, T - 1, D, N});
    auto x_prev = mx::slice(x, {0, 0, 0}, {B, T - 1, D});
    auto previous_tail = beta_tail * b_prev * mx::expand_dims(x_prev, -1);
    previous_all = mx::concatenate({mx::zeros({B, 1, D, N}, mx::float32), previous_tail}, 1);
  }

  auto scan_input = affine_scan_chunked(alpha_all, current_all + previous_all, B, T, D, N, scan_chunk_size);
  return mx::reshape(mx::sum(scan_input * c_rot, 3), {B * T, D});
}

mx::array mamba3_selective_scan_canonical_phase6_impl_channel_chunked(
    const mx::array& x_flat,
    const mx::array& dt_flat,
    const mx::array& lambda_flat,
    const mx::array& theta_flat,
    const mx::array& a_log,
    const mx::array& b_proj_flat,
    const mx::array& c_proj_flat,
    int B,
    int T,
    int D,
    int N,
    int G,
    int scan_chunk_size,
    int channel_chunk_size) {
  const int K = N / 2;
  const int channels_per_group = D / G;
  auto x = mx::reshape(x_flat, {B, T, D});
  auto dt_all = softplus(mx::reshape(dt_flat, {B, T, D}));
  auto lambda_all = mx::sigmoid(mx::reshape(lambda_flat, {B, T, D}));
  auto theta_all = mx::reshape(theta_flat, {B, T, D, K});
  auto A_all = -mx::exp(a_log);
  auto b_proj = mx::reshape(b_proj_flat, {B, T, G, N});
  auto c_proj = mx::reshape(c_proj_flat, {B, T, G, N});

  std::vector<mx::array> y_chunks;
  y_chunks.reserve(static_cast<size_t>((D + channel_chunk_size - 1) / channel_chunk_size));
  for (int g = 0; g < G; ++g) {
    const int group_d0 = g * channels_per_group;
    const int group_d1 = (g + 1) * channels_per_group;
    auto b_g = mx::reshape(mx::slice(b_proj, {0, 0, g, 0}, {B, T, g + 1, N}), {B, T, N});
    auto c_g = mx::reshape(mx::slice(c_proj, {0, 0, g, 0}, {B, T, g + 1, N}), {B, T, N});

    for (int d0 = group_d0; d0 < group_d1; d0 += channel_chunk_size) {
      const int d1 = std::min(group_d1, d0 + channel_chunk_size);
      const int DC = d1 - d0;
      auto x_c = mx::slice(x, {0, 0, d0}, {B, T, d1});
      auto dt = mx::slice(dt_all, {0, 0, d0}, {B, T, d1});
      auto lambda = mx::slice(lambda_all, {0, 0, d0}, {B, T, d1});
      auto theta = mx::slice(theta_all, {0, 0, d0, 0}, {B, T, d1, K});
      auto A = mx::slice(A_all, {d0, 0}, {d1, N});

      auto phi = mx::cumsum(mx::expand_dims(dt, -1) * theta, 1);
      auto A_btdn = mx::reshape(A, {1, 1, DC, N});
      auto dt_btd1 = mx::expand_dims(dt, -1);
      auto lambda_btd1 = mx::expand_dims(lambda, -1);
      auto alpha_all = mx::exp(dt_btd1 * A_btdn);
      auto beta_all = (1.0f - lambda_btd1) * dt_btd1 * alpha_all;
      auto gamma_all = lambda_btd1 * dt_btd1;
      auto b_rot = rotate_pairs_rt_all(b_g, phi, B, T, DC, N);
      auto c_rot = rotate_pairs_rt_all(c_g, phi, B, T, DC, N);
      auto current_all = gamma_all * b_rot * mx::expand_dims(x_c, -1);
      auto previous_all = mx::zeros({B, T, DC, N}, mx::float32);
      if (T > 1) {
        auto beta_tail = mx::slice(beta_all, {0, 1, 0, 0}, {B, T, DC, N});
        auto b_prev = mx::slice(b_rot, {0, 0, 0, 0}, {B, T - 1, DC, N});
        auto x_prev = mx::slice(x_c, {0, 0, 0}, {B, T - 1, DC});
        auto previous_tail = beta_tail * b_prev * mx::expand_dims(x_prev, -1);
        previous_all = mx::concatenate({mx::zeros({B, 1, DC, N}, mx::float32), previous_tail}, 1);
      }

      auto scan_input = affine_scan_chunked(alpha_all, current_all + previous_all, B, T, DC, N, scan_chunk_size);
      y_chunks.push_back(mx::reshape(mx::sum(scan_input * c_rot, 3), {B, T, DC}));
    }
  }
  return mx::reshape(concat_or_single(y_chunks, 2), {B * T, D});
}

mx::array mamba3_selective_scan_canonical_phase6(
    const mx::array& x_flat,
    const mx::array& dt_flat,
    const mx::array& lambda_flat,
    const mx::array& theta_flat,
    const mx::array& a_log,
    const mx::array& b_proj_flat,
    const mx::array& c_proj_flat,
    int B,
    int T,
    int D,
    int N,
    int G,
    int scan_chunk_size) {
  auto scanned = mx::custom_vjp(
      [B, T, D, N, G, scan_chunk_size](const std::vector<mx::array>& args) {
        return std::vector<mx::array>{
            mamba3_selective_scan_canonical_phase6_impl(
                args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                B, T, D, N, G, scan_chunk_size)};
      },
      [B, T, D, N, G, scan_chunk_size](
          const std::vector<mx::array>& args,
          const std::vector<mx::array>& cotangents,
          const std::vector<mx::array>& outputs) {
        return mamba3_selective_scan_canonical_phase6_vjp(
            args, outputs, cotangents, B, T, D, N, G, scan_chunk_size);
      });
  return scanned({x_flat, dt_flat, lambda_flat, theta_flat, a_log, b_proj_flat, c_proj_flat})[0];
}

struct RMSNormVJPResult {
  mx::array input;
  mx::array scale;
};

mx::array rmsnorm_forward(
    const mx::array& x,
    const mx::array& scale,
    float eps) {
  auto ms = mx::mean(mx::square(x), -1, true);
  auto inv = 1.0f / mx::sqrt(ms + eps);
  return x * inv * scale;
}

RMSNormVJPResult rmsnorm_vjp(
    const mx::array& x,
    const mx::array& scale,
    const mx::array& grad_out,
    float eps) {
  auto ms = mx::mean(mx::square(x), -1, true);
  auto inv = 1.0f / mx::sqrt(ms + eps);
  auto grad_scaled = grad_out * scale;
  auto mean_grad_dot = mx::mean(grad_scaled * x, -1, true);
  auto grad_x = inv * (grad_scaled - x * mean_grad_dot * mx::square(inv));
  auto grad_scale = mx::reshape(mx::sum(grad_out * x * inv, 0), scale.shape());
  return {grad_x, grad_scale};
}

struct DepthwiseConv1DVJPResult {
  mx::array input;
  mx::array weight;
};

DepthwiseConv1DVJPResult causal_depthwise_conv1d_vjp(
    const mx::array& x_flat,
    const mx::array& weight,
    const mx::array& grad_y_flat,
    int B,
    int T,
    int D,
    int K) {
  auto x = mx::reshape(x_flat, {B, T, D});
  auto grad_y = mx::reshape(grad_y_flat, {B, T, D});
  auto grad_x = mx::zeros({B, T, D}, mx::float32);
  std::vector<mx::array> grad_weight_cols;
  grad_weight_cols.reserve(static_cast<size_t>(K));
  for (int k = 0; k < K; ++k) {
    auto wk = mx::reshape(mx::slice(weight, {0, k}, {D, k + 1}), {1, 1, D});
    if (k < T) {
      auto shifted_grad = k == 0
          ? grad_y
          : mx::concatenate(
                {mx::slice(grad_y, {0, k, 0}, {B, T, D}),
                 mx::zeros({B, k, D}, mx::float32)},
                1);
      grad_x = grad_x + shifted_grad * wk;

      auto x_part = k == 0 ? x : mx::slice(x, {0, 0, 0}, {B, T - k, D});
      auto gy_part = k == 0 ? grad_y : mx::slice(grad_y, {0, k, 0}, {B, T, D});
      grad_weight_cols.push_back(mx::sum(x_part * gy_part, {0, 1}));
    } else {
      grad_weight_cols.push_back(mx::zeros({D}, mx::float32));
    }
  }
  return {mx::reshape(grad_x, {B * T, D}), mx::stack(grad_weight_cols, 1)};
}

mx::array silu_derivative(const mx::array& x) {
  auto sig = mx::sigmoid(x);
  return sig * (1.0f + x * (1.0f - sig));
}

int mamba3_canonical_block_expected_inputs(bool use_conv) {
  return use_conv ? 21 : 20;
}

void log_mamba3_canonical_block_once() {
  static std::atomic<bool> logged{false};
  if (!logged.exchange(true)) {
    std::cerr << "[mlx_ir] canonical Mamba3 using fused canonical block VJP"
              << " (set MIXLAB_DISABLE_MAMBA3_LOW_MEMORY_UPDATES=1 only for debugging)"
              << std::endl;
  }
}

mx::array mamba3_canonical_block_impl(
    const std::vector<mx::array>& args,
    int B,
    int T,
    bool use_conv,
    int scan_chunk_size,
    float eps) {
  if (args.size() != static_cast<size_t>(mamba3_canonical_block_expected_inputs(use_conv))) {
    throw std::runtime_error("OP_MAMBA3_CANONICAL_BLOCK input count mismatch");
  }
  int i = 0;
  const auto& x = args[i++];
  const auto& pre_norm = args[i++];
  const auto& w_x = args[i++];
  const mx::array* conv_w = nullptr;
  if (use_conv) {
    conv_w = &args[i++];
  }
  const auto& w_dt_low = args[i++];
  const auto& w_dt_high = args[i++];
  const auto& w_lambda_low = args[i++];
  const auto& w_lambda_high = args[i++];
  const auto& w_theta_low = args[i++];
  const auto& w_theta_high = args[i++];
  const auto& w_b = args[i++];
  const auto& w_c = args[i++];
  const auto& b_norm_scale = args[i++];
  const auto& c_norm_scale = args[i++];
  const auto& b_bias = args[i++];
  const auto& c_bias = args[i++];
  const auto& a_log = args[i++];
  const auto& dt_bias = args[i++];
  const auto& post_norm = args[i++];
  const auto& w_gate = args[i++];
  const auto& w_out = args[i++];

  const int inner = static_cast<int>(w_x.shape(1));
  const int state_size = static_cast<int>(a_log.shape(1));
  const int n_groups = static_cast<int>(w_b.shape(1)) / state_size;
  const int conv_kernel = use_conv ? static_cast<int>(conv_w->shape(1)) : 0;
  const int dt_rank = static_cast<int>(w_dt_low.shape(1));
  const int group_state = n_groups * state_size;
  auto x_norm = rmsnorm_forward(x, pre_norm, eps);
  auto x_proj = mx::matmul(x_norm, w_x);
  auto x_branch = use_conv
      ? causal_depthwise_conv1d(x_proj, *conv_w, B, T, inner, conv_kernel)
      : x_proj;

  auto aux_weight = mx::concatenate({w_dt_low, w_lambda_low, w_theta_low, w_b, w_c}, 1);
  auto aux_proj = mx::matmul(x_branch, aux_weight);
  auto dt_low = mx::slice(aux_proj, {0, 0}, {B * T, dt_rank});
  auto dt = mx::matmul(dt_low, w_dt_high) + dt_bias;
  auto lambda_low = mx::slice(aux_proj, {0, dt_rank}, {B * T, dt_rank * 2});
  auto lambda = mx::matmul(lambda_low, w_lambda_high);
  auto theta_low = mx::slice(aux_proj, {0, dt_rank * 2}, {B * T, dt_rank * 3});
  auto theta = mx::matmul(theta_low, w_theta_high);

  auto b_proj = mx::slice(aux_proj, {0, dt_rank * 3}, {B * T, dt_rank * 3 + group_state});
  auto b_norm = rmsnorm_forward(mx::reshape(b_proj, {B * T * n_groups, state_size}), b_norm_scale, eps);
  auto b_biased = mx::reshape(b_norm, {B * T, n_groups * state_size}) + b_bias;
  auto c_proj = mx::slice(aux_proj, {0, dt_rank * 3 + group_state}, {B * T, dt_rank * 3 + group_state * 2});
  auto c_norm = rmsnorm_forward(mx::reshape(c_proj, {B * T * n_groups, state_size}), c_norm_scale, eps);
  auto c_biased = mx::reshape(c_norm, {B * T, n_groups * state_size}) + c_bias;

  auto y = mamba3_selective_scan_canonical_phase6(
      x_branch, dt, lambda, theta, a_log, b_biased, c_biased,
      B, T, inner, state_size, n_groups, scan_chunk_size);
  auto y_norm = rmsnorm_forward(y, post_norm, eps);
  auto z = mx::matmul(x_norm, w_gate);
  auto y_gated = y_norm * z * mx::sigmoid(z);
  return x + mx::matmul(y_gated, w_out);
}

std::vector<mx::array> mamba3_canonical_block_vjp(
    const std::vector<mx::array>& args,
    const std::vector<mx::array>& cotangents,
    int B,
    int T,
    bool use_conv,
    int scan_chunk_size,
    float eps) {
  if (args.size() != static_cast<size_t>(mamba3_canonical_block_expected_inputs(use_conv)) ||
      cotangents.size() != 1) {
    throw std::runtime_error("OP_MAMBA3_CANONICAL_BLOCK VJP input count mismatch");
  }
  int i = 0;
  const auto& x = args[i++];
  const auto& pre_norm = args[i++];
  const auto& w_x = args[i++];
  const mx::array* conv_w = nullptr;
  if (use_conv) {
    conv_w = &args[i++];
  }
  const auto& w_dt_low = args[i++];
  const auto& w_dt_high = args[i++];
  const auto& w_lambda_low = args[i++];
  const auto& w_lambda_high = args[i++];
  const auto& w_theta_low = args[i++];
  const auto& w_theta_high = args[i++];
  const auto& w_b = args[i++];
  const auto& w_c = args[i++];
  const auto& b_norm_scale = args[i++];
  const auto& c_norm_scale = args[i++];
  const auto& b_bias = args[i++];
  const auto& c_bias = args[i++];
  const auto& a_log = args[i++];
  const auto& dt_bias = args[i++];
  const auto& post_norm = args[i++];
  const auto& w_gate = args[i++];
  const auto& w_out = args[i++];

  const int inner = static_cast<int>(w_x.shape(1));
  const int state_size = static_cast<int>(a_log.shape(1));
  const int n_groups = static_cast<int>(w_b.shape(1)) / state_size;
  const int conv_kernel = use_conv ? static_cast<int>(conv_w->shape(1)) : 0;
  const int dt_rank = static_cast<int>(w_dt_low.shape(1));
  const int group_state = n_groups * state_size;
  auto x_norm = rmsnorm_forward(x, pre_norm, eps);
  auto x_proj = mx::matmul(x_norm, w_x);
  auto x_branch = use_conv
      ? causal_depthwise_conv1d(x_proj, *conv_w, B, T, inner, conv_kernel)
      : x_proj;

  auto aux_weight = mx::concatenate({w_dt_low, w_lambda_low, w_theta_low, w_b, w_c}, 1);
  auto aux_proj = mx::matmul(x_branch, aux_weight);
  auto dt_low = mx::slice(aux_proj, {0, 0}, {B * T, dt_rank});
  auto dt = mx::matmul(dt_low, w_dt_high) + dt_bias;
  auto lambda_low = mx::slice(aux_proj, {0, dt_rank}, {B * T, dt_rank * 2});
  auto lambda = mx::matmul(lambda_low, w_lambda_high);
  auto theta_low = mx::slice(aux_proj, {0, dt_rank * 2}, {B * T, dt_rank * 3});
  auto theta = mx::matmul(theta_low, w_theta_high);
  auto b_proj = mx::slice(aux_proj, {0, dt_rank * 3}, {B * T, dt_rank * 3 + group_state});
  auto b_proj_group = mx::reshape(b_proj, {B * T * n_groups, state_size});
  auto b_norm = rmsnorm_forward(b_proj_group, b_norm_scale, eps);
  auto b_biased = mx::reshape(b_norm, {B * T, n_groups * state_size}) + b_bias;
  auto c_proj = mx::slice(aux_proj, {0, dt_rank * 3 + group_state}, {B * T, dt_rank * 3 + group_state * 2});
  auto c_proj_group = mx::reshape(c_proj, {B * T * n_groups, state_size});
  auto c_norm = rmsnorm_forward(c_proj_group, c_norm_scale, eps);
  auto c_biased = mx::reshape(c_norm, {B * T, n_groups * state_size}) + c_bias;
  auto y = mamba3_selective_scan_canonical_phase6(
      x_branch, dt, lambda, theta, a_log, b_biased, c_biased,
      B, T, inner, state_size, n_groups, scan_chunk_size);
  auto y_norm = rmsnorm_forward(y, post_norm, eps);
  auto z = mx::matmul(x_norm, w_gate);
  auto z_act = z * mx::sigmoid(z);
  auto y_gated = y_norm * z_act;

  auto grad_out = cotangents[0];
  auto grad_y_gated = mx::matmul(grad_out, mx::transpose(w_out, {1, 0}));
  auto grad_w_out = mx::matmul(mx::transpose(y_gated, {1, 0}), grad_out);
  auto grad_y_norm = grad_y_gated * z_act;
  auto grad_z = grad_y_gated * y_norm * silu_derivative(z);
  auto grad_w_gate = mx::matmul(mx::transpose(x_norm, {1, 0}), grad_z);
  auto grad_x_norm = mx::matmul(grad_z, mx::transpose(w_gate, {1, 0}));

  auto y_norm_vjp = rmsnorm_vjp(y, post_norm, grad_y_norm, eps);
  auto scan_grads = mamba3_selective_scan_canonical_phase6_vjp(
      {x_branch, dt, lambda, theta, a_log, b_biased, c_biased},
      {},
      {y_norm_vjp.input},
      B,
      T,
      inner,
      state_size,
      n_groups,
      scan_chunk_size);
  auto grad_x_branch = scan_grads[0];
  auto grad_dt = scan_grads[1];
  auto grad_lambda = scan_grads[2];
  auto grad_theta = scan_grads[3];
  auto grad_a_log = scan_grads[4];

  auto grad_b_bias = mx::reshape(mx::sum(scan_grads[5], 0), b_bias.shape());
  auto b_norm_vjp = rmsnorm_vjp(
      b_proj_group,
      b_norm_scale,
      mx::reshape(scan_grads[5], {B * T * n_groups, state_size}),
      eps);
  auto grad_b_proj = mx::reshape(b_norm_vjp.input, {B * T, n_groups * state_size});
  auto grad_w_b = mx::matmul(mx::transpose(x_branch, {1, 0}), grad_b_proj);
  grad_x_branch = grad_x_branch + mx::matmul(grad_b_proj, mx::transpose(w_b, {1, 0}));

  auto grad_c_bias = mx::reshape(mx::sum(scan_grads[6], 0), c_bias.shape());
  auto c_norm_vjp = rmsnorm_vjp(
      c_proj_group,
      c_norm_scale,
      mx::reshape(scan_grads[6], {B * T * n_groups, state_size}),
      eps);
  auto grad_c_proj = mx::reshape(c_norm_vjp.input, {B * T, n_groups * state_size});
  auto grad_w_c = mx::matmul(mx::transpose(x_branch, {1, 0}), grad_c_proj);
  grad_x_branch = grad_x_branch + mx::matmul(grad_c_proj, mx::transpose(w_c, {1, 0}));

  auto grad_dt_bias = mx::reshape(mx::sum(grad_dt, 0), dt_bias.shape());
  auto grad_w_dt_high = mx::matmul(mx::transpose(dt_low, {1, 0}), grad_dt);
  auto grad_dt_low = mx::matmul(grad_dt, mx::transpose(w_dt_high, {1, 0}));
  auto grad_w_dt_low = mx::matmul(mx::transpose(x_branch, {1, 0}), grad_dt_low);
  grad_x_branch = grad_x_branch + mx::matmul(grad_dt_low, mx::transpose(w_dt_low, {1, 0}));

  auto grad_w_lambda_high = mx::matmul(mx::transpose(lambda_low, {1, 0}), grad_lambda);
  auto grad_lambda_low = mx::matmul(grad_lambda, mx::transpose(w_lambda_high, {1, 0}));
  auto grad_w_lambda_low = mx::matmul(mx::transpose(x_branch, {1, 0}), grad_lambda_low);
  grad_x_branch = grad_x_branch + mx::matmul(grad_lambda_low, mx::transpose(w_lambda_low, {1, 0}));

  auto grad_w_theta_high = mx::matmul(mx::transpose(theta_low, {1, 0}), grad_theta);
  auto grad_theta_low = mx::matmul(grad_theta, mx::transpose(w_theta_high, {1, 0}));
  auto grad_w_theta_low = mx::matmul(mx::transpose(x_branch, {1, 0}), grad_theta_low);
  grad_x_branch = grad_x_branch + mx::matmul(grad_theta_low, mx::transpose(w_theta_low, {1, 0}));

  mx::array grad_x_proj = grad_x_branch;
  mx::array grad_conv_w = mx::array(0.0f, mx::float32);
  if (use_conv) {
    auto conv_vjp = causal_depthwise_conv1d_vjp(
        x_proj,
        *conv_w,
        grad_x_branch,
        B,
        T,
        inner,
        conv_kernel);
    grad_x_proj = conv_vjp.input;
    grad_conv_w = conv_vjp.weight;
  }
  auto grad_w_x = mx::matmul(mx::transpose(x_norm, {1, 0}), grad_x_proj);
  grad_x_norm = grad_x_norm + mx::matmul(grad_x_proj, mx::transpose(w_x, {1, 0}));

  auto pre_norm_vjp = rmsnorm_vjp(x, pre_norm, grad_x_norm, eps);
  auto grad_x = grad_out + pre_norm_vjp.input;

  std::vector<mx::array> out;
  out.reserve(args.size());
  out.push_back(grad_x);
  out.push_back(pre_norm_vjp.scale);
  out.push_back(grad_w_x);
  if (use_conv) {
    out.push_back(grad_conv_w);
  }
  out.push_back(grad_w_dt_low);
  out.push_back(grad_w_dt_high);
  out.push_back(grad_w_lambda_low);
  out.push_back(grad_w_lambda_high);
  out.push_back(grad_w_theta_low);
  out.push_back(grad_w_theta_high);
  out.push_back(grad_w_b);
  out.push_back(grad_w_c);
  out.push_back(b_norm_vjp.scale);
  out.push_back(c_norm_vjp.scale);
  out.push_back(grad_b_bias);
  out.push_back(grad_c_bias);
  out.push_back(grad_a_log);
  out.push_back(grad_dt_bias);
  out.push_back(y_norm_vjp.scale);
  out.push_back(grad_w_gate);
  out.push_back(grad_w_out);
  return out;
}

mx::array mamba3_canonical_block(
    const std::vector<mx::array>& args,
    int B,
    int T,
    bool use_conv,
    int scan_chunk_size,
    float eps) {
  log_mamba3_canonical_block_once();
  auto block = mx::custom_vjp(
      [B, T, use_conv, scan_chunk_size, eps](const std::vector<mx::array>& fn_args) {
        return std::vector<mx::array>{
            mamba3_canonical_block_impl(fn_args, B, T, use_conv, scan_chunk_size, eps)};
      },
      [B, T, use_conv, scan_chunk_size, eps](
          const std::vector<mx::array>& fn_args,
          const std::vector<mx::array>& cotangents,
          const std::vector<mx::array>& outputs) {
        (void)outputs;
        return mamba3_canonical_block_vjp(
            fn_args, cotangents, B, T, use_conv, scan_chunk_size, eps);
      });
  return block(args)[0];
}

mx::array gated_delta_scan_naive(
    const mx::array& q,
    const mx::array& k,
    const mx::array& v,
    const mx::array& beta,
    const mx::array& gate,
    int B,
    int T,
    int H,
    int Dk,
    int Dv) {
  auto state = mx::zeros({B, H, Dk, Dv}, mx::float32);
  auto out = mx::zeros({B, T, H, Dv}, mx::float32);
  for (int t = 0; t < T; ++t) {
    auto qt = mx::reshape(mx::slice(q, {0, t, 0, 0}, {B, t + 1, H, Dk}), {B, H, Dk});
    auto kt = mx::reshape(mx::slice(k, {0, t, 0, 0}, {B, t + 1, H, Dk}), {B, H, Dk});
    auto vt = mx::reshape(mx::slice(v, {0, t, 0, 0}, {B, t + 1, H, Dv}), {B, H, Dv});
    auto betat = mx::reshape(mx::slice(beta, {0, t, 0}, {B, t + 1, H}), {B, H, 1});
    auto gatet = mx::reshape(mx::slice(gate, {0, t, 0}, {B, t + 1, H}), {B, H, 1, 1});

    state = gatet * state;
    auto pred = mx::reshape(
        mx::matmul(mx::reshape(kt, {B, H, 1, Dk}), state),
        {B, H, Dv});
    auto err = vt - pred;
    auto err_scaled = err * betat;
    auto update = mx::reshape(kt, {B, H, Dk, 1}) * mx::reshape(err_scaled, {B, H, 1, Dv});
    state = state + update;

    auto ot = mx::reshape(
        mx::matmul(mx::reshape(qt, {B, H, 1, Dk}), state),
        {B, H, Dv});
    out = mx::slice_update(out, mx::reshape(ot, {B, 1, H, Dv}),
                           mx::Shape{0, t, 0, 0}, mx::Shape{B, t + 1, H, Dv});
  }
  return mx::reshape(out, {B * T * H, Dv});
}

mx::array gated_delta_scan_chunked(
    const mx::array& q_in,
    const mx::array& k_in,
    const mx::array& v_in,
    const mx::array& beta_in,
    const mx::array& gate_in,
    int B,
    int T,
    int H,
    int Dk,
    int Dv,
    int chunk_size) {
  if (chunk_size <= 1) {
    return gated_delta_scan_naive(q_in, k_in, v_in, beta_in, gate_in, B, T, H, Dk, Dv);
  }

  const int pad_len = (chunk_size - (T % chunk_size)) % chunk_size;
  const int T_pad = T + pad_len;
  const int n_chunks = T_pad / chunk_size;

  auto q = as_float32(mx::transpose(q_in, {0, 2, 1, 3}));
  auto k = as_float32(mx::transpose(k_in, {0, 2, 1, 3}));
  auto v = as_float32(mx::transpose(v_in, {0, 2, 1, 3}));
  auto beta = as_float32(mx::transpose(beta_in, {0, 2, 1}));
  auto gate = as_float32(mx::transpose(gate_in, {0, 2, 1}));

  if (pad_len > 0) {
    auto q_pad = mx::zeros({B, H, pad_len, Dk}, mx::float32);
    auto k_pad = mx::zeros({B, H, pad_len, Dk}, mx::float32);
    auto v_pad = mx::zeros({B, H, pad_len, Dv}, mx::float32);
    auto beta_pad = mx::zeros({B, H, pad_len}, mx::float32);
    auto gate_pad = mx::ones({B, H, pad_len}, mx::float32);
    q = mx::concatenate({q, q_pad}, 2);
    k = mx::concatenate({k, k_pad}, 2);
    v = mx::concatenate({v, v_pad}, 2);
    beta = mx::concatenate({beta, beta_pad}, 2);
    gate = mx::concatenate({gate, gate_pad}, 2);
  }

  // The op input is already the multiplicative gate factor in (0, 1].
  // On CUDA, very small factors can underflow to exact zero; re-taking log(0)
  // then creates -inf and later (-inf)-(-inf) NaNs in the chunk recurrence.
  // Floor before log so the chunked path remains finite while preserving
  // "effectively zero" decay for values below float32 significance.
  gate = mx::maximum(gate, mx::array(kGatedDeltaGateFloor, mx::float32));

  // These reshapes only split the strided time axis of the [B,H,T,*] views
  // above into [n_chunks, chunk_size]; they should stay metadata-only unless a
  // caller changes the axis order before entering this kernel.
  q = mx::reshape(q, {B, H, n_chunks, chunk_size, Dk});
  k = mx::reshape(k, {B, H, n_chunks, chunk_size, Dk});
  v = mx::reshape(v, {B, H, n_chunks, chunk_size, Dv});
  beta = mx::reshape(beta, {B, H, n_chunks, chunk_size});
  gate = mx::reshape(gate, {B, H, n_chunks, chunk_size});

  auto v_beta = as_float32(v * mx::expand_dims(beta, -1));
  auto k_beta = as_float32(k * mx::expand_dims(beta, -1));
  auto log_decay = as_float32(mx::cumsum(mx::log(gate), 3));
  auto decay_exp = stable_exp_nonpos(log_decay);

  auto time = mx::astype(mx::arange(chunk_size), mx::int32);
  auto row_idx = mx::expand_dims(time, 1);
  auto col_idx = mx::expand_dims(time, 0);
  auto strict_lower = row_idx > col_idx;
  auto lower_inclusive = row_idx >= col_idx;
  auto eye = mx::astype(row_idx == col_idx, mx::float32);

  auto decay_i = mx::expand_dims(log_decay, 4);
  auto decay_j = mx::expand_dims(log_decay, 3);
  auto decay_delta = stable_exp_nonpos(decay_i - decay_j);
  auto strict_lower_f = mx::astype(mx::expand_dims(mx::expand_dims(mx::expand_dims(strict_lower, 0), 0), 0), mx::float32);
  auto lower_inclusive_f = mx::astype(mx::expand_dims(mx::expand_dims(mx::expand_dims(lower_inclusive, 0), 0), 0), mx::float32);
  auto eye_f = mx::reshape(eye, {1, 1, 1, chunk_size, chunk_size});

  // GatedDeltaScan is used inside compiled value_and_grad graphs. MLX forbids
  // mx::eval() while transformations such as compile/vmap are tracing, so this
  // kernel must not materialize intermediates for timing or graph flushing.
  // The old timing path forced raw_attn/solve/state/out evaluation here, which
  // made OP_GATED_DELTA_SCAN fail under the compiled trainer path.
  constexpr bool timing = false;
  double raw_attn_ms = 0.0;
  double solve_ms = 0.0;
  double post_solve_ms = 0.0;
  double causal_attn_ms = 0.0;
  double chunk_loop_ms = 0.0;

  auto section_start = GatedDeltaTimingClock::now();
  auto raw_attn = as_float32(-mx::matmul(k_beta, mx::transpose(k, {0, 1, 2, 4, 3})) * decay_delta * strict_lower_f);
  if (timing) {
    mx::eval(raw_attn);
    raw_attn_ms = elapsed_ms(section_start, GatedDeltaTimingClock::now());
  }

  section_start = GatedDeltaTimingClock::now();
  auto solve_attn = [&]() -> mx::array {
    const int matrix_count = B * H * n_chunks;
    auto raw_attn_mats = mx::contiguous(mx::reshape(raw_attn, {matrix_count, chunk_size, chunk_size}));
    if (cuda_gpu_available()) {
      auto solve = mlx_ir::solve_strictly_lower_cuda_primitive(
          raw_attn_mats,
          matrix_count,
          chunk_size);
      return mx::reshape(solve, {B, H, n_chunks, chunk_size, chunk_size});
    }
    if (mlx_ir::gated_delta_metal_primitive_available()) {
      auto solve = mlx_ir::solve_strictly_lower_metal_primitive(
          raw_attn_mats,
          matrix_count,
          chunk_size);
      return mx::reshape(solve, {B, H, n_chunks, chunk_size, chunk_size});
    } else {
      // CPU fallback: use the older row-wise triangular solve when no custom
      // GPU primitive is available.
      auto solve_raw = raw_attn;
      for (int i = 1; i < chunk_size; ++i) {
        auto row = mx::slice(solve_raw, {0, 0, 0, i, 0}, {B, H, n_chunks, i + 1, i});
        row = mx::reshape(row, {B, H, n_chunks, i});
        auto prefix = mx::slice(solve_raw, {0, 0, 0, 0, 0}, {B, H, n_chunks, i, i});
        auto correction = as_float32(mx::sum(mx::reshape(row, {B, H, n_chunks, i, 1}) * prefix, 3));
        auto updated = as_float32(row + correction);
        solve_raw = mx::slice_update(
            solve_raw,
            mx::reshape(updated, {B, H, n_chunks, 1, i}),
            mx::Shape{0, 0, 0, i, 0},
            mx::Shape{B, H, n_chunks, i + 1, i});
      }
      return as_float32(solve_raw + eye_f);
    }
  }();
  if (timing) {
    mx::eval(solve_attn);
    solve_ms = elapsed_ms(section_start, GatedDeltaTimingClock::now());
  }

  section_start = GatedDeltaTimingClock::now();
  auto k_cumsum = as_float32(mx::matmul(solve_attn, v_beta));
  auto k_cumdecay = as_float32(mx::matmul(solve_attn, k_beta * mx::expand_dims(decay_exp, -1)));
  if (timing) {
    mx::eval(k_cumsum, k_cumdecay);
    post_solve_ms = elapsed_ms(section_start, GatedDeltaTimingClock::now());
  }

  section_start = GatedDeltaTimingClock::now();
  auto causal_attn = as_float32(mx::matmul(q, mx::transpose(k, {0, 1, 2, 4, 3})) * decay_delta * lower_inclusive_f);
  if (timing) {
    mx::eval(causal_attn);
    causal_attn_ms = elapsed_ms(section_start, GatedDeltaTimingClock::now());
  }

  section_start = GatedDeltaTimingClock::now();
  auto state = mx::zeros({B, H, Dk, Dv}, mx::float32);
  auto out = mx::zeros({B, H, n_chunks, chunk_size, Dv}, mx::float32);
  for (int chunk = 0; chunk < n_chunks; ++chunk) {
    // Each slice keeps a singleton chunk axis, and the reshapes below only
    // squeeze that axis away. This remains a view as long as the chunk packing
    // above keeps [n_chunks, chunk_size] contiguous in logical time order.
    auto q_i = mx::reshape(mx::slice(q, {0, 0, chunk, 0, 0}, {B, H, chunk + 1, chunk_size, Dk}), {B, H, chunk_size, Dk});
    auto k_i = mx::reshape(mx::slice(k, {0, 0, chunk, 0, 0}, {B, H, chunk + 1, chunk_size, Dk}), {B, H, chunk_size, Dk});
    auto v_i = mx::reshape(mx::slice(k_cumsum, {0, 0, chunk, 0, 0}, {B, H, chunk + 1, chunk_size, Dv}), {B, H, chunk_size, Dv});
    auto k_decay_i = mx::reshape(mx::slice(k_cumdecay, {0, 0, chunk, 0, 0}, {B, H, chunk + 1, chunk_size, Dk}), {B, H, chunk_size, Dk});
    auto decay_i_chunk = mx::reshape(mx::slice(log_decay, {0, 0, chunk, 0}, {B, H, chunk + 1, chunk_size}), {B, H, chunk_size});
    auto attn_i = mx::reshape(mx::slice(causal_attn, {0, 0, chunk, 0, 0}, {B, H, chunk + 1, chunk_size, chunk_size}), {B, H, chunk_size, chunk_size});

    auto v_prime = as_float32(mx::matmul(k_decay_i, state));
    auto v_new = as_float32(v_i - v_prime);
    auto decay_chunk_exp = stable_exp_nonpos(decay_i_chunk);
    auto o_inter = as_float32(mx::matmul(q_i * mx::expand_dims(decay_chunk_exp, -1), state));
    auto o_chunk = as_float32(o_inter + mx::matmul(attn_i, v_new));
    out = mx::slice_update(out, mx::reshape(o_chunk, {B, H, 1, chunk_size, Dv}),
                           mx::Shape{0, 0, chunk, 0, 0}, mx::Shape{B, H, chunk + 1, chunk_size, Dv});

    auto decay_last = mx::reshape(mx::slice(decay_i_chunk, {0, 0, chunk_size - 1}, {B, H, chunk_size}), {B, H});
    auto carry = stable_exp_nonpos(mx::expand_dims(decay_last, -1) - decay_i_chunk);
    auto state_update = as_float32(mx::matmul(
        mx::transpose(k_i * mx::expand_dims(carry, -1), {0, 1, 3, 2}),
        v_new));
    state = as_float32(state * mx::reshape(stable_exp_nonpos(decay_last), {B, H, 1, 1}) + state_update);
  }
  if (timing) {
    mx::eval(out);
    chunk_loop_ms = elapsed_ms(section_start, GatedDeltaTimingClock::now());
    record_gated_delta_timing(
        n_chunks,
        raw_attn_ms,
        solve_ms,
        post_solve_ms,
        causal_attn_ms,
        chunk_loop_ms);
  }

  // This merges adjacent [n_chunks, chunk_size] axes back into time, so it is
  // likewise expected to remain metadata-only.
  auto out_seq = mx::reshape(out, {B, H, T_pad, Dv});
  if (pad_len > 0) {
    out_seq = mx::slice(out_seq, {0, 0, 0, 0}, {B, H, T, Dv});
  }
  out_seq = mx::transpose(out_seq, {0, 2, 1, 3});
  return mx::reshape(out_seq, {B * T * H, Dv});
}

mx::array tensor_desc_to_array(const mlx_ir::TensorDesc& desc) {
  mx::Shape shape;
  shape.reserve(desc.shape.size());
  size_t elem_count = 1;
  for (int dim : desc.shape) {
    if (dim < 0) {
      throw std::runtime_error("negative tensor dimension");
    }
    shape.push_back(static_cast<mx::ShapeElem>(dim));
    elem_count *= static_cast<size_t>(dim);
  }
  const size_t elem_bytes = (desc.dtype == mlx_ir::TensorDesc::INT32) ? sizeof(int32_t) : sizeof(float);
  const size_t expected_bytes = elem_count * elem_bytes;
  if (desc.size_bytes < expected_bytes) {
    throw std::runtime_error("tensor data size smaller than shape");
  }
  if (desc.data == nullptr && desc.size_bytes > 0) {
    throw std::runtime_error("tensor data is null");
  }

  if (desc.dtype == mlx_ir::TensorDesc::INT32) {
    if ((desc.size_bytes % sizeof(int32_t)) != 0) {
      throw std::runtime_error("int32 tensor data is not aligned to element size");
    }
    const auto* ptr = static_cast<const int32_t*>(desc.data);
    return mx::array(ptr, shape, mx::int32);
  }
  if (desc.dtype == mlx_ir::TensorDesc::FLOAT32) {
    if ((desc.size_bytes % sizeof(float)) != 0) {
      throw std::runtime_error("float32 tensor data is not aligned to element size");
    }
    const auto* ptr = static_cast<const float*>(desc.data);
    return mx::array(ptr, shape, mx::float32);
  }
  throw std::runtime_error("unsupported tensor dtype");
}

std::vector<int> to_shape_vec(const mx::array& arr) {
  std::vector<int> out;
  out.reserve(static_cast<size_t>(arr.ndim()));
  for (int i = 0; i < arr.ndim(); ++i) {
    out.push_back(static_cast<int>(arr.shape(i)));
  }
  return out;
}

mx::array resolve_output(
    const mlx_ir::IRProgram& program,
    const std::unordered_map<std::string, mx::array>& env,
    const std::string* output_name) {
  if (output_name != nullptr && !output_name->empty()) {
    auto out_it = env.find(*output_name);
    if (out_it == env.end()) {
      throw std::runtime_error("IR program did not produce output: " + *output_name);
    }
    return out_it->second;
  }
  auto loss_it = env.find("loss");
  if (loss_it != env.end()) {
    return loss_it->second;
  }
  if (!program.ops.empty() && program.ops.back().n_outputs > 0) {
    auto tail_it = env.find(program.ops.back().outputs[0]);
    if (tail_it != env.end()) {
      return tail_it->second;
    }
  }
  throw std::runtime_error("IR program did not produce `loss`");
}

std::unordered_map<std::string, mx::array> resolve_outputs(
    const std::unordered_map<std::string, mx::array>& env,
    const std::vector<std::string>& output_names) {
  std::unordered_map<std::string, mx::array> out;
  out.reserve(output_names.size());
  for (const auto& output_name : output_names) {
    auto it = env.find(output_name);
    if (it == env.end()) {
      throw std::runtime_error("IR program did not produce output: " + output_name);
    }
    out.emplace(output_name, it->second);
  }
  return out;
}

std::string inferred_output_name(const mlx_ir::IRProgram& program, const std::string& output_name) {
  if (!output_name.empty()) {
    return output_name;
  }
  for (const auto& op : program.ops) {
    for (int i = 0; i < op.n_outputs; ++i) {
      if (op.outputs[i] == "loss") {
        return "loss";
      }
    }
  }
  if (!program.ops.empty() && program.ops.back().n_outputs > 0) {
    return program.ops.back().outputs[0];
  }
  return "";
}

} // namespace

namespace mlx_ir {

ArrayMap tensor_map_to_arrays(const TensorMap& inputs) {
  ArrayMap out;
  out.reserve(inputs.size());
  for (const auto& kv : inputs) {
    out.emplace(kv.first, tensor_desc_to_array(kv.second));
  }
  return out;
}

void report_gated_delta_timing_summary(const char* phase, int index) {
  if (!gated_delta_timing_enabled()) {
    return;
  }

  GatedDeltaTimingTotals totals;
  {
    std::lock_guard<std::mutex> lock(g_gated_delta_timing_mu);
    totals = g_gated_delta_timing;
    g_gated_delta_timing = GatedDeltaTimingTotals{};
  }
  const double total_ms =
      totals.raw_attn_ms +
      totals.solve_ms +
      totals.post_solve_ms +
      totals.causal_attn_ms +
      totals.chunk_loop_ms;
  std::cout << "[gated_delta_timing] " << (phase ? phase : "phase")
            << "=" << index
            << " calls=" << totals.calls
            << " chunks=" << totals.chunks
            << " raw_attn_ms=" << totals.raw_attn_ms
            << " solve_ms=" << totals.solve_ms
            << " post_solve_ms=" << totals.post_solve_ms
            << " causal_attn_ms=" << totals.causal_attn_ms
            << " chunk_loop_ms=" << totals.chunk_loop_ms
            << " total_ms=" << total_ms
            << std::endl;
}

mx::array ir_interpret(
    const IRProgram& program,
    const std::vector<mx::array>& weights,
    const TensorMap& inputs) {
  return ir_interpret(program, weights, inputs, "");
}

mx::array ir_interpret(
    const IRProgram& program,
    const std::vector<mx::array>& weights,
    const ArrayMap& inputs) {
  return ir_interpret(program, weights, inputs, "");
}

mx::array ir_interpret(
    const IRProgram& program,
    const std::vector<mx::array>& weights,
    const TensorMap& inputs,
    const std::string& output_name) {
  return ir_interpret(program, weights, inputs, output_name, false);
}

mx::array ir_interpret(
    const IRProgram& program,
    const std::vector<mx::array>& weights,
    const TensorMap& inputs,
    const std::string& output_name,
    bool training) {
  return ir_interpret(program, weights, tensor_map_to_arrays(inputs), output_name, training);
}

mx::array ir_interpret(
    const IRProgram& program,
    const std::vector<mx::array>& weights,
    const ArrayMap& inputs,
    const std::string& output_name) {
  return ir_interpret(program, weights, inputs, output_name, false);
}

mx::array ir_interpret(
    const IRProgram& program,
    const std::vector<mx::array>& weights,
    const ArrayMap& inputs,
    const std::string& output_name,
    bool training) {
  const std::vector<std::string> keep_outputs = output_name.empty()
      ? std::vector<std::string>{}
      : std::vector<std::string>{output_name};
  auto outputs = ir_interpret_outputs(program, weights, inputs, keep_outputs, training);
  if (output_name.empty()) {
    auto loss_it = outputs.find("loss");
    if (loss_it != outputs.end()) {
      return loss_it->second;
    }
    if (!program.ops.empty() && program.ops.back().n_outputs > 0) {
      auto tail_it = outputs.find(program.ops.back().outputs[0]);
      if (tail_it != outputs.end()) {
        return tail_it->second;
      }
    }
    throw std::runtime_error("IR program did not produce `loss`");
  }
  return outputs.at(output_name);
}

std::unordered_map<std::string, mx::array> ir_interpret_outputs(
    const IRProgram& program,
    const std::vector<mx::array>& weights,
    const TensorMap& inputs,
    const std::vector<std::string>& output_names) {
  return ir_interpret_outputs(program, weights, inputs, output_names, false);
}

std::unordered_map<std::string, mx::array> ir_interpret_outputs(
    const IRProgram& program,
    const std::vector<mx::array>& weights,
    const ArrayMap& inputs,
    const std::vector<std::string>& output_names) {
  return ir_interpret_outputs(program, weights, inputs, output_names, false);
}

std::unordered_map<std::string, mx::array> ir_interpret_outputs(
    const IRProgram& program,
    const std::vector<mx::array>& weights,
    const TensorMap& inputs,
    const std::vector<std::string>& output_names,
    bool training) {
  return ir_interpret_outputs(program, weights, tensor_map_to_arrays(inputs), output_names, training);
}

std::unordered_map<std::string, mx::array> ir_interpret_outputs(
    const IRProgram& program,
    const std::vector<mx::array>& weights,
    const ArrayMap& inputs,
    const std::vector<std::string>& output_names,
    bool training) {
  std::unordered_map<std::string, mx::array> env;
  env.reserve(program.ops.size() * 2 + static_cast<size_t>(weights.size()) + 8);
  std::unordered_set<std::string> pinned;
  pinned.reserve(static_cast<size_t>(inputs.size()) + weights.size() + 1);
  for (const auto& kv : inputs) {
    env.emplace(kv.first, kv.second);
    pinned.emplace(kv.first);
  }

  for (size_t i = 0; i < weights.size(); ++i) {
    auto name = "w" + std::to_string(i);
    env.emplace(name, weights[i]);
    pinned.emplace(std::move(name));
  }

  if (output_names.empty()) {
    const auto keep_output = inferred_output_name(program, "");
    if (!keep_output.empty()) {
      pinned.emplace(keep_output);
    }
  } else {
    for (const auto& output_name : output_names) {
      if (!output_name.empty()) {
        pinned.emplace(output_name);
      }
    }
  }

  std::unordered_map<std::string, int> remaining_uses;
  remaining_uses.reserve(program.ops.size() * 2 + 8);
  for (const auto& op : program.ops) {
    for (int i = 0; i < op.n_inputs; ++i) {
      if (!op.inputs[i].empty()) {
        remaining_uses[op.inputs[i]]++;
      }
    }
  }

  auto get = [&](const IRop& op, int idx) -> const mx::array& {
    if (idx < 0 || idx >= op.n_inputs) {
      throw std::runtime_error("IR input index out of range");
    }
    auto it = env.find(op.inputs[idx]);
    if (it == env.end()) {
      throw std::runtime_error("IR input not found: " + op.inputs[idx]);
    }
    return it->second;
  };

  auto op_overwrites_name = [&](const IRop& op, const std::string& name) {
    for (int i = 0; i < op.n_outputs; ++i) {
      if (op.outputs[i] == name) {
        return true;
      }
    }
    return false;
  };

  auto set_out = [&](const IRop& op, int idx, mx::array arr) {
    if (idx < 0 || idx >= op.n_outputs) {
      throw std::runtime_error("IR output index out of range");
    }
    auto it = env.find(op.outputs[idx]);
    if (it == env.end()) {
      env.emplace(op.outputs[idx], std::move(arr));
    } else {
      it->second = std::move(arr);
    }
  };

  for (size_t op_idx = 0; op_idx < program.ops.size(); ++op_idx) {
    const auto& op = program.ops[op_idx];
    try {
      switch (op.type) {
      case OP_EMBED: {
        set_out(op, 0, mx::take(get(op, 0), mx::astype(get(op, 1), mx::int32), 0));
        break;
      }
      case OP_MATMUL: {
        set_out(op, 0, mx::matmul(get(op, 0), get(op, 1)));
        break;
      }
      case OP_ADD: {
        set_out(op, 0, get(op, 0) + get(op, 1));
        break;
      }
      case OP_MUL: {
        set_out(op, 0, get(op, 0) * get(op, 1));
        break;
      }
      case OP_SCALAR_MUL: {
        set_out(op, 0, get(op, 0) * op.float_params[0]);
        break;
      }
      case OP_SIGMOID: {
        set_out(op, 0, mx::sigmoid(get(op, 0)));
        break;
      }
      case OP_SILU: {
        auto x = get(op, 0);
        set_out(op, 0, x * mx::sigmoid(x));
        break;
      }
      case OP_GELU: {
        auto x = get(op, 0);
        set_out(op, 0, 0.5f * x * (1.0f + mx::tanh(0.7978845608f * (x + 0.044715f * x * mx::square(x)))));
        break;
      }
      case OP_RELU: {
        set_out(op, 0, mx::maximum(get(op, 0), mx::array(0.0f)));
        break;
      }
      case OP_LEAKY_RELU: {
        auto x = get(op, 0);
        float slope = op.float_params[0];
        set_out(op, 0, mx::where(mx::greater(x, mx::array(0.0f)), x, x * mx::array(slope)));
        break;
      }
      case OP_XSA_PROJECT: {
        auto y = get(op, 0);
        auto v = get(op, 1);
        if (y.ndim() != v.ndim()) {
          throw std::runtime_error("OP_XSA_PROJECT expects y and v to have the same rank");
        }
        for (int i = 0; i < y.ndim(); ++i) {
          if (y.shape(i) != v.shape(i)) {
            throw std::runtime_error("OP_XSA_PROJECT expects y and v to have the same shape");
          }
        }
        auto dot_yv = mx::sum(y * v, -1, true);
        auto dot_vv = mx::sum(v * v, -1, true);
        auto coeff = dot_yv / (dot_vv + 1e-8f);
        set_out(op, 0, y - coeff * v);
        break;
      }
      case OP_TANH: {
        set_out(op, 0, mx::tanh(get(op, 0)));
        break;
      }
      case OP_SOFTMAX: {
        int axis = (op.n_int_params > 0) ? op.int_params[0] : -1;
        set_out(op, 0, mx::softmax(get(op, 0), axis));
        break;
      }
      case OP_RESHAPE: {
        if (op.n_int_params <= 0) {
          throw std::runtime_error("OP_RESHAPE missing shape params");
        }
        set_out(op, 0, mx::reshape(get(op, 0), make_shape(op.int_params, op.n_int_params)));
        break;
      }
      case OP_TRANSPOSE: {
        if (op.n_int_params <= 0) {
          throw std::runtime_error("OP_TRANSPOSE missing axes params");
        }
        std::vector<int> axes;
        axes.reserve(static_cast<size_t>(op.n_int_params));
        for (int i = 0; i < op.n_int_params; ++i) {
          axes.push_back(op.int_params[i]);
        }
        set_out(op, 0, mx::transpose(get(op, 0), axes));
        break;
      }
      case OP_SLICE: {
        // int_params: start, end, stride, axis
        if (op.n_int_params < 4) {
          throw std::runtime_error("OP_SLICE requires 4 int params");
        }
        const auto& x = get(op, 0);
        int axis = op.int_params[3];
        if (axis < 0 || axis >= x.ndim()) {
          throw std::runtime_error("OP_SLICE axis out of range");
        }
        mx::Shape starts;
        mx::Shape ends;
        mx::Shape strides;
        starts.reserve(static_cast<size_t>(x.ndim()));
        ends.reserve(static_cast<size_t>(x.ndim()));
        strides.reserve(static_cast<size_t>(x.ndim()));
        for (int d = 0; d < x.ndim(); ++d) {
          starts.push_back(0);
          ends.push_back(x.shape(d));
          strides.push_back(1);
        }
        starts[axis] = op.int_params[0];
        ends[axis] = op.int_params[1];
        strides[axis] = op.int_params[2];
        set_out(op, 0, mx::slice(x, starts, ends, strides));
        break;
      }
      case OP_CONCAT: {
        int axis = (op.n_int_params > 0) ? op.int_params[0] : 0;
        set_out(op, 0, mx::concatenate({get(op, 0), get(op, 1)}, axis));
        break;
      }
      case OP_CAUSAL_MASK: {
        if (op.n_int_params < 1) {
          throw std::runtime_error("OP_CAUSAL_MASK missing T");
        }
        int T = op.int_params[0];
        int window_size = (op.n_int_params >= 2) ? op.int_params[1] : 0;
        auto scores = get(op, 0);
        auto mask2d = mx::triu(mx::ones({T, T}, mx::bool_), 1);
        if (window_size > 0 && window_size < T) {
          auto pos = mx::astype(mx::arange(T), mx::int32);
          auto query_pos = mx::expand_dims(pos, 1);
          auto key_pos = mx::expand_dims(pos, 0);
          auto too_old = key_pos < (query_pos - mx::array(window_size - 1, mx::int32));
          mask2d = mx::logical_or(mask2d, too_old);
        }
        auto mask = mx::expand_dims(mx::expand_dims(mask2d, 0), 0);
        auto masked = mx::where(mask, mx::full_like(scores, -1e9f), scores);
        set_out(op, 0, masked);
        break;
      }
      case OP_PREFIX_CAUSAL_MASK: {
        if (op.n_int_params < 2) {
          throw std::runtime_error("OP_PREFIX_CAUSAL_MASK requires int params: selfT, prefixT");
        }
        int selfT = op.int_params[0];
        int prefixT = op.int_params[1];
        auto scores = get(op, 0);
        mx::array self_pos = mx::astype(mx::arange(selfT), mx::int32);
        mx::array cross_pos = mx::astype(mx::arange(prefixT), mx::int32);
        bool use_position_index = false;
        if (op.n_inputs >= 3) {
          self_pos = mx::astype(get(op, 1), mx::int32);
          cross_pos = mx::astype(get(op, 2), mx::int32);
          if (self_pos.ndim() != 1 || self_pos.shape(0) != selfT) {
            throw std::runtime_error("OP_PREFIX_CAUSAL_MASK self position index must be rank-1 int32 with length selfT");
          }
          if (cross_pos.ndim() != 1 || cross_pos.shape(0) != prefixT) {
            throw std::runtime_error("OP_PREFIX_CAUSAL_MASK cross position index must be rank-1 int32 with length prefixT");
          }
          use_position_index = true;
        } else if (op.n_int_params >= 6) {
          int selfStart = op.int_params[2];
          int selfStride = op.int_params[3];
          int crossStart = op.int_params[4];
          int crossStride = op.int_params[5];
          auto self_idx = mx::astype(mx::arange(selfT), mx::int32);
          auto cross_idx = mx::astype(mx::arange(prefixT), mx::int32);
          self_pos = self_idx * selfStride + selfStart;
          cross_pos = cross_idx * crossStride + crossStart;
          use_position_index = true;
        }
        mx::array causal_self = mx::triu(mx::ones({selfT, selfT}, mx::bool_), 1);
        mx::array cross_mask = mx::zeros({selfT, prefixT}, mx::bool_);
        if (use_position_index) {
          // Position-indexed causal rule: allow attention iff key_pos <= self_pos.
          // Future keys are masked where key_pos > self_pos.
          causal_self = mx::expand_dims(self_pos, 0) > mx::expand_dims(self_pos, 1);
          cross_mask = mx::expand_dims(cross_pos, 0) > mx::expand_dims(self_pos, 1);
        }
        auto mask2d = mx::concatenate({cross_mask, causal_self}, 1);
        auto mask = mx::expand_dims(mx::expand_dims(mask2d, 0), 0);
        auto masked = mx::where(mask, mx::full_like(scores, -1e9f), scores);
        set_out(op, 0, masked);
        break;
      }
      case OP_CROSS_ENTROPY: {
        set_out(op, 0, cross_entropy_mean(get(op, 0), mx::astype(get(op, 1), mx::int32)));
        break;
      }
      case OP_CROSS_ENTROPY_PER_TOKEN: {
        set_out(op, 0, cross_entropy_per_token(get(op, 0), mx::astype(get(op, 1), mx::int32)));
        break;
      }
      case OP_MASKED_CROSS_ENTROPY: {
        set_out(op, 0, masked_cross_entropy_mean(
            get(op, 0),
            mx::astype(get(op, 1), mx::int32),
            get(op, 2)));
        break;
      }
      case OP_MASKED_CROSS_ENTROPY_PER_TOKEN: {
        set_out(op, 0, masked_cross_entropy_per_token(
            get(op, 0),
            mx::astype(get(op, 1), mx::int32),
            get(op, 2)));
        break;
      }
      case OP_FIRST_BYTE_MASKED_CROSS_ENTROPY: {
        set_out(op, 0, first_byte_masked_cross_entropy_mean(
            get(op, 0),
            mx::astype(get(op, 1), mx::int32),
            get(op, 2)));
        break;
      }
      case OP_DISTILLATION_KL: {
        set_out(op, 0, distillation_kl_mean(get(op, 0), get(op, 1)));
        break;
      }
      case OP_DROPOUT: {
        if (op.n_float_params < 1) {
          throw std::runtime_error("OP_DROPOUT requires rate float param");
        }
        auto x = get(op, 0);
        float rate = op.float_params[0];
        if (rate < 0.0f || rate > 1.0f) {
          throw std::runtime_error("OP_DROPOUT rate must be in [0,1]");
        }
        if (!training || rate == 0.0f) {
          set_out(op, 0, x);
          break;
        }
        if (rate == 1.0f) {
          set_out(op, 0, mx::zeros_like(x));
          break;
        }
        float keep_prob = 1.0f - rate;
        auto mask = mx::astype(mx::random::bernoulli(keep_prob, x.shape()), x.dtype());
        set_out(op, 0, x * mask / keep_prob);
        break;
      }
      case OP_SQUARE: {
        set_out(op, 0, mx::square(get(op, 0)));
        break;
      }
      case OP_SUB: {
        set_out(op, 0, get(op, 0) - get(op, 1));
        break;
      }
      case OP_DIV: {
        auto denom = get(op, 1);
        if (op.n_float_params > 0) {
          denom = denom + op.float_params[0];
        }
        set_out(op, 0, get(op, 0) / denom);
        break;
      }
      case OP_CUMSUM: {
        int axis = (op.n_int_params > 0) ? op.int_params[0] : 0;
        set_out(op, 0, mx::cumsum(get(op, 0), axis));
        break;
      }
      case OP_ARGSORT: {
        int axis = (op.n_int_params > 0) ? op.int_params[0] : -1;
        set_out(op, 0, mx::argsort(get(op, 0), axis));
        break;
      }
      case OP_WHERE: {
        set_out(op, 0, mx::where(get(op, 0), get(op, 1), get(op, 2)));
        break;
      }
      case OP_LESS_THAN: {
        if (op.n_float_params < 1) {
          throw std::runtime_error("OP_LESS_THAN requires scalar");
        }
        set_out(op, 0, get(op, 0) < mx::array(op.float_params[0], mx::float32));
        break;
      }
      case OP_GREATER_EQ: {
        set_out(op, 0, get(op, 0) >= get(op, 1));
        break;
      }
      case OP_ARANGE: {
        if (op.n_int_params < 2) {
          throw std::runtime_error("OP_ARANGE requires start,end");
        }
        set_out(op, 0, mx::arange(op.int_params[0], op.int_params[1]));
        break;
      }
      case OP_MEAN_AXIS: {
        if (op.n_int_params < 1) {
          throw std::runtime_error("OP_MEAN_AXIS requires axis");
        }
        set_out(op, 0, mx::mean(get(op, 0), op.int_params[0]));
        break;
      }
      case OP_FULL: {
        if (op.n_int_params <= 0 || op.n_float_params < 1) {
          throw std::runtime_error("OP_FULL requires shape and value");
        }
        set_out(op, 0, mx::full(make_shape(op.int_params, op.n_int_params), op.float_params[0], mx::float32));
        break;
      }
      case OP_RANDOM_NORMAL: {
        if (op.n_int_params <= 0 || op.n_float_params < 2) {
          throw std::runtime_error("OP_RANDOM_NORMAL requires shape and (mean, stddev)");
        }
        // Fresh i.i.d. Gaussian per call. Wrapped in stop_gradient so the
        // autograd engine treats the sample as a constant input; there is no
        // useful derivative w.r.t. the sampling distribution at the IR level.
        set_out(op, 0, mx::stop_gradient(mx::random::normal(
                           make_shape(op.int_params, op.n_int_params),
                           mx::float32,
                           op.float_params[0],
                           op.float_params[1])));
        break;
      }
      case OP_STOP_GRADIENT: {
        set_out(op, 0, mx::stop_gradient(get(op, 0)));
        break;
      }
      case OP_ASTYPE: {
        set_out(op, 0, mx::astype(get(op, 0), mx::float32));
        break;
      }
      case OP_RUNNING_VAR: {
        if (op.n_int_params < 3) {
          throw std::runtime_error("OP_RUNNING_VAR requires B,T,D");
        }
        float alpha = (op.n_float_params > 0) ? op.float_params[0] : 0.1f;
        set_out(op, 0, running_variance_raw(get(op, 0), op.int_params[0], op.int_params[1], op.int_params[2], alpha));
        break;
      }
      case OP_SCAN: {
        if (op.n_int_params < 3) {
          throw std::runtime_error("OP_SCAN requires B,T,D");
        }
        int B = op.int_params[0];
        int T = op.int_params[1];
        int D = op.int_params[2];
        auto x = mx::reshape(get(op, 0), {B, T, D});
        auto decay_raw = get(op, 1);
        auto decay = mx::reshape(decay_raw, {static_cast<mx::ShapeElem>(decay_raw.size())});
        auto gate = mx::sigmoid(decay);
        auto keep = 1.0f - gate;

        if (T <= 32) {
          // Sequential scan for small T — FFT overhead not worthwhile.
          auto h = mx::zeros({B, D}, mx::float32);
          auto out = mx::zeros({B, T, D}, mx::float32);
          for (int t = 0; t < T; ++t) {
            auto xt = mx::reshape(mx::slice(x, {0, t, 0}, {B, t + 1, D}), {B, D});
            h = gate * h + keep * xt;
            out = mx::slice_update(out, mx::reshape(h, {B, 1, D}), mx::Shape{0, t, 0}, mx::Shape{B, t + 1, D});
          }
          set_out(op, 0, mx::reshape(out, {B * T, D}));
        } else {
          // FFT-based parallel scan. The gate is constant across T (shape [D]),
          // so the recurrence h[t] = gate*h[t-1] + (1-gate)*x[t] is a causal
          // convolution with exponential kernel k[τ] = (1-gate) * gate^τ.
          // Computed in O(T log T) via a single FFT/IFFT pair instead of T
          // sequential kernel launches.

          // Clamp gate away from 0 to avoid log(0) = -inf → NaN in the
          // kernel build (0 * -inf is undefined). 1e-7 is well below float32
          // precision; sigmoid only saturates this hard for |decay| > 16.
          auto gate_safe = mx::maximum(gate, mx::array(1e-7f));

          // Build exponential decay kernel: k[t,d] = keep[d] * gate[d]^t
          // via exp(t * log(gate)) for numerical stability at large t.
          auto t_range = mx::astype(mx::arange(T), mx::float32);
          auto log_gate = mx::log(gate_safe);
          auto t_col = mx::reshape(t_range, {T, 1});
          auto log_gate_row = mx::reshape(log_gate, {1, D});
          auto exponents = t_col * log_gate_row;
          auto kernel = mx::reshape(keep, {1, D}) * mx::exp(exponents);

          // Pad to next power of 2 >= 2T for linear (non-circular) convolution.
          // rfft handles zero-padding internally via the n parameter.
          int fft_len = 1;
          while (fft_len < 2 * T) fft_len <<= 1;

          // Forward FFT along the time axis (rfft pads/truncates to fft_len).
          auto X_freq = mx::fft::rfft(x, fft_len, 1);
          auto K_freq = mx::fft::rfft(kernel, fft_len, 0);

          // Broadcast-multiply in frequency domain (complex × complex).
          auto K_broad = mx::reshape(K_freq, {1, K_freq.shape(0), D});
          auto H_freq = X_freq * K_broad;

          // Inverse FFT back to time domain, take first T steps.
          auto h_full = mx::fft::irfft(H_freq, fft_len, 1);
          auto out = mx::slice(h_full, {0, 0, 0}, {B, T, D});
          set_out(op, 0, mx::reshape(out, {B * T, D}));
        }
        break;
      }
      case OP_MATRIX_SCAN: {
        if (op.n_int_params < 4) {
          throw std::runtime_error("OP_MATRIX_SCAN requires B,T,Da,Db");
        }
        int B = op.int_params[0];
        int T = op.int_params[1];
        int Da = op.int_params[2];
        int Db = op.int_params[3];
        auto update = mx::reshape(get(op, 0), {B, T, Da, Db});
        auto gate_full = mx::reshape(get(op, 1), {B, T, Da});
        auto H = mx::zeros({B, Da, Db}, mx::float32);
        auto out = mx::zeros({B, T, Da, Db}, mx::float32);
        for (int t = 0; t < T; ++t) {
          auto ut = mx::reshape(mx::slice(update, {0, t, 0, 0}, {B, t + 1, Da, Db}), {B, Da, Db});
          auto gt = mx::reshape(mx::slice(gate_full, {0, t, 0}, {B, t + 1, Da}), {B, Da, 1});
          H = gt * H + ut;
          out = mx::slice_update(out, mx::reshape(H, {B, 1, Da, Db}),
                                 mx::Shape{0, t, 0, 0}, mx::Shape{B, t + 1, Da, Db});
        }
        set_out(op, 0, mx::reshape(out, {B * T, Da, Db}));
        break;
      }
      case OP_SCAN_TV: {
        if (op.n_int_params < 3) {
          throw std::runtime_error("OP_SCAN_TV requires B,T,D");
        }
        int B = op.int_params[0];
        int T = op.int_params[1];
        int D = op.int_params[2];
        auto x = mx::reshape(get(op, 0), {B, T, D});
        auto gate_full = mx::reshape(get(op, 1), {B, T, D});
        auto h = mx::zeros({B, D}, mx::float32);
        auto out = mx::zeros({B, T, D}, mx::float32);
        for (int t = 0; t < T; ++t) {
          auto xt = mx::reshape(mx::slice(x, {0, t, 0}, {B, t + 1, D}), {B, D});
          auto gt = mx::reshape(mx::slice(gate_full, {0, t, 0}, {B, t + 1, D}), {B, D});
          h = gt * h + (1.0f - gt) * xt;
          out = mx::slice_update(out, mx::reshape(h, {B, 1, D}), mx::Shape{0, t, 0}, mx::Shape{B, t + 1, D});
        }
        set_out(op, 0, mx::reshape(out, {B * T, D}));
        break;
      }
      case OP_GATED_DELTA_SCAN: {
        if (op.n_int_params < 5) {
          throw std::runtime_error("OP_GATED_DELTA_SCAN requires B,T,H,Dk,Dv");
        }
        int B = op.int_params[0];
        int T = op.int_params[1];
        int H = op.int_params[2];
        int Dk = op.int_params[3];
        int Dv = op.int_params[4];
        auto q = mx::reshape(get(op, 0), {B, T, H, Dk});
        auto k = mx::reshape(get(op, 1), {B, T, H, Dk});
        auto v = mx::reshape(get(op, 2), {B, T, H, Dv});
        auto beta = mx::reshape(get(op, 3), {B, T, H});
        auto gate = mx::reshape(get(op, 4), {B, T, H});
        int chunk_size = (op.n_int_params >= 6) ? op.int_params[5] : 0;
        if (chunk_size <= 0) {
          set_out(op, 0, gated_delta_scan_naive(q, k, v, beta, gate, B, T, H, Dk, Dv));
        } else {
          set_out(op, 0, gated_delta_scan_chunked(q, k, v, beta, gate, B, T, H, Dk, Dv, chunk_size));
        }
        break;
      }
      case OP_DEPTHWISE_CONV1D: {
        if (op.n_int_params < 4) {
          throw std::runtime_error("OP_DEPTHWISE_CONV1D requires B,T,D,K");
        }
        int B = op.int_params[0];
        int T = op.int_params[1];
        int D = op.int_params[2];
        int K = op.int_params[3];
        set_out(op, 0, causal_depthwise_conv1d(get(op, 0), get(op, 1), B, T, D, K));
        break;
      }
      case OP_MAMBA3_SELECTIVE_SCAN: {
        if (op.n_inputs < 7 || op.n_int_params < 5) {
          throw std::runtime_error("OP_MAMBA3_SELECTIVE_SCAN requires 7 inputs and B,T,D,N,G params");
        }
        const int B = op.int_params[0];
        const int T = op.int_params[1];
        const int D = op.int_params[2];
        const int N = op.int_params[3];
        const int G = op.int_params[4];
        const int scan_chunk_size = op.n_int_params >= 6 ? op.int_params[5] : 0;
        set_out(
            op,
            0,
            mamba3_selective_scan_canonical_phase6(
                get(op, 0), get(op, 1), get(op, 2), get(op, 3), get(op, 4), get(op, 5), get(op, 6), B, T, D, N, G, scan_chunk_size));
        break;
      }
      case OP_MAMBA3_CANONICAL_BLOCK: {
        if (op.n_int_params < 4) {
          throw std::runtime_error("OP_MAMBA3_CANONICAL_BLOCK requires B,T,use_conv,scan_chunk params");
        }
        const int B = op.int_params[0];
        const int T = op.int_params[1];
        const bool use_conv = op.int_params[2] != 0;
        const int scan_chunk_size = op.int_params[3];
        const float eps = op.n_float_params > 0 ? op.float_params[0] : 1e-5f;
        const int expected_inputs = mamba3_canonical_block_expected_inputs(use_conv);
        if (op.n_inputs != expected_inputs) {
          throw std::runtime_error(
              "OP_MAMBA3_CANONICAL_BLOCK input count mismatch; got " +
              std::to_string(op.n_inputs) + " want " + std::to_string(expected_inputs));
        }
        std::vector<mx::array> block_inputs;
        block_inputs.reserve(static_cast<size_t>(op.n_inputs));
        for (int input_idx = 0; input_idx < op.n_inputs; ++input_idx) {
          block_inputs.push_back(get(op, input_idx));
        }
        set_out(
            op,
            0,
            mamba3_canonical_block(block_inputs, B, T, use_conv, scan_chunk_size, eps));
        break;
      }
      case OP_GATHER_POSITIONS: {
        if (op.n_int_params < 3) {
          throw std::runtime_error("OP_GATHER_POSITIONS requires B,K,D");
        }
        int B = op.int_params[0];
        int K = op.int_params[1];
        int D = op.int_params[2];
        auto x = get(op, 0);
        auto positions = mx::astype(get(op, 1), mx::int32);
        if (x.ndim() != 3 || x.shape(0) != B || x.shape(2) != D) {
          throw std::runtime_error("OP_GATHER_POSITIONS expects input shape [B,T,D]");
        }
        if (positions.ndim() != 1 || positions.shape(0) != K) {
          throw std::runtime_error("OP_GATHER_POSITIONS expects positions shape [K]");
        }
        set_out(op, 0, mx::take(x, positions, 1));
        break;
      }
      case OP_SCATTER_POSITIONS: {
        if (op.n_int_params < 4) {
          throw std::runtime_error("OP_SCATTER_POSITIONS requires B,T,K,D");
        }
        int B = op.int_params[0];
        int T = op.int_params[1];
        int K = op.int_params[2];
        int D = op.int_params[3];
        auto out = get(op, 0);
        auto updates = get(op, 1);
        auto positions = mx::astype(get(op, 2), mx::int32);
        if (out.ndim() != 3 || out.shape(0) != B || out.shape(1) != T || out.shape(2) != D) {
          throw std::runtime_error("OP_SCATTER_POSITIONS expects input shape [B,T,D]");
        }
        if (updates.ndim() != 3 || updates.shape(0) != B || updates.shape(1) != K || updates.shape(2) != D) {
          throw std::runtime_error("OP_SCATTER_POSITIONS expects updates shape [B,K,D]");
        }
        if (positions.ndim() != 1 || positions.shape(0) != K) {
          throw std::runtime_error("OP_SCATTER_POSITIONS expects positions shape [K]");
        }
        for (int k = 0; k < K; ++k) {
          auto pos_scalar = mx::reshape(mx::slice(positions, {k}, {k + 1}), {});
          mx::eval(pos_scalar);
          int t = pos_scalar.item<int>();
          if (t < 0 || t >= T) {
            throw std::runtime_error("OP_SCATTER_POSITIONS position out of range");
          }
        }
        auto scatter_idx = mx::broadcast_to(mx::reshape(positions, {1, K, 1}), {B, K, D});
        set_out(op, 0, mx::put_along_axis(out, scatter_idx, updates, 1));
        break;
      }
      case OP_GRADIENT_MAGNITUDES: {
        auto hidden = mx::stop_gradient(get(op, 0));
        if (hidden.ndim() != 3) {
          throw std::runtime_error("OP_GRADIENT_MAGNITUDES expects input rank-3 [B,T,D]");
        }

        int T = hidden.shape(1);
        auto magnitudes = mx::zeros({T}, mx::float32);
        if (T > 1) {
          auto next = mx::slice(hidden, {0, 1, 0}, {hidden.shape(0), hidden.shape(1), hidden.shape(2)});
          auto prev = mx::slice(hidden, {0, 0, 0}, {hidden.shape(0), hidden.shape(1) - 1, hidden.shape(2)});
          auto diff = mx::subtract(next, prev);
          auto step_magnitudes = mx::mean(mx::sum(mx::multiply(diff, diff), -1), 0);
          auto prefix = mx::zeros({1}, mx::float32);
          magnitudes = mx::concatenate({prefix, step_magnitudes}, 0);
        }
        set_out(op, 0, mx::stop_gradient(magnitudes));
        break;
      }
      case OP_RMSNORM: {
        if (op.n_float_params < 1) {
          throw std::runtime_error("OP_RMSNORM requires eps float param");
        }
        auto x = get(op, 0);
        auto scale = get(op, 1);
        auto ms = mx::mean(mx::square(x), -1, true);
        auto rms_inv = 1.0f / mx::sqrt(ms + op.float_params[0]);
        set_out(op, 0, x * rms_inv * scale);
        break;
      }
      case OP_ROPE: {
        if (op.n_int_params < 2) {
          throw std::runtime_error("OP_ROPE requires int params: T, head_dim");
        }
        int T = op.int_params[0];
        int HD = op.int_params[1];
        if (HD <= 0 || (HD % 2) != 0) {
          throw std::runtime_error("OP_ROPE requires even positive head_dim");
        }
        int rope_dims = (op.n_int_params > 2) ? op.int_params[2] : HD;
        if (rope_dims <= 0 || rope_dims >= HD) {
          rope_dims = HD;
        }
        if ((rope_dims % 2) != 0) {
          throw std::runtime_error("OP_ROPE requires even rope_dims");
        }
        int start = (op.n_int_params > 3) ? op.int_params[3] : 0;
        int stride = (op.n_int_params > 4) ? op.int_params[4] : 1;
        float base = (op.n_float_params > 0) ? op.float_params[0] : 10000.0f;

        auto dim_idx = mx::astype(mx::arange(0, rope_dims / 2), mx::float32);
        auto freqs = mx::exp(dim_idx * static_cast<float>(-std::log(base) * 2.0 / static_cast<double>(rope_dims)));
        auto positions = [&]() -> mx::array {
          if (op.n_inputs > 2 && !op.inputs[2].empty()) {
            auto positions_in = mx::astype(get(op, 2), mx::int32);
            if (positions_in.ndim() != 1 || positions_in.shape(0) != T) {
              throw std::runtime_error("OP_ROPE positions must be rank-1 int32 with length T");
            }
            return mx::astype(positions_in, mx::float32);
          }
          return mx::astype(mx::arange(0, T) * stride + start, mx::float32);
        }();
        auto angles = mx::reshape(positions, {T, 1}) * mx::reshape(freqs, {1, rope_dims / 2});
        auto cos_t = mx::reshape(mx::cos(angles), {1, 1, T, rope_dims / 2});
        auto sin_t = mx::reshape(mx::sin(angles), {1, 1, T, rope_dims / 2});

        auto apply_rope = [&](const mx::array& x) -> mx::array {
          if (x.ndim() != 4 || x.shape(2) != T || x.shape(3) != HD) {
            throw std::runtime_error("OP_ROPE expects q/k shape [B,H,T,head_dim]");
          }
          auto x_rot = mx::slice(x, {0, 0, 0, 0}, {x.shape(0), x.shape(1), x.shape(2), rope_dims});
          auto even = mx::slice(x_rot, {0, 0, 0, 0}, {x_rot.shape(0), x_rot.shape(1), x_rot.shape(2), x_rot.shape(3)}, {1, 1, 1, 2});
          auto odd = mx::slice(x_rot, {0, 0, 0, 1}, {x_rot.shape(0), x_rot.shape(1), x_rot.shape(2), x_rot.shape(3)}, {1, 1, 1, 2});
          auto rot_even = even * cos_t - odd * sin_t;
          auto rot_odd = even * sin_t + odd * cos_t;
          auto rotated = mx::reshape(mx::stack({rot_even, rot_odd}, 4), x_rot.shape());
          if (rope_dims == HD) {
            return rotated;
          }
          auto pass = mx::slice(x, {0, 0, 0, rope_dims}, {x.shape(0), x.shape(1), x.shape(2), x.shape(3)});
          return mx::concatenate({rotated, pass}, 3);
        };

        set_out(op, 0, apply_rope(get(op, 0)));
        set_out(op, 1, apply_rope(get(op, 1)));
        break;
      }
      case OP_ROPE_INDEXED: {
        if (op.n_int_params < 2) {
          throw std::runtime_error("OP_ROPE_INDEXED requires int params: K, head_dim");
        }
        if (op.n_inputs < 3) {
          throw std::runtime_error("OP_ROPE_INDEXED requires q, k, positions inputs");
        }
        int K = op.int_params[0];
        int HD = op.int_params[1];
        if (HD <= 0 || (HD % 2) != 0) {
          throw std::runtime_error("OP_ROPE_INDEXED requires even positive head_dim");
        }
        int rope_dims = (op.n_int_params > 2) ? op.int_params[2] : HD;
        if (rope_dims <= 0 || rope_dims >= HD) {
          rope_dims = HD;
        }
        if ((rope_dims % 2) != 0) {
          throw std::runtime_error("OP_ROPE_INDEXED requires even rope_dims");
        }
        float base = (op.n_float_params > 0) ? op.float_params[0] : 10000.0f;

        auto positions_in = mx::astype(get(op, 2), mx::int32);
        if (positions_in.ndim() != 1 || positions_in.shape(0) != K) {
          throw std::runtime_error("OP_ROPE_INDEXED expects positions shape [K]");
        }
        auto dim_idx = mx::astype(mx::arange(0, rope_dims / 2), mx::float32);
        auto freqs = mx::exp(dim_idx * static_cast<float>(-std::log(base) * 2.0 / static_cast<double>(rope_dims)));
        auto positions = mx::astype(positions_in, mx::float32);
        auto angles = mx::reshape(positions, {K, 1}) * mx::reshape(freqs, {1, rope_dims / 2});
        auto cos_t = mx::reshape(mx::cos(angles), {1, 1, K, rope_dims / 2});
        auto sin_t = mx::reshape(mx::sin(angles), {1, 1, K, rope_dims / 2});

        auto apply_rope = [&](const mx::array& x) -> mx::array {
          if (x.ndim() != 4 || x.shape(2) != K || x.shape(3) != HD) {
            throw std::runtime_error("OP_ROPE_INDEXED expects q/k shape [B,H,K,head_dim]");
          }
          auto x_rot = mx::slice(x, {0, 0, 0, 0}, {x.shape(0), x.shape(1), x.shape(2), rope_dims});
          auto even = mx::slice(x_rot, {0, 0, 0, 0}, {x_rot.shape(0), x_rot.shape(1), x_rot.shape(2), x_rot.shape(3)}, {1, 1, 1, 2});
          auto odd = mx::slice(x_rot, {0, 0, 0, 1}, {x_rot.shape(0), x_rot.shape(1), x_rot.shape(2), x_rot.shape(3)}, {1, 1, 1, 2});
          auto rot_even = even * cos_t - odd * sin_t;
          auto rot_odd = even * sin_t + odd * cos_t;
          auto rotated = mx::reshape(mx::stack({rot_even, rot_odd}, 4), x_rot.shape());
          if (rope_dims == HD) {
            return rotated;
          }
          auto pass = mx::slice(x, {0, 0, 0, rope_dims}, {x.shape(0), x.shape(1), x.shape(2), x.shape(3)});
          return mx::concatenate({rotated, pass}, 3);
        };

        set_out(op, 0, apply_rope(get(op, 0)));
        set_out(op, 1, apply_rope(get(op, 1)));
        break;
      }
      case OP_ROPE_STRIDED: {
        if (op.n_int_params < 2) {
          throw std::runtime_error("OP_ROPE_STRIDED requires int params: T, head_dim");
        }
        int T = op.int_params[0];
        int HD = op.int_params[1];
        if (HD <= 0 || (HD % 2) != 0) {
          throw std::runtime_error("OP_ROPE_STRIDED requires even positive head_dim");
        }
        int start = (op.n_int_params > 2) ? op.int_params[2] : 0;
        int stride = (op.n_int_params > 3) ? op.int_params[3] : 1;
        float base = (op.n_float_params > 0) ? op.float_params[0] : 10000.0f;

        auto dim_idx = mx::astype(mx::arange(0, HD / 2), mx::float32);
        auto freqs = mx::exp(dim_idx * static_cast<float>(-std::log(base) * 2.0 / static_cast<double>(HD)));
        auto positions = mx::astype(mx::arange(0, T) * stride + start, mx::float32);
        auto angles = mx::reshape(positions, {T, 1}) * mx::reshape(freqs, {1, HD / 2});
        auto cos_t = mx::reshape(mx::cos(angles), {1, 1, T, HD / 2});
        auto sin_t = mx::reshape(mx::sin(angles), {1, 1, T, HD / 2});

        auto apply_rope = [&](const mx::array& x) -> mx::array {
          if (x.ndim() != 4) {
            throw std::runtime_error("OP_ROPE_STRIDED expects rank-4 tensors");
          }
          auto even = mx::slice(x, {0, 0, 0, 0}, {x.shape(0), x.shape(1), x.shape(2), x.shape(3)}, {1, 1, 1, 2});
          auto odd = mx::slice(x, {0, 0, 0, 1}, {x.shape(0), x.shape(1), x.shape(2), x.shape(3)}, {1, 1, 1, 2});
          auto rot_even = even * cos_t - odd * sin_t;
          auto rot_odd = even * sin_t + odd * cos_t;
          return mx::reshape(mx::stack({rot_even, rot_odd}, 4), x.shape());
        };

        set_out(op, 0, apply_rope(get(op, 0)));
        set_out(op, 1, apply_rope(get(op, 1)));
        break;
      }
      case OP_SQRT: {
        set_out(op, 0, mx::sqrt(get(op, 0)));
        break;
      }
      case OP_RSQRT: {
        set_out(op, 0, 1.0f / mx::sqrt(get(op, 0)));
        break;
      }
      case OP_SIN: {
        set_out(op, 0, mx::sin(get(op, 0)));
        break;
      }
      case OP_COS: {
        set_out(op, 0, mx::cos(get(op, 0)));
        break;
      }
      case OP_EXP: {
        set_out(op, 0, mx::exp(get(op, 0)));
        break;
      }
      case OP_SOFTPLUS: {
        auto x = get(op, 0);
        set_out(op, 0, mx::logaddexp(x, mx::zeros_like(x)));
        break;
      }
      case OP_OUTER: {
        // Outer product supports both:
        // - vectors: a [Da], b [Db] -> out [Da, Db]
        // - batched matrices: a [B, Da], b [B, Db] -> out [B, Da, Db]
        auto a = get(op, 0);
        auto b = get(op, 1);
        if (a.ndim() == 1 && b.ndim() == 1) {
          int Da = a.shape(0);
          int Db = b.shape(0);
          auto a_exp = mx::reshape(a, {Da, 1});
          auto b_exp = mx::reshape(b, {1, Db});
          set_out(op, 0, a_exp * b_exp);
          break;
        }
        if (a.ndim() != 2 || b.ndim() != 2) {
          throw std::runtime_error("OP_OUTER requires rank-1 or rank-2 inputs");
        }
        if (a.shape(0) != b.shape(0)) {
          throw std::runtime_error("OP_OUTER batched inputs must share batch dimension");
        }
        int B_dim = a.shape(0);
        int Da = a.shape(1);
        int Db = b.shape(1);
        auto a_exp = mx::reshape(a, {B_dim, Da, 1});
        auto b_exp = mx::reshape(b, {B_dim, 1, Db});
        set_out(op, 0, a_exp * b_exp);
        break;
      }
      case OP_SQUEEZE: {
        if (op.n_int_params < 1) {
          throw std::runtime_error("OP_SQUEEZE requires axis int param");
        }
        auto x = get(op, 0);
        int axis = op.int_params[0];
        if (axis < 0) {
          axis += x.ndim();
        }
        if (axis < 0 || axis >= x.ndim()) {
          throw std::runtime_error("OP_SQUEEZE axis out of range");
        }
        if (x.shape(axis) != 1) {
          throw std::runtime_error("OP_SQUEEZE axis dimension must be 1");
        }
        mx::Shape shape;
        shape.reserve(static_cast<size_t>(x.ndim() - 1));
        for (int i = 0; i < x.ndim(); ++i) {
          if (i != axis) {
            shape.push_back(x.shape(i));
          }
        }
        set_out(op, 0, mx::reshape(x, shape));
        break;
      }
      default:
        throw std::runtime_error("unsupported IR opcode: " + std::to_string(op.type));
      }

      for (int i = 0; i < op.n_inputs; ++i) {
        const auto& name = op.inputs[i];
        if (name.empty() || pinned.find(name) != pinned.end() || op_overwrites_name(op, name)) {
          continue;
        }
        auto it = remaining_uses.find(name);
        if (it == remaining_uses.end()) {
          continue;
        }
        it->second--;
        if (it->second <= 0) {
          env.erase(name);
          remaining_uses.erase(it);
        }
      }

      for (int i = 0; i < op.n_outputs; ++i) {
        const auto& name = op.outputs[i];
        if (name.empty() || pinned.find(name) != pinned.end()) {
          continue;
        }
        auto it = remaining_uses.find(name);
        if (it == remaining_uses.end() || it->second <= 0) {
          env.erase(name);
        }
      }
    } catch (const std::exception& e) {
      throw std::runtime_error(
          "IR op #" + std::to_string(op_idx) + " type=" + std::to_string(op.type) +
          " failed: " + e.what());
    }
  }
  if (output_names.empty()) {
    return {{"loss", resolve_output(program, env, nullptr)}};
  }
  return resolve_outputs(env, output_names);
}

mx::array ir_interpret(
    const IRProgram& program,
    const std::vector<mx::array>& weights,
    const mx::array& tokens,
    const mx::array& targets) {
  return ir_interpret(program, weights, tokens, targets, false);
}

mx::array ir_interpret(
    const IRProgram& program,
    const std::vector<mx::array>& weights,
    const mx::array& tokens,
    const mx::array& targets,
    bool training) {
  auto tokens_i32 = mx::astype(tokens, mx::int32);
  auto targets_i32 = mx::astype(targets, mx::int32);
  mx::eval(tokens_i32, targets_i32);

  std::vector<int32_t> tokens_host(tokens_i32.size());
  std::vector<int32_t> targets_host(targets_i32.size());
  std::memcpy(tokens_host.data(), tokens_i32.data<int32_t>(), tokens_host.size() * sizeof(int32_t));
  std::memcpy(targets_host.data(), targets_i32.data<int32_t>(), targets_host.size() * sizeof(int32_t));

  TensorMap inputs;
  inputs.reserve(2);
  inputs.emplace("tokens", TensorDesc{
                             TensorDesc::INT32,
                             to_shape_vec(tokens_i32),
                             tokens_host.data(),
                             tokens_host.size() * sizeof(int32_t),
                         });
  inputs.emplace("targets", TensorDesc{
                              TensorDesc::INT32,
                              to_shape_vec(targets_i32),
                              targets_host.data(),
                              targets_host.size() * sizeof(int32_t),
                          });
  return ir_interpret(program, weights, inputs, "", training);
}

} // namespace mlx_ir
