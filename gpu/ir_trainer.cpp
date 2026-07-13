#include "ir_trainer.h"
#include "backward_trace.h"
#include "optimizer_step_guard.h"

#include <mlx/compile.h>
#include <mlx/transforms.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace mx = mlx::core;

namespace mlx_ir {
namespace {

struct StepDebugConfig {
  bool enabled = false;
  int start = 0;
  int end = 0;
  int top_k = 4;
};

struct ArrayDebugStats {
  mx::array l2 = mx::array(0.0f, mx::float32);
  mx::array max_abs = mx::array(0.0f, mx::float32);
  mx::array mean_abs = mx::array(0.0f, mx::float32);
  mx::array nan_count = mx::array(0.0f, mx::float32);
  mx::array inf_count = mx::array(0.0f, mx::float32);
  mx::array nonfinite_count = mx::array(0.0f, mx::float32);
};

int getenv_int(const char* name, int fallback) {
  const char* raw = std::getenv(name);
  if (!raw || raw[0] == '\0') {
    return fallback;
  }
  char* end = nullptr;
  long parsed = std::strtol(raw, &end, 10);
  if (end == raw || (end && *end != '\0')) {
    return fallback;
  }
  if (parsed < std::numeric_limits<int>::min()) {
    return std::numeric_limits<int>::min();
  }
  if (parsed > std::numeric_limits<int>::max()) {
    return std::numeric_limits<int>::max();
  }
  return static_cast<int>(parsed);
}

bool env_truthy(const char* name) {
  const char* raw = std::getenv(name);
  if (!raw || raw[0] == '\0') {
    return false;
  }
  return std::strcmp(raw, "0") != 0 &&
      std::strcmp(raw, "false") != 0 &&
      std::strcmp(raw, "FALSE") != 0;
}

bool env_is_set(const char* name) {
  const char* raw = std::getenv(name);
  return raw != nullptr && raw[0] != '\0';
}

bool program_has_op(const IRProgram& program, int op_type) {
  for (const auto& op : program.ops) {
    if (op.type == op_type) {
      return true;
    }
  }
  return false;
}

bool program_has_canonical_mamba3(const IRProgram& program) {
  return program_has_op(program, OP_MAMBA3_SELECTIVE_SCAN) ||
      program_has_op(program, OP_MAMBA3_CANONICAL_BLOCK);
}

bool program_has_fused_canonical_mamba3_block(const IRProgram& program) {
  return program_has_op(program, OP_MAMBA3_CANONICAL_BLOCK);
}

void log_compile_cache_event(
    const char* kind,
    const char* event,
    int step_count,
    size_t cache_size,
    const std::string& signature) {
  if (!env_truthy("MIXLAB_MLX_COMPILE_LOG")) {
    return;
  }
  std::cerr << "[mlx_ir] compile-cache " << kind
            << " " << event
            << " step=" << step_count
            << " cache_size=" << cache_size
            << " signature_len=" << signature.size()
            << std::endl;
}

bool use_compiled_training_step(const IRProgram& program) {
  if (env_truthy("MIXLAB_FORCE_COMPILED_STEP")) {
    return true;
  }
  if (env_truthy("MIXLAB_DISABLE_COMPILED_STEP")) {
    return false;
  }
  if (program_has_fused_canonical_mamba3_block(program) &&
      !env_truthy("MIXLAB_DISABLE_MAMBA3_COMPILED_STEP")) {
    return true;
  }
  return !program_has_canonical_mamba3(program);
}

bool checkpoint_training_step(const IRProgram& program) {
  if (env_truthy("MIXLAB_DISABLE_TRAINING_CHECKPOINT")) {
    return false;
  }
  return program_has_canonical_mamba3(program);
}

StepDebugConfig step_debug_config() {
  static const StepDebugConfig cfg = []() {
    StepDebugConfig out;
    const int start = getenv_int("MIXLAB_MLX_DEBUG_STEP_START", -1);
    const int end = getenv_int("MIXLAB_MLX_DEBUG_STEP_END", -1);
    if (start >= 0 && end >= start) {
      out.enabled = true;
      out.start = start;
      out.end = end;
      out.top_k = std::max(1, getenv_int("MIXLAB_MLX_DEBUG_TOPK", 4));
    }
    return out;
  }();
  return cfg;
}

bool should_log_step_debug(int step) {
  const auto cfg = step_debug_config();
  return cfg.enabled && step >= cfg.start && step <= cfg.end;
}

bool should_log_mamba3_host_timing(int step) {
  if (!env_truthy("MIXLAB_MAMBA3_HOST_TIMING")) {
    return false;
  }
  if (env_truthy("MIXLAB_DISABLE_MAMBA3_HOST_TIMING")) {
    return false;
  }
  const int start = std::max(0, getenv_int("MIXLAB_MAMBA3_HOST_TIMING_START", 100));
  const int every = std::max(1, getenv_int("MIXLAB_MAMBA3_HOST_TIMING_EVERY", 100));
  return step >= start && ((step - start) % every) == 0;
}

void validate_fused_mamba3_cuda_primitive_config(const IRProgram& program) {
  if (!program_has_fused_canonical_mamba3_block(program)) {
    return;
  }
  if (!env_truthy("MIXLAB_MAMBA3_DISABLE_CUDA_PRIMITIVE")) {
    return;
  }
  if (env_truthy("MIXLAB_ALLOW_MAMBA3_MLX_SCAN_FALLBACK")) {
    return;
  }
  throw std::runtime_error(
      "MIXLAB_MAMBA3_DISABLE_CUDA_PRIMITIVE=1 is unsupported for fused canonical "
      "Mamba3 training: it disables only the selective-scan CUDA primitive inside "
      "OP_MAMBA3_CANONICAL_BLOCK, not the fused block lowering, and the MLX-composed "
      "scan fallback can produce invalid or oversized CUDA graphs at this scale. "
      "Unset MIXLAB_MAMBA3_DISABLE_CUDA_PRIMITIVE for production, or set "
      "MIXLAB_ALLOW_MAMBA3_MLX_SCAN_FALLBACK=1 for small debug-only fallback runs.");
}

using HostClock = std::chrono::steady_clock;

long long elapsed_us(HostClock::time_point start, HostClock::time_point end) {
  return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

ArrayDebugStats build_array_debug_stats(const mx::array& src) {
  auto x = mx::astype(src, mx::float32);
  auto abs_x = mx::abs(x);
  auto nan_mask = mx::astype(mx::isnan(x), mx::float32);
  auto inf_mask = mx::astype(mx::isinf(x), mx::float32);
  auto nonfinite_mask = mx::astype(mx::logical_not(mx::isfinite(x)), mx::float32);
  return ArrayDebugStats{
      mx::sqrt(mx::sum(mx::square(x))),
      mx::max(abs_x),
      mx::mean(abs_x),
      mx::sum(nan_mask),
      mx::sum(inf_mask),
      mx::sum(nonfinite_mask),
  };
}

void append_array_debug_stats(std::vector<mx::array>& eval_arrays, const ArrayDebugStats& stats) {
  eval_arrays.push_back(stats.l2);
  eval_arrays.push_back(stats.max_abs);
  eval_arrays.push_back(stats.mean_abs);
  eval_arrays.push_back(stats.nan_count);
  eval_arrays.push_back(stats.inf_count);
  eval_arrays.push_back(stats.nonfinite_count);
}

void eval_arrays_with_context(
    const std::vector<mx::array>& arrays,
    const std::string& context) {
  try {
    mx::eval(arrays);
  } catch (const std::exception& e) {
    throw std::runtime_error(context + ": " + e.what());
  }
}

struct WeightDebugSummary {
  int idx = -1;
  float l2 = 0.0f;
  float max_abs = 0.0f;
  float mean_abs = 0.0f;
  float nan_count = 0.0f;
  float inf_count = 0.0f;
  float nonfinite_count = 0.0f;
};

WeightDebugSummary materialize_weight_debug_summary(int idx, const ArrayDebugStats& stats) {
  return WeightDebugSummary{
      idx,
      stats.l2.item<float>(),
      stats.max_abs.item<float>(),
      stats.mean_abs.item<float>(),
      stats.nan_count.item<float>(),
      stats.inf_count.item<float>(),
      stats.nonfinite_count.item<float>(),
  };
}

void log_submit_step_debug(
    const IRTrainer& trainer,
    int step_index,
    const std::vector<mx::array>& grads) {
  if (!should_log_step_debug(step_index)) {
    return;
  }

  std::vector<ArrayDebugStats> grad_stats;
  grad_stats.reserve(grads.size());
  std::vector<ArrayDebugStats> adam_v_stats;
  adam_v_stats.reserve(trainer.adam_v.size());
  std::vector<mx::array> eval_arrays;
  eval_arrays.reserve((grads.size() + trainer.adam_v.size()) * 6);

  for (const auto& grad : grads) {
    grad_stats.push_back(build_array_debug_stats(grad));
    append_array_debug_stats(eval_arrays, grad_stats.back());
  }
  for (size_t i = 0; i < trainer.adam_v.size(); ++i) {
    if (trainer.has_adam_state[i] == 0) {
      continue;
    }
    adam_v_stats.push_back(build_array_debug_stats(trainer.adam_v[i]));
    append_array_debug_stats(eval_arrays, adam_v_stats.back());
  }

  mx::eval(eval_arrays);

  std::vector<WeightDebugSummary> grad_summaries;
  grad_summaries.reserve(grad_stats.size());
  float total_grad_l2_sq = 0.0f;
  float total_grad_nonfinite = 0.0f;
  int grad_bad_weights = 0;
  for (size_t i = 0; i < grad_stats.size(); ++i) {
    auto summary = materialize_weight_debug_summary(static_cast<int>(i), grad_stats[i]);
    total_grad_l2_sq += summary.l2 * summary.l2;
    total_grad_nonfinite += summary.nonfinite_count;
    if (summary.nonfinite_count > 0.0f) {
      grad_bad_weights++;
    }
    grad_summaries.push_back(summary);
  }

  std::vector<WeightDebugSummary> adam_v_summaries;
  adam_v_summaries.reserve(adam_v_stats.size());
  float total_adam_v_nonfinite = 0.0f;
  int adam_v_bad_weights = 0;
  for (size_t i = 0; i < adam_v_stats.size(); ++i) {
    auto summary = materialize_weight_debug_summary(static_cast<int>(i), adam_v_stats[i]);
    total_adam_v_nonfinite += summary.nonfinite_count;
    if (summary.nonfinite_count > 0.0f) {
      adam_v_bad_weights++;
    }
    adam_v_summaries.push_back(summary);
  }

  const auto cfg = step_debug_config();
  const auto rank_by_max_abs = [](const WeightDebugSummary& a, const WeightDebugSummary& b) {
    if (a.nonfinite_count != b.nonfinite_count) {
      return a.nonfinite_count > b.nonfinite_count;
    }
    return a.max_abs > b.max_abs;
  };
  std::sort(grad_summaries.begin(), grad_summaries.end(), rank_by_max_abs);
  std::sort(adam_v_summaries.begin(), adam_v_summaries.end(), rank_by_max_abs);

  std::cerr << "[mlx_ir_debug] step=" << step_index
            << " grad_total_l2=" << std::sqrt(total_grad_l2_sq)
            << " grad_nonfinite=" << total_grad_nonfinite
            << " grad_bad_weights=" << grad_bad_weights
            << " adam_v_nonfinite=" << total_adam_v_nonfinite
            << " adam_v_bad_weights=" << adam_v_bad_weights
            << std::endl;

  const int grad_limit = std::min<int>(cfg.top_k, static_cast<int>(grad_summaries.size()));
  for (int i = 0; i < grad_limit; ++i) {
    const auto& s = grad_summaries[static_cast<size_t>(i)];
    std::cerr << "[mlx_ir_debug] step=" << step_index
              << " grad[" << s.idx << "]"
              << " l2=" << s.l2
              << " max_abs=" << s.max_abs
              << " mean_abs=" << s.mean_abs
              << " nan=" << s.nan_count
              << " inf=" << s.inf_count
              << " nonfinite=" << s.nonfinite_count
              << std::endl;
  }

  const int adam_v_limit = std::min<int>(cfg.top_k, static_cast<int>(adam_v_summaries.size()));
  for (int i = 0; i < adam_v_limit; ++i) {
    const auto& s = adam_v_summaries[static_cast<size_t>(i)];
    std::cerr << "[mlx_ir_debug] step=" << step_index
              << " adam_v[" << s.idx << "]"
              << " l2=" << s.l2
              << " max_abs=" << s.max_abs
              << " mean_abs=" << s.mean_abs
              << " nan=" << s.nan_count
              << " inf=" << s.inf_count
              << " nonfinite=" << s.nonfinite_count
              << std::endl;
  }
}

struct LoraOptimizerState {
  mx::array adam_m = mx::array(0.0f, mx::float32);
  mx::array adam_v = mx::array(0.0f, mx::float32);
  uint8_t has_adam_state = 0;
  mx::array muon_momentum = mx::array(0.0f, mx::float32);
  uint8_t has_muon_state = 0;
  mx::array muon_second_moment = mx::array(0.0f, mx::float32);
  uint8_t has_muon_second_moment_state = 0;
};

struct LoRAAdapterState {
  bool enabled = false;
  WeightOptimizerSpec spec;
  OptimizerGroupConfig group;
  mx::array a = mx::array(0.0f, mx::float32);
  mx::array b = mx::array(0.0f, mx::float32);
  LoraOptimizerState a_state;
  LoraOptimizerState b_state;
};

// Polar Express NS coefficients for 5 iterations.
// Source: You Jiacheng et al., "The Polar Express", arXiv:2505.16932 /
// ICLR 2026 Implementation 1 (public OpenReview PDF, coeffs_list[:5]).
static constexpr float POLAR_EXPRESS_COEFFS[5][3] = {
    {8.28721201814563f, -23.595886519098837f, 17.300387312530933f},
    {4.107059111542203f, -2.9478499167379106f, 0.5448431082926601f},
    {3.9486908534822946f, -2.908902115962949f, 0.5518191394370137f},
    {3.3184196573706015f, -2.488488024314874f, 0.51004894012372f},
    {2.300652019954817f, -1.6689039845747493f, 0.4188073119525673f},
};

// Polar Express NS coefficients for 4 iterations.
// TODO: verify against the public 4-step reference used by downstream PRs.
static constexpr float POLAR_EXPRESS_COEFFS_4[4][3] = {
    {8.205143229901171f, -23.388676146107957f, 17.18176706298294f},
    {4.103027595395544f, -2.943094594729132f, 0.5443447237556095f},
    {3.948067509815022f, -2.908901580537106f, 0.5497770787381114f},
    {2.4869497712926485f, -1.7693988155519828f, 0.42942000953867464f},
};

static mx::array zeropower_via_newtonschulz5(
    const mx::array& grad,
    int steps,
    NewtonSchulzVariant variant = NewtonSchulzVariant::Fixed,
    float eps = 1e-7f) {
  constexpr float fixed_a = 3.4445f;
  constexpr float fixed_b = -4.7750f;
  constexpr float fixed_c = 2.0315f;
  auto x = mx::astype(grad, mx::bfloat16);
  auto norm = mx::sqrt(mx::sum(mx::square(mx::astype(x, mx::float32))));
  x = x / (norm + mx::array(eps, mx::float32));
  bool transposed = grad.shape(0) > grad.shape(1);
  if (transposed) {
    x = mx::transpose(x, {1, 0});
  }
  const bool use_polar_express =
      variant == NewtonSchulzVariant::PolarExpress && (steps == 4 || steps == 5);
  for (int i = 0; i < steps; ++i) {
    float a = fixed_a;
    float b = fixed_b;
    float c = fixed_c;
    if (use_polar_express) {
      const float(*coeffs)[3] = steps == 4 ? POLAR_EXPRESS_COEFFS_4 : POLAR_EXPRESS_COEFFS;
      a = coeffs[i][0];
      b = coeffs[i][1];
      c = coeffs[i][2];
    }
    auto xt = mx::transpose(x, {1, 0});
    auto A = mx::matmul(x, xt);
    auto B = b * A + c * mx::matmul(A, A);
    x = a * x + mx::matmul(B, x);
  }
  if (transposed) {
    x = mx::transpose(x, {1, 0});
  }
  return mx::astype(x, grad.dtype());
}

static mx::array row_l2_normalize(const mx::array& x, float eps = 1e-7f) {
  auto x_f32 = mx::astype(x, mx::float32);
  auto norm = mx::sqrt(mx::sum(mx::square(x_f32), 1, true));
  return mx::astype(x_f32 / (norm + mx::array(eps, mx::float32)), x.dtype());
}

static mx::array init_normuon_second_moment(const mx::array& x) {
  const auto rows = x.shape(0);
  const auto cols = x.shape(1);
  if (rows >= cols) {
    return mx::zeros(mx::Shape{rows, 1}, mx::float32);
  }
  return mx::zeros(mx::Shape{1, cols}, mx::float32);
}

static mx::array normuon_normalize(
    const mx::array& x,
    mx::array& second_moment,
    float beta2,
    float eps = 1e-10f) {
  auto x_f32 = mx::astype(x, mx::float32);
  const int neuron_axis = x.shape(0) >= x.shape(1) ? 1 : 0;
  auto neuron_mean_sq = mx::mean(mx::square(x_f32), neuron_axis, true);
  second_moment = beta2 * second_moment + (1.0f - beta2) * neuron_mean_sq;

  auto scale = mx::array(1.0f, mx::float32) / mx::sqrt(mx::maximum(second_moment, mx::array(eps, mx::float32)));
  auto normalized = x_f32 * scale;
  auto old_norm = mx::sqrt(mx::sum(mx::square(x_f32)));
  auto new_norm = mx::sqrt(mx::sum(mx::square(normalized)));
  normalized = normalized * (old_norm / mx::maximum(new_norm, mx::array(eps, mx::float32)));
  return mx::astype(normalized, x.dtype());
}

mx::array apply_weight_decay(
    const mx::array& w,
    const mx::array& g,
    const OptimizerGroupConfig& group,
    bool decay,
    int step_count,
    float effective_lr) {
  auto weight_decay_term = [&]() -> mx::array {
    auto decay_term = w;
    const int training_step = std::max(0, step_count - 1);
    if (group.cautious_weight_decay && training_step >= group.cautious_weight_decay_activation_step) {
      auto mask = mx::astype(mx::greater(w * g, mx::array(0.0f, mx::float32)), w.dtype());
      decay_term = decay_term * mask;
    }
    return group.weight_decay * decay_term;
  };
  if (!decay || group.weight_decay <= 0.0f) {
    return w;
  }
  return w - effective_lr * weight_decay_term();
}

mx::array weight_decay_update_term(
    const mx::array& w,
    const mx::array& g,
    const OptimizerGroupConfig& group,
    bool decay,
    int step_count) {
  if (!decay || group.weight_decay <= 0.0f) {
    return mx::zeros_like(w);
  }
  auto decay_term = w;
  const int training_step = std::max(0, step_count - 1);
  if (group.cautious_weight_decay && training_step >= group.cautious_weight_decay_activation_step) {
    auto mask = mx::astype(mx::greater(w * g, mx::array(0.0f, mx::float32)), w.dtype());
    decay_term = decay_term * mask;
  }
  return group.weight_decay * decay_term;
}

mx::array lamb_trust_ratio(const mx::array& w, const mx::array& update, float cap) {
  auto w_norm = mx::sqrt(mx::sum(mx::square(mx::astype(w, mx::float32))));
  auto update_norm = mx::sqrt(mx::sum(mx::square(mx::astype(update, mx::float32))));
  auto raw_ratio = w_norm / update_norm;
  if (std::isfinite(cap) && cap > 0.0f) {
    raw_ratio = mx::minimum(raw_ratio, mx::array(cap, mx::float32));
  }
  auto positive_norms = mx::logical_and(
      mx::greater(w_norm, mx::array(0.0f, mx::float32)),
      mx::greater(update_norm, mx::array(0.0f, mx::float32)));
  auto valid_ratio = mx::logical_and(positive_norms, mx::isfinite(raw_ratio));
  return mx::where(valid_ratio, raw_ratio, mx::array(1.0f, mx::float32));
}

using StepForwardFn = std::function<std::vector<mx::array>(const std::vector<mx::array>&)>;
using StepArrayFn = std::function<std::vector<mx::array>(const std::vector<mx::array>&)>;

enum class Mamba3LowMemoryGradientMode {
  MaterializedChunks,
  SingleBackward,
  RecomputeChunks,
};

struct MaterializedGrad {
  mx::Shape shape;
  mx::array flat = mx::array(0.0f, mx::float32);
  std::vector<float> cpu_values;
};

struct MaterializedGradResult {
  std::vector<mx::array> output_values;
  std::vector<MaterializedGrad> grads;
  double norm_sq = 0.0;
};

struct WeightGradChunk {
  size_t start = 0;
  size_t end = 0;
  size_t elements = 0;
};

std::vector<int> weight_argnums_chunk(size_t start, size_t end) {
  std::vector<int> out;
  out.reserve(end - start);
  for (size_t i = start; i < end; ++i) {
    out.push_back(static_cast<int>(i));
  }
  return out;
}

bool use_compiled_mamba3_grad_chunks() {
  if (env_truthy("MIXLAB_DISABLE_MAMBA3_COMPILED_GRAD_CHUNKS")) {
    return false;
  }
  return env_truthy("MIXLAB_MAMBA3_COMPILED_GRAD_CHUNKS");
}

std::string mamba3_grad_chunks_signature(
    const std::string& step_signature,
    const std::vector<WeightGradChunk>& chunks,
    bool checkpoint_grad_forward) {
  std::ostringstream oss;
  oss << step_signature
      << "|mamba3_grad_chunks=1"
      << "|checkpoint=" << (checkpoint_grad_forward ? 1 : 0);
  for (const auto& chunk : chunks) {
    oss << "|chunk=" << chunk.start << "," << chunk.end << "," << chunk.elements;
  }
  return oss.str();
}

StepArrayFn compile_weight_grad_chunk_fn(
    const StepForwardFn& forward_fn,
    size_t start,
    size_t end,
    bool include_values) {
  auto grad_fn = mx::value_and_grad(forward_fn, weight_argnums_chunk(start, end));
  return mx::compile(
      [grad_fn, include_values](const std::vector<mx::array>& fn_args) {
        auto result = grad_fn(fn_args);
        std::vector<mx::array> out;
        out.reserve((include_values ? result.first.size() : 0) + result.second.size());
        if (include_values) {
          out.insert(out.end(), result.first.begin(), result.first.end());
        }
        out.insert(out.end(), result.second.begin(), result.second.end());
        return out;
      },
      false);
}

void reset_compiled_mamba3_grad_chunks_if_needed(
    IRTrainer& trainer,
    const std::string& signature,
    size_t chunk_count) {
  if (trainer.compiled_mamba3_grad_chunks_signature == signature &&
      trainer.compiled_mamba3_grad_chunks.size() == chunk_count) {
    return;
  }
  trainer.compiled_mamba3_grad_chunks.clear();
  trainer.compiled_mamba3_grad_chunks.resize(chunk_count);
  trainer.compiled_mamba3_grad_chunks_signature = signature;
  trainer.compiled_mamba3_grad_chunks_disabled = false;
  trainer.compiled_mamba3_grad_chunks_fallback_logged = false;
}

void disable_compiled_mamba3_grad_chunks(IRTrainer& trainer, const std::exception& e) {
  trainer.compiled_mamba3_grad_chunks_disabled = true;
  for (auto& compiled : trainer.compiled_mamba3_grad_chunks) {
    compiled = StepArrayFn{};
  }
  if (!trainer.compiled_mamba3_grad_chunks_fallback_logged) {
    std::cerr << "[mlx_ir] canonical Mamba3 compiled grad chunk failed ("
              << e.what()
              << "); falling back to uncompiled gpu-resident chunks"
              << " (unset MIXLAB_MAMBA3_COMPILED_GRAD_CHUNKS to skip this path)"
              << std::endl;
    trainer.compiled_mamba3_grad_chunks_fallback_logged = true;
  }
}

std::vector<mx::array> compute_weight_grad_chunk(
    const StepForwardFn& forward_fn,
    const std::vector<mx::array>& args,
    size_t start,
    size_t end,
    const std::string& context) {
  auto grad_fn = mx::value_and_grad(forward_fn, weight_argnums_chunk(start, end));
  try {
    return grad_fn(args).second;
  } catch (const std::exception& e) {
    throw std::runtime_error(context + ": " + e.what());
  }
}

std::pair<std::vector<mx::array>, std::vector<mx::array>> compute_weight_value_grad_chunk(
    const StepForwardFn& forward_fn,
    const std::vector<mx::array>& args,
    size_t start,
    size_t end,
    const std::string& context) {
  auto grad_fn = mx::value_and_grad(forward_fn, weight_argnums_chunk(start, end));
  try {
    return grad_fn(args);
  } catch (const std::exception& e) {
    throw std::runtime_error(context + ": " + e.what());
  }
}

std::vector<WeightGradChunk> build_weight_grad_chunks(
    const std::vector<mx::array>& weights,
    int max_weights,
    int max_elements) {
  if (weights.empty()) {
    return {};
  }
  const size_t max_count = static_cast<size_t>(std::max(1, max_weights));
  const size_t max_elems = static_cast<size_t>(std::max(1, max_elements));
  std::vector<WeightGradChunk> chunks;
  size_t start = 0;
  size_t elements = 0;
  for (size_t i = 0; i < weights.size(); ++i) {
    const size_t weight_elems = weights[i].size();
    const bool count_full = i > start && (i - start) >= max_count;
    const bool elems_full = i > start && elements + weight_elems > max_elems;
    if (count_full || elems_full) {
      chunks.push_back(WeightGradChunk{start, i, elements});
      start = i;
      elements = 0;
    }
    elements += weight_elems;
  }
  chunks.push_back(WeightGradChunk{start, weights.size(), elements});
  return chunks;
}

mx::array materialized_grad_to_array(const MaterializedGrad& grad) {
  if (!grad.cpu_values.empty()) {
    return mx::array(grad.cpu_values.data(), grad.shape, mx::float32);
  }
  return mx::reshape(grad.flat, grad.shape);
}

bool materialize_mamba3_grads_on_cpu() {
  return env_truthy("MIXLAB_MAMBA3_MATERIALIZE_GRADS_ON_CPU");
}

float gradient_clip_scale_from_norm_sq(double total_norm_sq, float max_grad_norm) {
  if (max_grad_norm <= 0.0f) {
    return 1.0f;
  }
  const double total_norm = std::sqrt(total_norm_sq);
  return static_cast<float>(std::min(
      1.0,
      static_cast<double>(max_grad_norm) / (total_norm + 1e-6)));
}

MaterializedGradResult materialize_weight_grad_chunks(
    IRTrainer& trainer,
    const StepForwardFn& forward_fn,
    const std::vector<mx::array>& args,
    size_t n_weights,
    const std::vector<WeightGradChunk>& chunks,
    const std::string& compiled_signature) {
  MaterializedGradResult out;
  out.grads.resize(n_weights);
  const bool materialize_on_cpu = materialize_mamba3_grads_on_cpu();
  const bool compiled_chunks_enabled = use_compiled_mamba3_grad_chunks();
  if (compiled_chunks_enabled) {
    reset_compiled_mamba3_grad_chunks_if_needed(trainer, compiled_signature, chunks.size());
  }
  for (size_t chunk_idx = 0; chunk_idx < chunks.size(); ++chunk_idx) {
    const auto& chunk = chunks[chunk_idx];
    bool chunk_complete = false;
    while (!chunk_complete) {
      std::vector<mx::array> grads;
      bool used_compiled_chunk = false;
      if (compiled_chunks_enabled && !trainer.compiled_mamba3_grad_chunks_disabled) {
        try {
          auto& compiled = trainer.compiled_mamba3_grad_chunks[chunk_idx];
          if (!compiled) {
            compiled = compile_weight_grad_chunk_fn(
                forward_fn,
                chunk.start,
                chunk.end,
                chunk_idx == 0);
          }
          auto packed = compiled(args);
          if (chunk_idx == 0) {
            const size_t grad_count = chunk.end - chunk.start;
            if (packed.size() < grad_count) {
              throw std::runtime_error("compiled first grad chunk returned too few outputs");
            }
            out.output_values.assign(
                packed.begin(),
                packed.begin() + static_cast<std::ptrdiff_t>(packed.size() - grad_count));
            grads.assign(
                packed.begin() + static_cast<std::ptrdiff_t>(out.output_values.size()),
                packed.end());
          } else {
            grads = std::move(packed);
          }
          used_compiled_chunk = true;
        } catch (const std::exception& e) {
          disable_compiled_mamba3_grad_chunks(trainer, e);
        }
      }
      if (!used_compiled_chunk) {
        if (chunk_idx == 0) {
          auto result = compute_weight_value_grad_chunk(
              forward_fn,
              args,
              chunk.start,
              chunk.end,
              "canonical Mamba3 materialized grad chunk [" + std::to_string(chunk.start) +
              "," + std::to_string(chunk.end) + ")");
          out.output_values = std::move(result.first);
          grads = std::move(result.second);
        } else {
          grads = compute_weight_grad_chunk(
              forward_fn,
              args,
              chunk.start,
              chunk.end,
              "canonical Mamba3 materialized grad chunk [" + std::to_string(chunk.start) +
              "," + std::to_string(chunk.end) + ")");
        }
      }
      std::vector<mx::array> flat_grads;
      flat_grads.reserve(grads.size());
      auto partial_norm_sq = mx::array(0.0f, mx::float32);
      for (const auto& grad : grads) {
        auto flat = mx::reshape(
            mx::astype(grad, mx::float32),
            {static_cast<mx::ShapeElem>(grad.size())});
        partial_norm_sq = partial_norm_sq + mx::sum(mx::square(flat));
        flat_grads.push_back(flat);
      }
      std::vector<mx::array> eval_arrays = flat_grads;
      eval_arrays.push_back(partial_norm_sq);
      try {
        eval_arrays_with_context(
            eval_arrays,
            "canonical Mamba3 materialized grad eval chunk [" +
            std::to_string(chunk.start) + "," + std::to_string(chunk.end) + ")");
      } catch (const std::exception& e) {
        if (used_compiled_chunk) {
          disable_compiled_mamba3_grad_chunks(trainer, e);
          continue;
        }
        throw;
      }
      const float chunk_norm_sq = partial_norm_sq.item<float>();
      out.norm_sq += static_cast<double>(chunk_norm_sq);
      for (size_t i = chunk.start; i < chunk.end; ++i) {
        const auto& flat = flat_grads[i - chunk.start];
        auto& saved = out.grads[i];
        saved.shape = grads[i - chunk.start].shape();
        if (materialize_on_cpu) {
          saved.cpu_values.resize(flat.size());
          std::memcpy(
              saved.cpu_values.data(),
              flat.data<float>(),
              saved.cpu_values.size() * sizeof(float));
        } else {
          saved.flat = mx::stop_gradient(flat);
        }
      }
      chunk_complete = true;
    }
  }
  return out;
}

bool is_retriable_cuda_memory_error(const std::exception& e) {
  const std::string message = e.what();
  return message.find("out of memory") != std::string::npos ||
      message.find("cudaMallocAsync") != std::string::npos ||
      message.find("cudaGraphInstantiate") != std::string::npos;
}

std::vector<int> mamba3_grad_chunk_element_candidates(IRTrainer& trainer, int requested) {
  const int floor = 8 * 1024 * 1024;
  const int safe_requested = std::max(1, requested);
  if (env_is_set("MIXLAB_MAMBA3_GRAD_CHUNK_ELEMS")) {
    return {safe_requested};
  }
  if (!env_truthy("MIXLAB_MAMBA3_ADAPTIVE_GRAD_CHUNKS")) {
    return {safe_requested};
  }
  if (trainer.adaptive_mamba3_grad_chunk_elements > 0) {
    return {trainer.adaptive_mamba3_grad_chunk_elements};
  }
  std::vector<int> out{
      32 * 1024 * 1024,
      16 * 1024 * 1024,
      floor,
  };
  out.erase(
      std::remove_if(
          out.begin(),
          out.end(),
          [safe_requested](int candidate) { return candidate < safe_requested; }),
      out.end());
  if (out.empty() || out.back() != safe_requested) {
    out.push_back(safe_requested);
  }
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

void log_mamba3_grad_chunk_fallback_once(
    IRTrainer& trainer,
    int failed_elements,
    int next_elements,
    const std::exception& e) {
  if (trainer.adaptive_mamba3_grad_chunk_fallback_logged) {
    return;
  }
  std::cerr << "[mlx_ir] canonical Mamba3 grad_chunk_elems="
            << failed_elements
            << " failed with CUDA memory pressure ("
            << e.what()
            << "); retrying with grad_chunk_elems="
            << next_elements
            << " (set MIXLAB_MAMBA3_GRAD_CHUNK_ELEMS to pin a value)"
            << std::endl;
  trainer.adaptive_mamba3_grad_chunk_fallback_logged = true;
}

void log_mamba3_single_backward_fallback_once(IRTrainer& trainer, const std::exception& e) {
  if (trainer.mamba3_single_backward_fallback_logged) {
    return;
  }
  std::cerr << "[mlx_ir] canonical Mamba3 single-backward gradient eval failed ("
            << e.what()
            << "); falling back to materialized gradient chunks"
            << " (set MIXLAB_MAMBA3_MATERIALIZED_GRADS=1 to skip single-backward)"
            << std::endl;
  trainer.mamba3_single_backward_fallback_logged = true;
}

void log_fused_mamba3_compiled_step_fallback_once(IRTrainer& trainer, const std::exception& e) {
  if (trainer.fused_mamba3_compiled_step_fallback_logged) {
    return;
  }
  std::cerr << "[mlx_ir] fused canonical Mamba3 compiled training step failed ("
            << e.what()
            << "); falling back to uncompiled low-memory training step"
            << " (set MIXLAB_DISABLE_MAMBA3_COMPILED_STEP=1 to skip compiled retry)"
            << std::endl;
  trainer.fused_mamba3_compiled_step_fallback_logged = true;
}

void log_fused_mamba3_compiled_update_step_fallback_once(IRTrainer& trainer, const std::exception& e) {
  if (trainer.fused_mamba3_compiled_update_step_fallback_logged) {
    return;
  }
  std::cerr << "[mlx_ir] fused canonical Mamba3 compiled AdamW update step failed ("
            << e.what()
            << "); falling back to compiled-gradient training step"
            << " (unset MIXLAB_MAMBA3_COMPILED_UPDATE_STEP to skip full-update compile)"
            << std::endl;
  trainer.fused_mamba3_compiled_update_step_fallback_logged = true;
}

void log_fused_mamba3_compiled_optimizer_update_fallback_once(IRTrainer& trainer, const std::exception& e) {
  if (trainer.fused_mamba3_compiled_optimizer_update_fallback_logged) {
    return;
  }
  std::cerr << "[mlx_ir] fused canonical Mamba3 compiled AdamW optimizer update failed ("
            << e.what()
            << "); falling back to host-built optimizer update"
            << " (set MIXLAB_DISABLE_MAMBA3_COMPILED_OPTIMIZER_UPDATE=1 to skip this path)"
            << std::endl;
  trainer.fused_mamba3_compiled_optimizer_update_fallback_logged = true;
}

bool supports_compiled_fused_mamba3_adamw_update_step(const IRTrainer& trainer) {
  if (!program_has_fused_canonical_mamba3_block(trainer.program) ||
      !env_truthy("MIXLAB_MAMBA3_COMPILED_UPDATE_STEP") ||
      env_truthy("MIXLAB_DISABLE_MAMBA3_COMPILED_UPDATE_STEP") ||
      trainer.fused_mamba3_compiled_update_step_disabled ||
      trainer.qat_mode != QATMode::None ||
      should_log_step_debug(trainer.step_count)) {
    return false;
  }
  if (trainer.weight_optimizers.size() != trainer.weights.size() ||
      trainer.adam_m.size() != trainer.weights.size() ||
      trainer.adam_v.size() != trainer.weights.size() ||
      trainer.has_adam_state.size() != trainer.weights.size()) {
    return false;
  }
  for (const auto& group : trainer.optimizer_groups) {
    if (group.kind != OptimizerKind::AdamW || group.cautious_weight_decay) {
      return false;
    }
  }
  for (size_t i = 0; i < trainer.weights.size(); ++i) {
    if (trainer.has_adam_state[i] == 0 ||
        i >= trainer.weight_optimizers.size() ||
        trainer.weight_optimizers[i].group_index >= trainer.optimizer_groups.size()) {
      return false;
    }
  }
  return true;
}

bool supports_compiled_fused_mamba3_adamw_optimizer_update(const IRTrainer& trainer) {
  if (!program_has_fused_canonical_mamba3_block(trainer.program) ||
      env_truthy("MIXLAB_DISABLE_MAMBA3_COMPILED_OPTIMIZER_UPDATE") ||
      env_truthy("MIXLAB_MAMBA3_COMPILED_UPDATE_STEP") ||
      trainer.fused_mamba3_compiled_optimizer_update_disabled ||
      trainer.qat_mode != QATMode::None ||
      should_log_step_debug(trainer.step_count)) {
    return false;
  }
  if (trainer.weight_optimizers.size() != trainer.weights.size() ||
      trainer.adam_m.size() != trainer.weights.size() ||
      trainer.adam_v.size() != trainer.weights.size() ||
      trainer.has_adam_state.size() != trainer.weights.size()) {
    return false;
  }
  for (const auto& group : trainer.optimizer_groups) {
    if (group.kind != OptimizerKind::AdamW || group.cautious_weight_decay) {
      return false;
    }
  }
  for (size_t i = 0; i < trainer.weights.size(); ++i) {
    if (trainer.has_adam_state[i] == 0 ||
        i >= trainer.weight_optimizers.size() ||
        trainer.weight_optimizers[i].group_index >= trainer.optimizer_groups.size()) {
      return false;
    }
  }
  return true;
}

std::string compiled_adamw_update_step_signature(
    const std::string& step_signature,
    const IRTrainer& trainer,
    size_t input_count) {
  std::ostringstream oss;
  oss << step_signature
      << "|compiled_adamw_update=1"
      << "|inputs=" << input_count
      << "|max_grad_norm=" << trainer.max_grad_norm;
  for (size_t i = 0; i < trainer.weight_optimizers.size(); ++i) {
    const auto& spec = trainer.weight_optimizers[i];
    oss << "|wopt=" << i << "," << spec.group_index << "," << (spec.decay ? 1 : 0);
  }
  for (size_t i = 0; i < trainer.optimizer_groups.size(); ++i) {
    const auto& group = trainer.optimizer_groups[i];
    oss << "|group=" << i
        << "," << static_cast<int>(group.kind)
        << "," << group.lr
        << "," << group.beta1
        << "," << group.beta2
        << "," << group.eps
        << "," << group.weight_decay;
  }
  return oss.str();
}

std::string compiled_adamw_optimizer_update_signature(const IRTrainer& trainer) {
  std::ostringstream oss;
  oss << "compiled_adamw_optimizer_update=1"
      << "|nw=" << trainer.weights.size()
      << "|max_grad_norm=" << trainer.max_grad_norm;
  for (size_t i = 0; i < trainer.weights.size(); ++i) {
    oss << "|shape=" << i;
    for (auto dim : trainer.weights[i].shape()) {
      oss << "," << dim;
    }
  }
  for (size_t i = 0; i < trainer.weight_optimizers.size(); ++i) {
    const auto& spec = trainer.weight_optimizers[i];
    oss << "|wopt=" << i << "," << spec.group_index << "," << (spec.decay ? 1 : 0);
  }
  for (size_t i = 0; i < trainer.optimizer_groups.size(); ++i) {
    const auto& group = trainer.optimizer_groups[i];
    oss << "|group=" << i
        << "," << static_cast<int>(group.kind)
        << "," << group.lr
        << "," << group.beta1
        << "," << group.beta2
        << "," << group.eps
        << "," << group.weight_decay;
  }
  return oss.str();
}

std::vector<mx::array> build_compiled_adamw_update_step_args(
    const IRTrainer& trainer,
    const std::vector<mx::array>& input_arrays) {
  std::vector<mx::array> args;
  const size_t n_weights = trainer.weights.size();
  args.reserve(n_weights * 3 + input_arrays.size() + 1 + trainer.optimizer_groups.size() * 2);
  args.insert(args.end(), trainer.weights.begin(), trainer.weights.end());
  args.insert(args.end(), trainer.adam_m.begin(), trainer.adam_m.end());
  args.insert(args.end(), trainer.adam_v.begin(), trainer.adam_v.end());
  args.insert(args.end(), input_arrays.begin(), input_arrays.end());
  args.push_back(mx::array(trainer.lr_scale, mx::float32));
  const int optimizer_step = trainer.optimizer_step_count + 1;
  for (const auto& group : trainer.optimizer_groups) {
    args.push_back(mx::array(
        1.0f - std::pow(group.beta1, static_cast<float>(optimizer_step)),
        mx::float32));
    args.push_back(mx::array(
        1.0f - std::pow(group.beta2, static_cast<float>(optimizer_step)),
        mx::float32));
  }
  return args;
}

std::vector<mx::array> build_compiled_adamw_optimizer_update_args(
    const IRTrainer& trainer,
    const std::vector<mx::array>& grads) {
  const size_t n_weights = trainer.weights.size();
  if (grads.size() != n_weights) {
    throw std::runtime_error("compiled AdamW optimizer update gradient count mismatch");
  }
  std::vector<mx::array> args;
  args.reserve(n_weights * 4 + 1 + trainer.optimizer_groups.size() * 2);
  args.insert(args.end(), trainer.weights.begin(), trainer.weights.end());
  args.insert(args.end(), trainer.adam_m.begin(), trainer.adam_m.end());
  args.insert(args.end(), trainer.adam_v.begin(), trainer.adam_v.end());
  args.insert(args.end(), grads.begin(), grads.end());
  args.push_back(mx::array(trainer.lr_scale, mx::float32));
  const int optimizer_step = trainer.optimizer_step_count + 1;
  for (const auto& group : trainer.optimizer_groups) {
    args.push_back(mx::array(
        1.0f - std::pow(group.beta1, static_cast<float>(optimizer_step)),
        mx::float32));
    args.push_back(mx::array(
        1.0f - std::pow(group.beta2, static_cast<float>(optimizer_step)),
        mx::float32));
  }
  return args;
}

float gradient_clip_scale_value_from_grads(
    const std::vector<mx::array>& grads,
    float max_grad_norm,
    int chunk_size) {
  if (max_grad_norm <= 0.0f) {
    return 1.0f;
  }
  double total_norm_sq = 0.0;
  const size_t step = static_cast<size_t>(std::max(1, chunk_size));
  for (size_t start = 0; start < grads.size(); start += step) {
    const size_t end = std::min(grads.size(), start + step);
    auto partial_norm_sq = mx::array(0.0f, mx::float32);
    for (size_t i = start; i < end; ++i) {
      partial_norm_sq = partial_norm_sq + mx::sum(mx::square(grads[i]));
    }
    try {
      total_norm_sq += static_cast<double>(partial_norm_sq.item<float>());
    } catch (const std::exception& e) {
      throw std::runtime_error(
          "canonical Mamba3 grad norm chunk [" + std::to_string(start) + "," +
          std::to_string(end) + ") failed: " + e.what());
    }
  }
  return gradient_clip_scale_from_norm_sq(total_norm_sq, max_grad_norm);
}

float gradient_clip_scale_value_chunked_autodiff(
    const StepForwardFn& forward_fn,
    const std::vector<mx::array>& args,
    size_t n_weights,
    float max_grad_norm,
    int chunk_size) {
  if (max_grad_norm <= 0.0f) {
    return 1.0f;
  }
  double total_norm_sq = 0.0;
  const size_t step = static_cast<size_t>(std::max(1, chunk_size));
  for (size_t start = 0; start < n_weights; start += step) {
    const size_t end = std::min(n_weights, start + step);
    auto grads = compute_weight_grad_chunk(
        forward_fn,
        args,
        start,
        end,
        "canonical Mamba3 grad function chunk [" + std::to_string(start) +
        "," + std::to_string(end) + ")");
    auto partial_norm_sq = mx::array(0.0f, mx::float32);
    for (const auto& grad : grads) {
      partial_norm_sq = partial_norm_sq + mx::sum(mx::square(grad));
    }
    try {
      total_norm_sq += static_cast<double>(partial_norm_sq.item<float>());
    } catch (const std::exception& e) {
      throw std::runtime_error(
          "canonical Mamba3 grad norm chunk [" + std::to_string(start) + "," +
          std::to_string(end) + ") failed: " + e.what());
    }
  }
  const double total_norm = std::sqrt(total_norm_sq);
  return static_cast<float>(std::min(
      1.0,
      static_cast<double>(max_grad_norm) / (total_norm + 1e-6)));
}

bool use_low_memory_mamba3_updates(const IRProgram& program) {
  if (env_truthy("MIXLAB_DISABLE_MAMBA3_LOW_MEMORY_UPDATES")) {
    return false;
  }
  return program_has_canonical_mamba3(program);
}

int low_memory_mamba3_update_chunk_size() {
  return std::max(1, getenv_int("MIXLAB_MAMBA3_UPDATE_CHUNK", 64));
}

int low_memory_mamba3_grad_chunk_size() {
  return std::max(1, getenv_int("MIXLAB_MAMBA3_GRAD_CHUNK", 64));
}

int low_memory_mamba3_grad_chunk_elements() {
  return std::max(1, getenv_int("MIXLAB_MAMBA3_GRAD_CHUNK_ELEMS", 8 * 1024 * 1024));
}

bool checkpoint_chunked_mamba3_gradients() {
  return env_truthy("MIXLAB_MAMBA3_CHECKPOINT_CHUNKED_GRADS");
}

Mamba3LowMemoryGradientMode low_memory_mamba3_gradient_mode(const IRProgram& program) {
  if (env_truthy("MIXLAB_MAMBA3_SINGLE_BACKWARD")) {
    return Mamba3LowMemoryGradientMode::SingleBackward;
  }
  if (env_truthy("MIXLAB_MAMBA3_CHUNKED_AUTODIFF") ||
      env_truthy("MIXLAB_MAMBA3_RECOMPUTE_GRADS_FOR_UPDATES")) {
    return Mamba3LowMemoryGradientMode::RecomputeChunks;
  }
  if (env_truthy("MIXLAB_MAMBA3_MATERIALIZED_GRADS")) {
    return Mamba3LowMemoryGradientMode::MaterializedChunks;
  }
  if (program_has_fused_canonical_mamba3_block(program)) {
    return Mamba3LowMemoryGradientMode::SingleBackward;
  }
  return Mamba3LowMemoryGradientMode::MaterializedChunks;
}

bool should_checkpoint_mamba3_grad_forward(
    const IRProgram& program,
    bool checkpoint_step,
    Mamba3LowMemoryGradientMode gradient_mode) {
  if (!checkpoint_step) {
    return false;
  }
  if (gradient_mode == Mamba3LowMemoryGradientMode::SingleBackward) {
    if (env_truthy("MIXLAB_MAMBA3_CHECKPOINT_SINGLE_BACKWARD")) {
      return true;
    }
    return !program_has_fused_canonical_mamba3_block(program);
  }
  return checkpoint_chunked_mamba3_gradients();
}

const char* low_memory_mamba3_gradient_mode_label(Mamba3LowMemoryGradientMode mode) {
  switch (mode) {
    case Mamba3LowMemoryGradientMode::MaterializedChunks:
      return materialize_mamba3_grads_on_cpu() ? "cpu-materialized-chunk" : "gpu-resident-chunk";
    case Mamba3LowMemoryGradientMode::SingleBackward:
      return "single-backward";
    case Mamba3LowMemoryGradientMode::RecomputeChunks:
      return "recomputed-chunk";
  }
  return "unknown";
}

void init_lora_optimizer_state(const mx::array& weight, const OptimizerGroupConfig& group, LoraOptimizerState& state) {
  switch (group.kind) {
    case OptimizerKind::AdamW:
    case OptimizerKind::Lamb:
      state.adam_m = mx::zeros_like(weight);
      state.adam_v = mx::zeros_like(weight);
      state.has_adam_state = 1;
      state.muon_momentum = mx::array(0.0f, mx::float32);
      state.has_muon_state = 0;
      state.muon_second_moment = mx::array(0.0f, mx::float32);
      state.has_muon_second_moment_state = 0;
      break;
    case OptimizerKind::Muon:
      if (weight.ndim() != 2) {
        throw std::runtime_error("Muon LoRA adapters must be rank-2");
      }
      state.adam_m = mx::array(0.0f, mx::float32);
      state.adam_v = mx::array(0.0f, mx::float32);
      state.has_adam_state = 0;
      state.muon_momentum = mx::zeros_like(weight);
      state.has_muon_state = 1;
      if (group.muon_normalization == MuonNormalization::NorMuon) {
        state.muon_second_moment = init_normuon_second_moment(weight);
        state.has_muon_second_moment_state = 1;
      } else {
        state.muon_second_moment = mx::array(0.0f, mx::float32);
        state.has_muon_second_moment_state = 0;
      }
      break;
    default:
      throw std::runtime_error("unsupported optimizer kind");
  }
}

void apply_optimizer_update(
    mx::array& w,
    const mx::array& g,
    const OptimizerGroupConfig& group,
    bool decay,
    int step_count,
    float lr_scale,
    LoraOptimizerState& state) {
  const float effective_lr = group.lr * lr_scale;
  auto grad = mx::astype(g, mx::float32);
  switch (group.kind) {
    case OptimizerKind::AdamW: {
      if (state.has_adam_state == 0) {
        throw std::runtime_error("AdamW state missing for LoRA adapter");
      }
      const float b1t = 1.0f - std::pow(group.beta1, static_cast<float>(step_count));
      const float b2t = 1.0f - std::pow(group.beta2, static_cast<float>(step_count));
      const float one_minus_beta1 = 1.0f - group.beta1;
      const float one_minus_beta2 = 1.0f - group.beta2;
      state.adam_m = group.beta1 * state.adam_m + one_minus_beta1 * grad;
      state.adam_v = group.beta2 * state.adam_v + one_minus_beta2 * mx::square(grad);

      auto mhat = state.adam_m / b1t;
      auto vhat = state.adam_v / b2t;

      w = apply_weight_decay(w, grad, group, decay, step_count, effective_lr);
      w = w - effective_lr * mhat / (mx::sqrt(vhat) + group.eps);
      break;
    }
    case OptimizerKind::Lamb: {
      if (state.has_adam_state == 0) {
        throw std::runtime_error("LAMB state missing for LoRA adapter");
      }
      const float b1t = 1.0f - std::pow(group.beta1, static_cast<float>(step_count));
      const float b2t = 1.0f - std::pow(group.beta2, static_cast<float>(step_count));
      const float one_minus_beta1 = 1.0f - group.beta1;
      const float one_minus_beta2 = 1.0f - group.beta2;
      state.adam_m = group.beta1 * state.adam_m + one_minus_beta1 * grad;
      state.adam_v = group.beta2 * state.adam_v + one_minus_beta2 * mx::square(grad);

      auto mhat = state.adam_m / b1t;
      auto vhat = state.adam_v / b2t;
      auto update = mhat / (mx::sqrt(vhat) + group.eps);
      update = update + weight_decay_update_term(w, grad, group, decay, step_count);
      auto trust_ratio = lamb_trust_ratio(w, update, group.lamb_trust_ratio_cap);
      w = w - effective_lr * trust_ratio * update;
      break;
    }
    case OptimizerKind::Muon: {
      if (state.has_muon_state == 0) {
        throw std::runtime_error("Muon state missing for LoRA adapter");
      }
      if (w.ndim() != 2) {
        throw std::runtime_error("Muon only supports rank-2 LoRA adapters");
      }
      state.muon_momentum = group.beta1 * state.muon_momentum + grad;
      mx::array update = group.nesterov ? (grad + group.beta1 * state.muon_momentum) : state.muon_momentum;
      update = zeropower_via_newtonschulz5(update, group.backend_steps, group.newton_schulz_variant);
      const auto rows = static_cast<float>(w.shape(0));
      const auto cols = static_cast<float>(w.shape(1));
      const float aspect = std::sqrt(std::max(1.0f, rows / cols));
      update = update * mx::array(aspect, mx::float32);
      switch (group.muon_normalization) {
        case MuonNormalization::None:
          break;
        case MuonNormalization::RowL2:
          update = row_l2_normalize(update);
          break;
        case MuonNormalization::NorMuon:
          if (state.has_muon_second_moment_state == 0) {
            throw std::runtime_error("NorMuon state missing for LoRA adapter");
          }
          update = normuon_normalize(update, state.muon_second_moment, group.beta2);
          break;
      }
      w = apply_weight_decay(w, grad, group, decay, step_count, effective_lr);
      w = w - effective_lr * update;
      break;
    }
    default:
      throw std::runtime_error("unsupported optimizer kind");
  }
}

std::vector<mx::array> effective_lora_weights(
    const std::vector<mx::array>& base_weights,
    const std::vector<LoRAAdapterState>& adapters) {
  std::vector<mx::array> effective;
  effective.reserve(base_weights.size());
  for (size_t i = 0; i < base_weights.size(); ++i) {
    if (i < adapters.size() && adapters[i].enabled) {
      effective.push_back(base_weights[i] + mx::matmul(adapters[i].a, adapters[i].b));
      continue;
    }
    effective.push_back(base_weights[i]);
  }
  return effective;
}

mx::array fake_quantize_weight(const mx::array& weight, QATMode mode) {
  if (mode == QATMode::None || weight.ndim() != 2) {
    return weight;
  }
  constexpr float kInt8Max = 127.0f;
  auto w = mx::astype(weight, mx::float32);
  auto abs_max = mx::max(mx::abs(w), 1, true);
  auto scale = mx::maximum(abs_max / kInt8Max, mx::array(1.0f / kInt8Max, mx::float32));
  auto q = mx::round(mx::clip(w / scale, mx::array(-kInt8Max, mx::float32), mx::array(kInt8Max, mx::float32)));
  if (mode == QATMode::Int6) {
    q = mx::round(q / 4.0f) * 4.0f;
    q = mx::clip(q, mx::array(-128.0f, mx::float32), mx::array(124.0f, mx::float32));
  }
  auto w_fake = q * scale;
  return mx::stop_gradient(w_fake - w) + w;
}

std::vector<mx::array> effective_training_weights(
    const std::vector<mx::array>& base_weights,
    QATMode qat_mode,
    ComputeDType compute_dtype) {
  if (qat_mode == QATMode::None && compute_dtype == ComputeDType::Float32) {
    return base_weights;
  }
  std::vector<mx::array> effective;
  effective.reserve(base_weights.size());
  for (const auto& weight : base_weights) {
    auto exec_weight = fake_quantize_weight(weight, qat_mode);
    if (compute_dtype == ComputeDType::BFloat16) {
      exec_weight = mx::astype(exec_weight, mx::bfloat16);
    }
    effective.push_back(exec_weight);
  }
  return effective;
}

std::vector<mx::array> effective_compute_weights(
    const std::vector<mx::array>& base_weights,
    ComputeDType compute_dtype) {
  if (compute_dtype == ComputeDType::Float32) {
    return base_weights;
  }
  std::vector<mx::array> effective;
  effective.reserve(base_weights.size());
  for (const auto& weight : base_weights) {
    effective.push_back(mx::astype(weight, mx::bfloat16));
  }
  return effective;
}

std::vector<mx::array> effective_compute_weights(
    const std::vector<mx::array>& base_weights,
    ComputeDType compute_dtype,
    const std::vector<size_t>& weight_indices) {
  if (compute_dtype == ComputeDType::Float32 || weight_indices.empty()) {
    return base_weights;
  }
  auto effective = base_weights;
  for (size_t idx : weight_indices) {
    if (idx >= effective.size()) {
      throw std::runtime_error("selective compute weight index out of range");
    }
    effective[idx] = mx::astype(effective[idx], mx::bfloat16);
  }
  return effective;
}

void collect_state_for_eval(
    const IRTrainer& trainer,
    std::vector<mx::array>& eval_arrays,
    bool include_cached_outputs) {
  if (include_cached_outputs) {
    auto magnitudes_it = trainer.last_outputs.find("magnitudes");
    if (magnitudes_it != trainer.last_outputs.end()) {
      eval_arrays.push_back(magnitudes_it->second);
    }
    auto hidden_it = trainer.last_outputs.find("x_hidden");
    if (hidden_it != trainer.last_outputs.end()) {
      eval_arrays.push_back(hidden_it->second);
    }
    auto logits_it = trainer.last_outputs.find("logits");
    if (logits_it != trainer.last_outputs.end()) {
      eval_arrays.push_back(logits_it->second);
    }
  }
  for (const auto& w : trainer.weights) {
    eval_arrays.push_back(w);
  }
  for (size_t i = 0; i < trainer.adam_m.size(); ++i) {
    if (trainer.has_adam_state[i] != 0) {
      eval_arrays.push_back(trainer.adam_m[i]);
      eval_arrays.push_back(trainer.adam_v[i]);
    }
  }
  for (size_t i = 0; i < trainer.muon_momentum.size(); ++i) {
    if (trainer.has_muon_state[i] != 0) {
      eval_arrays.push_back(trainer.muon_momentum[i]);
    }
  }
  for (size_t i = 0; i < trainer.muon_second_moment.size(); ++i) {
    if (trainer.has_muon_second_moment_state[i] != 0) {
      eval_arrays.push_back(trainer.muon_second_moment[i]);
    }
  }
  for (size_t i = 0; i < trainer.sgd_momentum.size(); ++i) {
    if (trainer.has_sgd_state[i] != 0) {
      eval_arrays.push_back(trainer.sgd_momentum[i]);
    }
  }
}

void detach_array_vector(std::vector<mx::array>& arrays) {
  for (auto& arr : arrays) {
    arr.detach();
  }
}

void detach_output_map(std::unordered_map<std::string, mx::array>& outputs) {
  for (auto& [_, arr] : outputs) {
    arr.detach();
  }
}

void detach_trainer_state(IRTrainer& trainer) {
  detach_array_vector(trainer.weights);
  for (size_t i = 0; i < trainer.adam_m.size(); ++i) {
    if (trainer.has_adam_state[i] != 0) {
      trainer.adam_m[i].detach();
      trainer.adam_v[i].detach();
    }
  }
  for (size_t i = 0; i < trainer.muon_momentum.size(); ++i) {
    if (trainer.has_muon_state[i] != 0) {
      trainer.muon_momentum[i].detach();
    }
  }
  for (size_t i = 0; i < trainer.muon_second_moment.size(); ++i) {
    if (trainer.has_muon_second_moment_state[i] != 0) {
      trainer.muon_second_moment[i].detach();
    }
  }
  for (size_t i = 0; i < trainer.sgd_momentum.size(); ++i) {
    if (trainer.has_sgd_state[i] != 0) {
      trainer.sgd_momentum[i].detach();
    }
  }
}

bool program_produces_output(const IRProgram& program, const std::string& output_name) {
  for (const auto& op : program.ops) {
    for (int i = 0; i < op.n_outputs; ++i) {
      if (op.outputs[i] == output_name) {
        return true;
      }
    }
  }
  return false;
}

std::string evaluation_loss_name(const IRProgram& program) {
  return program_produces_output(program, "eval_loss") ? "eval_loss" : "loss";
}

std::vector<std::string> collect_cached_output_names(const IRProgram& program, const std::string& loss_name);

std::vector<std::string> collect_cached_output_names(const IRProgram& program) {
  return collect_cached_output_names(program, "loss");
}

std::vector<std::string> collect_cached_output_names(const IRProgram& program, const std::string& loss_name) {
  bool capture_magnitudes = false;
  bool capture_x_hidden = false;
  bool capture_logits = false;
  bool capture_qr = false;
  bool capture_moe_aux = false;
  bool capture_moe_entropy = false;
  for (const auto& op : program.ops) {
    for (int i = 0; i < op.n_outputs; ++i) {
      if (op.outputs[i] == "magnitudes") {
        capture_magnitudes = true;
      }
      if (op.outputs[i] == "x_hidden") {
        capture_x_hidden = true;
      }
      if (op.outputs[i] == "logits") {
        capture_logits = true;
      }
      if (op.outputs[i] == "qr") {
        capture_qr = true;
      }
      if (op.outputs[i] == "moe_aux_loss") {
        capture_moe_aux = true;
      }
      if (op.outputs[i] == "moe_router_entropy") {
        capture_moe_entropy = true;
      }
    }
  }

  std::vector<std::string> output_names{loss_name};
  if (capture_magnitudes) {
    output_names.push_back("magnitudes");
  }
  if (capture_x_hidden) {
    output_names.push_back("x_hidden");
  }
  if (capture_logits) {
    output_names.push_back("logits");
  }
  if (capture_qr) {
    output_names.push_back("qr");
  }
  if (capture_moe_aux) {
    output_names.push_back("moe_aux_loss");
  }
  if (capture_moe_entropy) {
    output_names.push_back("moe_router_entropy");
  }
  return output_names;
}

bool capture_training_outputs() {
  if (env_truthy("MIXLAB_CAPTURE_TRAIN_OUTPUTS") ||
      env_truthy("MIXLAB_MAMBA3_CAPTURE_TRAIN_OUTPUTS")) {
    return true;
  }
  if (env_truthy("MIXLAB_LOSS_ONLY_TRAIN_OUTPUTS") ||
      env_truthy("MIXLAB_MAMBA3_LOSS_ONLY_TRAIN_OUTPUTS")) {
    return false;
  }
  return false;
}

std::vector<std::string> collect_training_step_output_names(const IRTrainer& trainer) {
  std::vector<std::string> output_names = capture_training_outputs()
      ? collect_cached_output_names(trainer.program)
      : std::vector<std::string>{"loss"};
  for (const auto& name : trainer.training_step_extra_output_names) {
    if (name != "loss" &&
        program_produces_output(trainer.program, name) &&
        std::find(output_names.begin(), output_names.end(), name) == output_names.end()) {
      output_names.push_back(name);
    }
  }
  return output_names;
}

std::vector<std::string> sorted_input_names(const TensorMap& inputs) {
  std::vector<std::string> names;
  names.reserve(inputs.size());
  for (const auto& kv : inputs) {
    names.push_back(kv.first);
  }
  std::sort(names.begin(), names.end());
  return names;
}

std::string named_step_signature(
    const IRProgram& program,
    QATMode qat_mode,
    ComputeDType compute_dtype,
    const TensorMap& inputs,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    size_t n_weights) {
  std::ostringstream oss;
  oss << "nw=" << n_weights
      << "|qat=" << static_cast<int>(qat_mode)
      << "|dtype=" << static_cast<int>(compute_dtype)
      << "|prog_ops=" << program.ops.size()
      << "|prog_weights=" << program.n_weights;
  for (size_t i = 0; i < program.ops.size(); ++i) {
    const auto& op = program.ops[i];
    oss << "|op=" << i << "," << op.type;
    for (int j = 0; j < op.n_inputs; ++j) {
      oss << ",in:" << op.inputs[j];
    }
    for (int j = 0; j < op.n_outputs; ++j) {
      oss << ",out:" << op.outputs[j];
    }
    for (int j = 0; j < op.n_int_params; ++j) {
      oss << ",ip:" << op.int_params[j];
    }
    for (int j = 0; j < op.n_float_params; ++j) {
      oss << ",fp:" << op.float_params[j];
    }
  }
  for (const auto& output_name : output_names) {
    oss << "|out=" << output_name;
  }
  for (const auto& name : input_names) {
    const auto& desc = inputs.at(name);
    oss << "|in=" << name << ";dt=" << static_cast<int>(desc.dtype);
    for (int dim : desc.shape) {
      oss << "," << dim;
    }
  }
  return oss.str();
}

std::string categorical_sampler_signature(
    const IRProgram& program,
    ComputeDType compute_dtype,
    const TensorMap& inputs,
    const std::vector<std::string>& input_names,
    const std::string& output_name,
    int rows,
    int vocab,
    float temperature,
    size_t n_weights) {
  std::ostringstream oss;
  oss << named_step_signature(
             program,
             QATMode::None,
             compute_dtype,
             inputs,
             input_names,
             std::vector<std::string>{output_name},
             n_weights)
      << "|categorical_sampler=1"
      << "|rows=" << rows
      << "|vocab=" << vocab
      << "|temperature=" << std::setprecision(9) << temperature;
  return oss.str();
}

std::optional<size_t> parse_weight_input_name(const std::string& name, size_t n_weights) {
  if (name.size() < 2 || name[0] != 'w') {
    return std::nullopt;
  }
  size_t idx = 0;
  for (size_t i = 1; i < name.size(); ++i) {
    const char c = name[i];
    if (c < '0' || c > '9') {
      return std::nullopt;
    }
    idx = idx * 10 + static_cast<size_t>(c - '0');
    if (idx >= n_weights) {
      return std::nullopt;
    }
  }
  return idx;
}

std::vector<uint8_t> required_ops_for_named_outputs(
    const IRProgram& program,
    const std::vector<std::string>& output_names) {
  std::unordered_set<std::string> required;
  required.reserve(output_names.size() + 16);
  for (const auto& output_name : output_names) {
    if (!output_name.empty()) {
      required.emplace(output_name);
    }
  }

  std::vector<uint8_t> needed(program.ops.size(), 0);
  for (size_t rev = program.ops.size(); rev > 0; --rev) {
    const size_t op_idx = rev - 1;
    const auto& op = program.ops[op_idx];
    bool op_needed = false;
    for (int i = 0; i < op.n_outputs; ++i) {
      if (!op.outputs[i].empty() && required.find(op.outputs[i]) != required.end()) {
        op_needed = true;
        break;
      }
    }
    if (!op_needed) {
      continue;
    }
    needed[op_idx] = 1;
    for (int i = 0; i < op.n_outputs; ++i) {
      if (!op.outputs[i].empty()) {
        required.erase(op.outputs[i]);
      }
    }
    for (int i = 0; i < op.n_inputs; ++i) {
      if (!op.inputs[i].empty()) {
        required.emplace(op.inputs[i]);
      }
    }
  }
  return needed;
}

std::vector<size_t> required_weight_indices_for_outputs(
    const IRProgram& program,
    const std::vector<std::string>& output_names,
    size_t n_weights) {
  const auto needed_ops = required_ops_for_named_outputs(program, output_names);
  std::vector<uint8_t> seen(n_weights, 0);
  std::vector<size_t> out;
  for (size_t op_idx = 0; op_idx < program.ops.size(); ++op_idx) {
    if (!needed_ops.empty() && !needed_ops[op_idx]) {
      continue;
    }
    const auto& op = program.ops[op_idx];
    for (int i = 0; i < op.n_inputs; ++i) {
      auto idx = parse_weight_input_name(op.inputs[i], n_weights);
      if (!idx.has_value() || seen[*idx]) {
        continue;
      }
      seen[*idx] = 1;
      out.push_back(*idx);
    }
  }
  return out;
}

mx::array sample_categorical_logits(
    const mx::array& logits,
    int rows,
    int vocab,
    float temperature,
    const mx::array& key) {
  if (logits.ndim() != 2 || logits.shape(0) != rows || logits.shape(1) != vocab) {
    throw std::runtime_error("categorical sampler output shape mismatch");
  }

  auto finite = mx::isfinite(logits);
  auto positive_inf = mx::logical_and(mx::isinf(logits), logits > mx::array(0.0f, mx::float32));
  auto finite_count = mx::sum(mx::astype(finite, mx::float32), 1, true);
  auto positive_inf_count = mx::sum(mx::astype(positive_inf, mx::float32), 1, true);
  auto has_finite = finite_count > mx::array(0.0f, mx::float32);
  auto has_positive_inf = positive_inf_count > mx::array(0.0f, mx::float32);

  auto scaled = logits / mx::array(temperature, mx::float32);
  auto minus_large = mx::full_like(scaled, -1e9f);
  auto finite_logits = mx::where(finite, scaled, minus_large);
  auto inf_logits = mx::where(positive_inf, mx::zeros_like(scaled), minus_large);
  auto uniform_logits = mx::zeros_like(scaled);
  auto sample_logits = mx::where(
      has_positive_inf,
      inf_logits,
      mx::where(has_finite, finite_logits, uniform_logits));

  return mx::reshape(
      mx::astype(mx::random::categorical(sample_logits, 1, std::make_optional(key)), mx::int32),
      {static_cast<mx::ShapeElem>(rows)});
}

bool named_step_metadata_matches(const IRTrainer& trainer, const TensorMap& inputs) {
  if (!trainer.cached_named_step_metadata_valid ||
      trainer.cached_named_step_input_names.size() != inputs.size()) {
    return false;
  }
  for (size_t i = 0; i < trainer.cached_named_step_input_names.size(); ++i) {
    auto it = inputs.find(trainer.cached_named_step_input_names[i]);
    if (it == inputs.end()) {
      return false;
    }
    const auto& desc = it->second;
    if (desc.dtype != trainer.cached_named_step_input_dtypes[i] ||
        desc.shape != trainer.cached_named_step_input_shapes[i]) {
      return false;
    }
  }
  return true;
}

void refresh_named_step_metadata(IRTrainer& trainer, const TensorMap& inputs) {
  trainer.cached_named_step_argnums.resize(trainer.weights.size());
  std::iota(
      trainer.cached_named_step_argnums.begin(),
      trainer.cached_named_step_argnums.end(),
      0);
  trainer.cached_named_step_output_names = collect_training_step_output_names(trainer);
  trainer.cached_named_step_input_names = sorted_input_names(inputs);
  trainer.cached_named_step_input_dtypes.clear();
  trainer.cached_named_step_input_shapes.clear();
  trainer.cached_named_step_input_dtypes.reserve(trainer.cached_named_step_input_names.size());
  trainer.cached_named_step_input_shapes.reserve(trainer.cached_named_step_input_names.size());
  for (const auto& name : trainer.cached_named_step_input_names) {
    const auto& desc = inputs.at(name);
    trainer.cached_named_step_input_dtypes.push_back(desc.dtype);
    trainer.cached_named_step_input_shapes.push_back(desc.shape);
  }
  trainer.cached_named_step_signature = named_step_signature(
      trainer.program,
      trainer.qat_mode,
      trainer.compute_dtype,
      inputs,
      trainer.cached_named_step_input_names,
      trainer.cached_named_step_output_names,
      trainer.weights.size());
  trainer.cached_named_step_metadata_valid = true;
}

void ensure_named_step_metadata(IRTrainer& trainer, const TensorMap& inputs) {
  if (!named_step_metadata_matches(trainer, inputs)) {
    refresh_named_step_metadata(trainer, inputs);
  }
}

float materialize_step_loss(
    mx::array& loss_array,
    int step_index,
    const char* state_label) {
  float loss = 0.0f;
  try {
    loss = loss_array.item<float>();
  } catch (const std::exception& e) {
    throw std::runtime_error(
        std::string(state_label) + " step " + std::to_string(step_index) +
        " failed while materializing loss: " + e.what());
  }
  if (!std::isfinite(loss)) {
    throw std::runtime_error(
        std::string(state_label) + " step " + std::to_string(step_index) +
        " produced non-finite loss");
  }
  return loss;
}

float finalize_pending_step(
    IRTrainer& trainer,
    std::unordered_map<std::string, mx::array>* outputs) {
  if (!trainer.has_pending_step_) {
    throw std::runtime_error("no pending step to collect");
  }
  const float loss = materialize_step_loss(
      trainer.pending_loss_,
      trainer.pending_step_index_,
      "pending");
  if (outputs != nullptr) {
    *outputs = std::move(trainer.pending_outputs_);
  }
  trainer.pending_outputs_.clear();
  trainer.has_pending_step_ = false;
  trainer.pending_step_index_ = 0;
  return loss;
}

float finalize_ready_step(
    IRTrainer& trainer,
    std::unordered_map<std::string, mx::array>* outputs) {
  if (!trainer.has_ready_step_) {
    throw std::runtime_error("no ready step to collect");
  }
  const float loss = materialize_step_loss(
      trainer.ready_loss_,
      trainer.ready_step_index_,
      "ready");
  if (outputs != nullptr) {
    *outputs = std::move(trainer.ready_outputs_);
  }
  trainer.ready_outputs_.clear();
  trainer.has_ready_step_ = false;
  trainer.ready_step_index_ = 0;
  return loss;
}

void move_pending_step_to_ready(IRTrainer& trainer) {
  if (!trainer.has_pending_step_) {
    throw std::runtime_error("no pending step to queue");
  }
  if (trainer.has_ready_step_) {
    throw std::runtime_error("previous step loss must be collected before submitting another step");
  }
  trainer.ready_loss_ = std::move(trainer.pending_loss_);
  trainer.ready_outputs_ = std::move(trainer.pending_outputs_);
  trainer.ready_step_index_ = trainer.pending_step_index_;
  trainer.has_ready_step_ = true;
  trainer.pending_outputs_.clear();
  trainer.has_pending_step_ = false;
  trainer.pending_step_index_ = 0;
}

} // namespace

IRTrainer::IRTrainer()
    : pending_loss_(mx::array(0.0f, mx::float32)),
      ready_loss_(mx::array(0.0f, mx::float32)) {}

float IRTrainer::step(const mx::array& tokens, const mx::array& targets) {
  flush();
  if (weights.empty()) {
    throw std::runtime_error("IR trainer has no weights");
  }
  step_count++;

  std::vector<int> argnums(weights.size());
  std::iota(argnums.begin(), argnums.end(), 0);
  auto fn = mx::value_and_grad(
      [this, tokens, targets](const std::vector<mx::array>& w) {
        auto effective = effective_training_weights(w, qat_mode, compute_dtype);
        return ir_interpret(program, effective, tokens, targets, true);
      },
      argnums);

  auto out = fn(weights);
  auto loss = out.first;
  auto grads = std::move(out.second);
  OptimizerStepTransaction transaction(*this);
  auto raw_gradient_nonfinite = sanitize_and_clip_gradients(grads, max_grad_norm);
  apply_optimizer_updates(grads);

  std::vector<mx::array> eval_arrays;
  eval_arrays.reserve(
      1 + weights.size() + adam_m.size() * 2 + muon_momentum.size() + muon_second_moment.size() + sgd_momentum.size());
  eval_arrays.push_back(loss);
  collect_state_for_eval(*this, eval_arrays, false);
  transaction.append_validation_arrays(loss, grads, eval_arrays, raw_gradient_nonfinite);
  mx::eval(eval_arrays);
  const bool committed = transaction.finish();
  if (!committed) {
    std::unordered_map<std::string, mx::array> outputs;
    sanitize_skipped_step_reporting(loss, outputs, transaction.loss_was_nonfinite());
  }
  loss.detach();
  detach_trainer_state(*this);
  report_gated_delta_timing_summary("step", step_count);
  return loss.item<float>();
}

void IRTrainer::apply_optimizer_updates(const std::vector<mx::array>& grads) {
  if (grads.size() != weights.size()) {
    throw std::runtime_error("gradient/weight size mismatch");
  }

  for (size_t i = 0; i < weights.size(); ++i) {
    apply_weight_optimizer_update(i, grads[i]);
  }
}

void IRTrainer::apply_weight_optimizer_update(size_t i, const mx::array& g) {
  if (i >= weights.size()) {
    throw std::runtime_error("weight optimizer index out of range");
  }
  if (i >= weight_optimizers.size()) {
    throw std::runtime_error("missing weight optimizer spec");
  }
  const auto& spec = weight_optimizers[i];
  if (spec.group_index >= optimizer_groups.size()) {
    throw std::runtime_error("weight optimizer group index out of range");
  }
  const auto& group = optimizer_groups[spec.group_index];
  const float effective_lr = group.lr * lr_scale;
  const int optimizer_step = optimizer_step_count + 1;
  auto& w = weights[i];
  auto grad = mx::astype(g, mx::float32);

  switch (group.kind) {
    case OptimizerKind::AdamW: {
      if (has_adam_state[i] == 0) {
        throw std::runtime_error("AdamW state missing for weight");
      }
      const float b1t = 1.0f - std::pow(group.beta1, static_cast<float>(optimizer_step));
      const float b2t = 1.0f - std::pow(group.beta2, static_cast<float>(optimizer_step));
      const float one_minus_beta1 = 1.0f - group.beta1;
      const float one_minus_beta2 = 1.0f - group.beta2;
      adam_m[i] = group.beta1 * adam_m[i] + one_minus_beta1 * grad;
      adam_v[i] = group.beta2 * adam_v[i] + one_minus_beta2 * mx::square(grad);

      auto mhat = adam_m[i] / b1t;
      auto vhat = adam_v[i] / b2t;

      w = apply_weight_decay(w, grad, group, spec.decay, optimizer_step, effective_lr);
      w = w - effective_lr * mhat / (mx::sqrt(vhat) + group.eps);
      break;
    }
    case OptimizerKind::Lamb: {
      if (has_adam_state[i] == 0) {
        throw std::runtime_error("LAMB state missing for weight");
      }
      const float b1t = 1.0f - std::pow(group.beta1, static_cast<float>(optimizer_step));
      const float b2t = 1.0f - std::pow(group.beta2, static_cast<float>(optimizer_step));
      const float one_minus_beta1 = 1.0f - group.beta1;
      const float one_minus_beta2 = 1.0f - group.beta2;
      adam_m[i] = group.beta1 * adam_m[i] + one_minus_beta1 * grad;
      adam_v[i] = group.beta2 * adam_v[i] + one_minus_beta2 * mx::square(grad);

      auto mhat = adam_m[i] / b1t;
      auto vhat = adam_v[i] / b2t;
      auto update = mhat / (mx::sqrt(vhat) + group.eps);
      update = update + weight_decay_update_term(w, grad, group, spec.decay, optimizer_step);
      auto trust_ratio = lamb_trust_ratio(w, update, group.lamb_trust_ratio_cap);
      w = w - effective_lr * trust_ratio * update;
      break;
    }
    case OptimizerKind::Muon: {
      if (has_muon_state[i] == 0) {
        throw std::runtime_error("Muon state missing for weight");
      }
      if (w.ndim() != 2) {
        throw std::runtime_error("Muon only supports rank-2 weights");
      }
      muon_momentum[i] = group.beta1 * muon_momentum[i] + grad;
      mx::array update = group.nesterov ? (grad + group.beta1 * muon_momentum[i]) : muon_momentum[i];
      update = zeropower_via_newtonschulz5(update, group.backend_steps, group.newton_schulz_variant);
      const auto rows = static_cast<float>(w.shape(0));
      const auto cols = static_cast<float>(w.shape(1));
      const float aspect = std::sqrt(std::max(1.0f, rows / cols));
      update = update * mx::array(aspect, mx::float32);
      switch (group.muon_normalization) {
        case MuonNormalization::None:
          break;
        case MuonNormalization::RowL2:
          update = row_l2_normalize(update);
          break;
        case MuonNormalization::NorMuon:
          if (has_muon_second_moment_state[i] == 0) {
            throw std::runtime_error("NorMuon state missing for weight");
          }
          update = normuon_normalize(update, muon_second_moment[i], group.beta2);
          break;
      }
      w = apply_weight_decay(w, grad, group, spec.decay, optimizer_step, effective_lr);
      w = w - effective_lr * update;
      break;
    }
    case OptimizerKind::SGD: {
      if (has_sgd_state[i] == 0) {
        throw std::runtime_error("SGD state missing for weight");
      }
      sgd_momentum[i] = group.beta1 * sgd_momentum[i] + grad;
      w = apply_weight_decay(w, grad, group, spec.decay, optimizer_step, effective_lr);
      w = w - effective_lr * sgd_momentum[i];
      break;
    }
    default:
      throw std::runtime_error("unsupported optimizer kind");
  }
}

void IRTrainer::collect_weight_state_for_eval(size_t i, std::vector<mx::array>& eval_arrays) const {
  if (i >= weights.size()) {
    throw std::runtime_error("weight eval index out of range");
  }
  eval_arrays.push_back(weights[i]);
  if (i < adam_m.size() && has_adam_state[i] != 0) {
    eval_arrays.push_back(adam_m[i]);
    eval_arrays.push_back(adam_v[i]);
  }
  if (i < muon_momentum.size() && has_muon_state[i] != 0) {
    eval_arrays.push_back(muon_momentum[i]);
  }
  if (i < muon_second_moment.size() && has_muon_second_moment_state[i] != 0) {
    eval_arrays.push_back(muon_second_moment[i]);
  }
  if (i < sgd_momentum.size() && has_sgd_state[i] != 0) {
    eval_arrays.push_back(sgd_momentum[i]);
  }
}

float IRTrainer::step_named(const TensorMap& inputs) {
  flush();
  submit_step(inputs);
  return collect_loss();
}

void IRTrainer::submit_step(const TensorMap& inputs) {
  const auto submit_t0 = HostClock::now();
  if (weights.empty()) {
    throw std::runtime_error("IR trainer has no weights");
  }
  if (has_ready_step_) {
    throw std::runtime_error("previous step loss must be collected before submitting another step");
  }
  if (has_pending_step_) {
    move_pending_step_to_ready(*this);
  }
  step_count++;
  validate_fused_mamba3_cuda_primitive_config(program);
  const bool timing_enabled =
      program_has_fused_canonical_mamba3_block(program) &&
      should_log_mamba3_host_timing(step_count);
  long long timing_prep_us = 0;
  long long timing_grad_us = -1;
  long long timing_opt_us = -1;
  long long timing_eval_us = -1;
  auto log_timing = [&](const char* path) {
    if (!timing_enabled) {
      return;
    }
    const auto now = HostClock::now();
    std::cout << "[mlx_ir] canonical Mamba3 host timing"
              << " step=" << step_count
              << " path=" << path
              << " prep_us=" << timing_prep_us;
    if (timing_grad_us >= 0) {
      std::cout << " grad_us=" << timing_grad_us;
    }
    if (timing_opt_us >= 0) {
      std::cout << " opt_us=" << timing_opt_us;
    }
    if (timing_eval_us >= 0) {
      std::cout << " eval_us=" << timing_eval_us;
    }
    std::cout << " total_us=" << elapsed_us(submit_t0, now)
              << std::endl;
  };

  ensure_named_step_metadata(*this, inputs);
  const auto& argnums = cached_named_step_argnums;
  const auto& output_names = cached_named_step_output_names;
  const auto& input_names = cached_named_step_input_names;
  const auto& signature = cached_named_step_signature;

  auto input_arrays_by_name = tensor_map_to_arrays(inputs);
  std::vector<mx::array> ordered_input_arrays;
  ordered_input_arrays.reserve(input_names.size());
  for (const auto& name : input_names) {
    ordered_input_arrays.push_back(input_arrays_by_name.at(name));
  }
  std::vector<mx::array> args;
  args.reserve(weights.size() + input_names.size());
  args.insert(args.end(), weights.begin(), weights.end());
  args.insert(args.end(), ordered_input_arrays.begin(), ordered_input_arrays.end());
  if (timing_enabled) {
    timing_prep_us = elapsed_us(submit_t0, HostClock::now());
  }

  StepForwardFn forward_fn;
  auto get_forward_fn = [&]() -> const StepForwardFn& {
    if (forward_fn) {
      return forward_fn;
    }
    const auto local_input_names = input_names;
    const auto local_output_names = output_names;
    forward_fn = StepForwardFn(
      [this, local_input_names, local_output_names](const std::vector<mx::array>& fn_args) {
        const auto n_weights = weights.size();
        if (fn_args.size() < n_weights + local_input_names.size()) {
          throw std::runtime_error("IR trainer argument count mismatch");
        }
        std::vector<mx::array> w;
        w.reserve(n_weights);
        for (size_t i = 0; i < n_weights; ++i) {
          w.push_back(fn_args[i]);
        }
        auto effective = effective_training_weights(w, qat_mode, compute_dtype);
        ArrayMap input_map;
        input_map.reserve(local_input_names.size());
        for (size_t i = 0; i < local_input_names.size(); ++i) {
          input_map.emplace(local_input_names[i], fn_args[n_weights + i]);
        }
        auto outputs = ir_interpret_outputs(program, effective, input_map, local_output_names, true);
        std::vector<mx::array> values;
        values.reserve(local_output_names.size());
        for (const auto& output_name : local_output_names) {
          values.push_back(outputs.at(output_name));
        }
        return values;
      });
    return forward_fn;
  };

  bool use_compiled_step = use_compiled_training_step(program);
  const bool trace_backward =
      backward_trace_enabled_for_step(step_count) ||
      (env_truthy("MIXLAB_MLX_BACKWARD_TRACE_AFTER_SKIP") &&
       consecutive_skipped_optimizer_steps > 0);
  if (trace_backward) {
    use_compiled_step = false;
  }
  if (program_has_fused_canonical_mamba3_block(program) &&
      fused_mamba3_compiled_step_disabled &&
      !env_truthy("MIXLAB_FORCE_COMPILED_STEP")) {
    use_compiled_step = false;
  }
  const bool checkpoint_step = checkpoint_training_step(program);
  auto run_low_memory_mamba3_step = [&]() {
    const int grad_chunk_size = low_memory_mamba3_grad_chunk_size();
    const int requested_grad_chunk_elements = low_memory_mamba3_grad_chunk_elements();
    const int update_chunk_size = low_memory_mamba3_update_chunk_size();
    auto gradient_mode = low_memory_mamba3_gradient_mode(program);
    if (gradient_mode == Mamba3LowMemoryGradientMode::SingleBackward &&
        mamba3_single_backward_disabled) {
      gradient_mode = Mamba3LowMemoryGradientMode::MaterializedChunks;
    }
    const auto grad_chunk_element_candidates =
        gradient_mode == Mamba3LowMemoryGradientMode::MaterializedChunks
        ? mamba3_grad_chunk_element_candidates(*this, requested_grad_chunk_elements)
        : std::vector<int>{requested_grad_chunk_elements};
    const int grad_chunk_elements = grad_chunk_element_candidates.front();
    const auto grad_chunks = build_weight_grad_chunks(weights, grad_chunk_size, grad_chunk_elements);
    const bool checkpoint_grad_forward =
        should_checkpoint_mamba3_grad_forward(program, checkpoint_step, gradient_mode);
    if (!memory_safe_step_notice_logged_) {
      std::cerr << "[mlx_ir] canonical Mamba3 detected; using uncompiled"
                << (checkpoint_grad_forward ? " checkpointed" : "")
                << " training step to avoid oversized CUDA graphs"
                << " (set MIXLAB_FORCE_COMPILED_STEP=1 to override)" << std::endl;
      memory_safe_step_notice_logged_ = true;
    }
    if (!low_memory_update_notice_logged_) {
      std::cerr << "[mlx_ir] canonical Mamba3 using low-memory "
                << low_memory_mamba3_gradient_mode_label(gradient_mode)
                << " gradient eval"
                << " (grad_chunk=" << grad_chunk_size
                << " grad_chunk_elems=" << grad_chunk_elements
                << " grad_chunks=" << grad_chunks.size()
                << " update_chunk=" << update_chunk_size
                << " compiled_grad_chunks=" << (use_compiled_mamba3_grad_chunks() ? "on" : "off")
                << "; set MIXLAB_MAMBA3_CHECKPOINT_CHUNKED_GRADS=1 if chunked grads OOM"
                << "; set MIXLAB_MAMBA3_SINGLE_BACKWARD=1 to force the full backward graph"
                << "; set MIXLAB_MAMBA3_MATERIALIZED_GRADS=1 to force materialized chunks"
                << "; set MIXLAB_MAMBA3_CHECKPOINT_SINGLE_BACKWARD=1 if single backward OOMs"
                << "; set MIXLAB_MAMBA3_CHUNKED_AUTODIFF=1 for the slowest no-cache fallback"
                << "; set MIXLAB_MAMBA3_COMPILED_GRAD_CHUNKS=1 to try per-chunk CUDA graphs"
                << "; set MIXLAB_MAMBA3_ADAPTIVE_GRAD_CHUNKS=1 to probe larger grad chunks"
                << "; set MIXLAB_DISABLE_MAMBA3_LOW_MEMORY_UPDATES=1 to disable)"
                << std::endl;
      low_memory_update_notice_logged_ = true;
    }

    auto grad_forward_fn_for_mode =
        [&](Mamba3LowMemoryGradientMode mode) -> StepForwardFn {
          const auto base_forward_fn = get_forward_fn();
          return should_checkpoint_mamba3_grad_forward(program, checkpoint_step, mode)
              ? mx::checkpoint(base_forward_fn)
              : base_forward_fn;
        };
    std::vector<mx::array> output_values;
    std::vector<mx::array> all_grads;
    std::vector<MaterializedGrad> materialized_grads;
    double materialized_norm_sq = 0.0;
    float single_clip_scale_value = 1.0f;
    bool single_clip_scale_ready = false;
    if (gradient_mode == Mamba3LowMemoryGradientMode::SingleBackward) {
      try {
        auto grad_fn = mx::value_and_grad(grad_forward_fn_for_mode(gradient_mode), argnums);
        auto result = grad_fn(args);
        output_values = std::move(result.first);
        all_grads = std::move(result.second);
        if (all_grads.size() != weights.size()) {
          throw std::runtime_error("canonical Mamba3 gradient count mismatch");
        }
        eval_arrays_with_context(
            output_values,
            "canonical Mamba3 single-backward output eval");
        single_clip_scale_value = gradient_clip_scale_value_from_grads(
            all_grads,
            max_grad_norm,
            grad_chunk_size);
        single_clip_scale_ready = true;
      } catch (const std::exception& e) {
        if (!is_retriable_cuda_memory_error(e) ||
            env_truthy("MIXLAB_MAMBA3_SINGLE_BACKWARD")) {
          throw;
        }
        log_mamba3_single_backward_fallback_once(*this, e);
        mamba3_single_backward_disabled = true;
        gradient_mode = Mamba3LowMemoryGradientMode::MaterializedChunks;
        output_values.clear();
        all_grads.clear();
      }
    }
    if (gradient_mode == Mamba3LowMemoryGradientMode::MaterializedChunks) {
      auto grad_forward_fn = grad_forward_fn_for_mode(gradient_mode);
      bool materialized_ok = false;
      std::exception_ptr last_materialize_error;
      for (size_t candidate_idx = 0; candidate_idx < grad_chunk_element_candidates.size(); ++candidate_idx) {
        const int candidate_elements = grad_chunk_element_candidates[candidate_idx];
        const auto candidate_chunks = build_weight_grad_chunks(weights, grad_chunk_size, candidate_elements);
        const auto candidate_signature = mamba3_grad_chunks_signature(
            signature,
            candidate_chunks,
            checkpoint_grad_forward);
        try {
          auto materialized = materialize_weight_grad_chunks(
              *this,
              grad_forward_fn,
              args,
              weights.size(),
              candidate_chunks,
              candidate_signature);
          output_values = std::move(materialized.output_values);
          materialized_grads = std::move(materialized.grads);
          materialized_norm_sq = materialized.norm_sq;
          adaptive_mamba3_grad_chunk_elements = candidate_elements;
          materialized_ok = true;
          break;
        } catch (const std::exception& e) {
          last_materialize_error = std::current_exception();
          if (!is_retriable_cuda_memory_error(e) ||
              candidate_idx + 1 >= grad_chunk_element_candidates.size() ||
              env_is_set("MIXLAB_MAMBA3_GRAD_CHUNK_ELEMS")) {
            std::rethrow_exception(last_materialize_error);
          }
          log_mamba3_grad_chunk_fallback_once(
              *this,
              candidate_elements,
              grad_chunk_element_candidates[candidate_idx + 1],
              e);
          compiled_mamba3_grad_chunks.clear();
          compiled_mamba3_grad_chunks_signature.clear();
          compiled_mamba3_grad_chunks_disabled = false;
        }
      }
      if (!materialized_ok) {
        if (last_materialize_error) {
          std::rethrow_exception(last_materialize_error);
        }
        throw std::runtime_error("canonical Mamba3 materialized gradients were not produced");
      }
    } else if (gradient_mode == Mamba3LowMemoryGradientMode::RecomputeChunks) {
      auto grad_forward_fn = grad_forward_fn_for_mode(gradient_mode);
      output_values = get_forward_fn()(args);
    }
    if (output_values.size() != output_names.size()) {
      throw std::runtime_error("IR trainer output count mismatch");
    }
    auto loss = output_values[0];
    std::unordered_map<std::string, mx::array> outputs;
    for (size_t i = 0; i < output_names.size(); ++i) {
      outputs.emplace(output_names[i], output_values[i]);
    }
    std::vector<mx::array> output_eval_arrays;
    output_eval_arrays.reserve(1 + outputs.size());
    output_eval_arrays.push_back(loss);
    for (const auto& [_, output] : outputs) {
      output_eval_arrays.push_back(output);
    }
    eval_arrays_with_context(
        output_eval_arrays,
        "canonical Mamba3 low-memory output eval");
    loss.detach();
    detach_output_map(outputs);
    OptimizerStepTransaction transaction(*this);

    float clip_scale_value = 1.0f;
    if (gradient_mode == Mamba3LowMemoryGradientMode::SingleBackward) {
      clip_scale_value = single_clip_scale_ready
          ? single_clip_scale_value
          : gradient_clip_scale_value_from_grads(
                all_grads,
                max_grad_norm,
                grad_chunk_size);
    } else if (gradient_mode == Mamba3LowMemoryGradientMode::MaterializedChunks) {
      clip_scale_value = gradient_clip_scale_from_norm_sq(
          materialized_norm_sq,
          max_grad_norm);
    } else {
      auto grad_forward_fn = grad_forward_fn_for_mode(gradient_mode);
      clip_scale_value = gradient_clip_scale_value_chunked_autodiff(
              grad_forward_fn,
              args,
              weights.size(),
              max_grad_norm,
              grad_chunk_size);
    }
    const auto clip_scale = mx::array(clip_scale_value, mx::float32);
    uint64_t low_memory_gradient_nonfinite = 0;

    for (size_t start = 0; start < weights.size(); start += static_cast<size_t>(update_chunk_size)) {
      const size_t end = std::min(weights.size(), start + static_cast<size_t>(update_chunk_size));
      std::vector<mx::array> grads;
      if (gradient_mode == Mamba3LowMemoryGradientMode::SingleBackward) {
        grads.assign(all_grads.begin() + static_cast<std::ptrdiff_t>(start),
                     all_grads.begin() + static_cast<std::ptrdiff_t>(end));
      } else if (gradient_mode == Mamba3LowMemoryGradientMode::MaterializedChunks) {
        grads.reserve(end - start);
        for (size_t i = start; i < end; ++i) {
          grads.push_back(materialized_grad_to_array(materialized_grads[i]));
        }
      } else {
        auto grad_forward_fn = grad_forward_fn_for_mode(gradient_mode);
        grads = compute_weight_grad_chunk(
            grad_forward_fn,
            args,
            start,
            end,
            "canonical Mamba3 optimizer grad chunk [" + std::to_string(start) +
            "," + std::to_string(end) + ")");
      }
      if (grads.size() != end - start) {
        throw std::runtime_error("canonical Mamba3 gradient chunk size mismatch");
      }
      std::vector<mx::array> update_eval_arrays;
      update_eval_arrays.reserve((end - start) * 3 + 1);
      auto chunk_gradient_nonfinite = mx::array(0.0f, mx::float32);
      for (size_t i = start; i < end; ++i) {
        chunk_gradient_nonfinite = chunk_gradient_nonfinite + mx::sum(mx::astype(
            mx::logical_not(mx::isfinite(grads[i - start])),
            mx::float32));
        auto clipped_grad = grads[i - start] * clip_scale;
        apply_weight_optimizer_update(i, clipped_grad);
        collect_weight_state_for_eval(i, update_eval_arrays);
      }
      update_eval_arrays.push_back(chunk_gradient_nonfinite);
      eval_arrays_with_context(
          update_eval_arrays,
          "canonical Mamba3 optimizer update chunk [" + std::to_string(start) +
          "," + std::to_string(end) + ")");
      const float chunk_bad = chunk_gradient_nonfinite.item<float>();
      if (std::isfinite(chunk_bad) && chunk_bad > 0.0f) {
        low_memory_gradient_nonfinite += static_cast<uint64_t>(chunk_bad);
      }
    }

    std::vector<mx::array> validation_arrays;
    validation_arrays.reserve(3);
    const std::vector<mx::array> no_validation_grads;
    transaction.append_validation_arrays(
        loss,
        no_validation_grads,
        validation_arrays,
        mx::array(static_cast<float>(low_memory_gradient_nonfinite), mx::float32));
    eval_arrays_with_context(validation_arrays, "canonical Mamba3 optimizer transaction validation");
    const bool committed = transaction.finish();
    if (!committed) {
      sanitize_skipped_step_reporting(loss, outputs, transaction.loss_was_nonfinite());
    }
    detach_trainer_state(*this);
    report_gated_delta_timing_summary("step", step_count);
    pending_loss_ = loss;
    pending_outputs_ = std::move(outputs);
    has_pending_step_ = true;
    pending_step_index_ = step_count;
    log_timing("low-memory");
  };
  if (!use_compiled_step && use_low_memory_mamba3_updates(program)) {
    run_low_memory_mamba3_step();
    return;
  }

  auto run_compiled_fused_mamba3_adamw_update_step = [&]() -> bool {
    if (!supports_compiled_fused_mamba3_adamw_update_step(*this)) {
      return false;
    }
    if (!fused_mamba3_compiled_update_step_notice_logged) {
      std::cerr << "[mlx_ir] fused canonical Mamba3 using compiled AdamW update training step"
                << " (experimental; unset MIXLAB_MAMBA3_COMPILED_UPDATE_STEP to use compiled-gradient path)"
                << std::endl;
      fused_mamba3_compiled_update_step_notice_logged = true;
    }
    try {
      const auto update_signature = compiled_adamw_update_step_signature(
          signature,
          *this,
          ordered_input_arrays.size());
      if (!compiled_named_update_step || compiled_named_update_step_signature != update_signature) {
        const size_t n_weights = weights.size();
        const size_t input_count = ordered_input_arrays.size();
        const auto local_weight_optimizers = weight_optimizers;
        const auto local_groups = optimizer_groups;
        const float local_max_grad_norm = max_grad_norm;
        const auto base_forward_fn = get_forward_fn();
        auto update_forward_fn = StepForwardFn(
            [base_forward_fn, n_weights, input_count](const std::vector<mx::array>& full_args) {
              const size_t input_offset = n_weights * 3;
              if (full_args.size() < input_offset + input_count) {
                throw std::runtime_error("compiled AdamW update step argument count mismatch");
              }
              std::vector<mx::array> forward_args;
              forward_args.reserve(n_weights + input_count);
              forward_args.insert(
                  forward_args.end(),
                  full_args.begin(),
                  full_args.begin() + static_cast<std::ptrdiff_t>(n_weights));
              forward_args.insert(
                  forward_args.end(),
                  full_args.begin() + static_cast<std::ptrdiff_t>(input_offset),
                  full_args.begin() + static_cast<std::ptrdiff_t>(input_offset + input_count));
              return base_forward_fn(forward_args);
            });
        auto grad_fn = mx::value_and_grad(update_forward_fn, argnums);
        compiled_named_update_step = mx::compile(
            [grad_fn,
             n_weights,
             input_count,
             local_weight_optimizers,
             local_groups,
             local_max_grad_norm](const std::vector<mx::array>& fn_args) {
              const size_t m_offset = n_weights;
              const size_t v_offset = n_weights * 2;
              const size_t input_offset = n_weights * 3;
              const size_t scalar_offset = input_offset + input_count;
              const size_t expected_args = scalar_offset + 1 + local_groups.size() * 2;
              if (fn_args.size() != expected_args) {
                throw std::runtime_error("compiled AdamW update step received wrong argument count");
              }

              auto result = grad_fn(fn_args);
              auto grads = std::move(result.second);
              if (grads.size() != n_weights) {
                throw std::runtime_error("compiled AdamW update step gradient count mismatch");
              }

              auto total_norm_sq = mx::array(0.0f, mx::float32);
              for (const auto& grad : grads) {
                total_norm_sq = total_norm_sq + mx::sum(mx::square(grad));
              }
              auto clip_scale = mx::array(1.0f, mx::float32);
              if (local_max_grad_norm > 0.0f) {
                clip_scale = mx::minimum(
                    mx::array(1.0f, mx::float32),
                    mx::array(local_max_grad_norm, mx::float32) /
                        (mx::sqrt(total_norm_sq) + mx::array(1e-6f, mx::float32)));
              }

              const auto& lr_scale_arg = fn_args[scalar_offset];
              std::vector<mx::array> updated_weights;
              std::vector<mx::array> updated_m;
              std::vector<mx::array> updated_v;
              updated_weights.reserve(n_weights);
              updated_m.reserve(n_weights);
              updated_v.reserve(n_weights);
              for (size_t i = 0; i < n_weights; ++i) {
                const auto& spec = local_weight_optimizers[i];
                const auto& group = local_groups[spec.group_index];
                const auto& b1t = fn_args[scalar_offset + 1 + static_cast<size_t>(spec.group_index) * 2];
                const auto& b2t = fn_args[scalar_offset + 2 + static_cast<size_t>(spec.group_index) * 2];
                auto grad = grads[i] * clip_scale;
                auto m = group.beta1 * fn_args[m_offset + i] + (1.0f - group.beta1) * grad;
                auto v = group.beta2 * fn_args[v_offset + i] + (1.0f - group.beta2) * mx::square(grad);
                auto mhat = m / b1t;
                auto vhat = v / b2t;
                auto effective_lr = group.lr * lr_scale_arg;
                auto weight = fn_args[i];
                if (spec.decay && group.weight_decay > 0.0f) {
                  weight = weight - (effective_lr * group.weight_decay) * weight;
                }
                updated_weights.push_back(
                    weight - effective_lr * mhat / (mx::sqrt(vhat) + group.eps));
                updated_m.push_back(m);
                updated_v.push_back(v);
              }

              std::vector<mx::array> out;
              out.reserve(result.first.size() + n_weights * 3);
              out.insert(out.end(), result.first.begin(), result.first.end());
              out.insert(out.end(), updated_weights.begin(), updated_weights.end());
              out.insert(out.end(), updated_m.begin(), updated_m.end());
              out.insert(out.end(), updated_v.begin(), updated_v.end());
              return out;
            },
            false);
        compiled_named_update_step_signature = update_signature;
      }

      const auto update_t0 = HostClock::now();
      auto update_args = build_compiled_adamw_update_step_args(*this, ordered_input_arrays);
      auto update_out = compiled_named_update_step(update_args);
      if (timing_enabled) {
        timing_opt_us = elapsed_us(update_t0, HostClock::now());
      }
      const size_t expected_out = output_names.size() + weights.size() * 3;
      if (update_out.size() != expected_out) {
        throw std::runtime_error("compiled AdamW update step output count mismatch");
      }

      auto loss = update_out[0];
      std::unordered_map<std::string, mx::array> outputs;
      for (size_t i = 0; i < output_names.size(); ++i) {
        outputs.emplace(output_names[i], update_out[i]);
      }

      size_t offset = output_names.size();
      std::vector<mx::array> next_weights;
      std::vector<mx::array> next_adam_m;
      std::vector<mx::array> next_adam_v;
      next_weights.reserve(weights.size());
      next_adam_m.reserve(adam_m.size());
      next_adam_v.reserve(adam_v.size());
      for (size_t i = 0; i < weights.size(); ++i) {
        next_weights.push_back(update_out[offset + i]);
      }
      offset += weights.size();
      for (size_t i = 0; i < adam_m.size(); ++i) {
        next_adam_m.push_back(update_out[offset + i]);
      }
      offset += adam_m.size();
      for (size_t i = 0; i < adam_v.size(); ++i) {
        next_adam_v.push_back(update_out[offset + i]);
      }

      OptimizerStepTransaction transaction(*this);
      weights = std::move(next_weights);
      adam_m = std::move(next_adam_m);
      adam_v = std::move(next_adam_v);
      std::vector<mx::array> eval_arrays;
      eval_arrays.reserve(1 + outputs.size() + weights.size() + adam_m.size() + adam_v.size() + 3);
      eval_arrays.push_back(loss);
      for (const auto& [_, output] : outputs) {
        eval_arrays.push_back(output);
      }
      collect_state_for_eval(*this, eval_arrays, false);
      transaction.append_validation_arrays(
          loss, {}, eval_arrays, mx::array(0.0f, mx::float32));
      const auto eval_t0 = HostClock::now();
      mx::eval(eval_arrays);
      if (timing_enabled) {
        timing_eval_us = elapsed_us(eval_t0, HostClock::now());
      }
      const bool committed = transaction.finish();
      if (!committed) {
        sanitize_skipped_step_reporting(loss, outputs, transaction.loss_was_nonfinite());
      }
      loss.detach();
      detach_output_map(outputs);
      detach_trainer_state(*this);
      report_gated_delta_timing_summary("step", step_count);
      pending_loss_ = loss;
      pending_outputs_ = std::move(outputs);
      has_pending_step_ = true;
      pending_step_index_ = step_count;
      log_timing("compiled-update-step");
      return true;
    } catch (const OptimizerStepCircuitBreaker&) {
      throw;
    } catch (const std::exception& e) {
      if (env_truthy("MIXLAB_FORCE_MAMBA3_COMPILED_UPDATE_STEP")) {
        throw;
      }
      log_fused_mamba3_compiled_update_step_fallback_once(*this, e);
      fused_mamba3_compiled_update_step_disabled = true;
      compiled_named_update_step = nullptr;
      compiled_named_update_step_signature.clear();
      return false;
    }
  };

  if (use_compiled_step && run_compiled_fused_mamba3_adamw_update_step()) {
    return;
  }

  std::vector<mx::array> step_out;
  std::unique_ptr<BackwardTraceCollector> backward_trace;
  if (trace_backward) {
    backward_trace = std::make_unique<BackwardTraceCollector>(step_count);
  }
  if (use_compiled_step) {
    if (program_has_fused_canonical_mamba3_block(program) &&
        !fused_mamba3_compiled_step_notice_logged) {
      std::cerr << "[mlx_ir] fused canonical Mamba3 using compiled training step"
                << " (set MIXLAB_DISABLE_MAMBA3_COMPILED_STEP=1 to use low-memory fallback)"
                << std::endl;
      fused_mamba3_compiled_step_notice_logged = true;
    }
    try {
      if (!compiled_named_step || compiled_named_step_signature != signature) {
        auto cached = compiled_named_step_cache.find(signature);
        if (cached != compiled_named_step_cache.end()) {
          compiled_named_step_cache_hits++;
          log_compile_cache_event(
              "training_step",
              "hit",
              step_count,
              compiled_named_step_cache.size(),
              signature);
          compiled_named_step = cached->second;
        } else {
          compiled_named_step_cache_misses++;
          log_compile_cache_event(
              "training_step",
              "miss",
              step_count,
              compiled_named_step_cache.size(),
              signature);
          const bool compiled_checkpoint_step =
              checkpoint_step &&
              (!program_has_fused_canonical_mamba3_block(program) ||
               env_truthy("MIXLAB_MAMBA3_CHECKPOINT_COMPILED_STEP"));
          const auto base_forward_fn = get_forward_fn();
          auto compiled_forward_fn = compiled_checkpoint_step
              ? mx::checkpoint(base_forward_fn)
              : base_forward_fn;
          auto grad_fn = mx::value_and_grad(compiled_forward_fn, argnums);
          compiled_named_step = mx::compile(
              [grad_fn](const std::vector<mx::array>& fn_args) {
                auto result = grad_fn(fn_args);
                std::vector<mx::array> out;
                out.reserve(result.first.size() + result.second.size());
                out.insert(out.end(), result.first.begin(), result.first.end());
                out.insert(out.end(), result.second.begin(), result.second.end());
                return out;
              },
              false);
          compiled_named_step_cache[signature] = compiled_named_step;
        }
        compiled_named_step_signature = signature;
      } else {
        compiled_named_step_cache_hits++;
      }
      const auto grad_t0 = HostClock::now();
      step_out = compiled_named_step(args);
      if (timing_enabled) {
        timing_grad_us = elapsed_us(grad_t0, HostClock::now());
      }
    } catch (const std::exception& e) {
      if (!program_has_fused_canonical_mamba3_block(program) ||
          !use_low_memory_mamba3_updates(program) ||
          env_truthy("MIXLAB_FORCE_COMPILED_STEP")) {
        if (!compiled_named_step_signature.empty()) {
          compiled_named_step_cache.erase(compiled_named_step_signature);
        }
        compiled_named_step = nullptr;
        compiled_named_step_signature.clear();
        throw;
      }
      log_fused_mamba3_compiled_step_fallback_once(*this, e);
      fused_mamba3_compiled_step_disabled = true;
      if (!compiled_named_step_signature.empty()) {
        compiled_named_step_cache.erase(compiled_named_step_signature);
      }
      compiled_named_step = nullptr;
      compiled_named_step_signature.clear();
      run_low_memory_mamba3_step();
      return;
    }
  } else {
    if (trace_backward) {
      std::cerr << "[mlx_ir] backward trace enabled"
                << " training_step=" << step_count
                << " path=eager"
                << std::endl;
    } else if (program_has_canonical_mamba3(program) &&
               !memory_safe_step_notice_logged_) {
      std::cerr << "[mlx_ir] canonical Mamba3 detected; using uncompiled"
                << (checkpoint_step ? " checkpointed" : "")
                << " training step to avoid oversized CUDA graphs"
                << " (set MIXLAB_FORCE_COMPILED_STEP=1 to override)" << std::endl;
      memory_safe_step_notice_logged_ = true;
    }
    const auto base_forward_fn = get_forward_fn();
    auto eager_forward_fn = checkpoint_step ? mx::checkpoint(base_forward_fn) : base_forward_fn;
    auto grad_fn = mx::value_and_grad(eager_forward_fn, argnums);
    const auto grad_t0 = HostClock::now();
    BackwardTraceScope trace_scope(backward_trace.get());
    auto result = grad_fn(args);
    if (timing_enabled) {
      timing_grad_us = elapsed_us(grad_t0, HostClock::now());
    }
    step_out.reserve(result.first.size() + result.second.size());
    step_out.insert(step_out.end(), result.first.begin(), result.first.end());
    step_out.insert(step_out.end(), result.second.begin(), result.second.end());
  }

  if (step_out.size() != output_names.size() + weights.size()) {
    throw std::runtime_error("IR trainer output count mismatch");
  }
  auto loss = step_out[0];
  std::unordered_map<std::string, mx::array> outputs;
  for (size_t i = 0; i < output_names.size(); ++i) {
    outputs.emplace(output_names[i], step_out[i]);
  }
  std::vector<mx::array> grads;
  grads.reserve(weights.size());
  for (size_t i = 0; i < weights.size(); ++i) {
    grads.push_back(step_out[output_names.size() + i]);
  }
  auto run_compiled_fused_mamba3_adamw_optimizer_update = [&]() -> bool {
    if (!supports_compiled_fused_mamba3_adamw_optimizer_update(*this)) {
      return false;
    }
    if (!fused_mamba3_compiled_optimizer_update_notice_logged) {
      std::cerr << "[mlx_ir] fused canonical Mamba3 using compiled AdamW optimizer update"
                << " (set MIXLAB_DISABLE_MAMBA3_COMPILED_OPTIMIZER_UPDATE=1 to use host-built update)"
                << std::endl;
      fused_mamba3_compiled_optimizer_update_notice_logged = true;
    }
    try {
      const auto update_signature = compiled_adamw_optimizer_update_signature(*this);
      if (!compiled_mamba3_optimizer_update ||
          compiled_mamba3_optimizer_update_signature != update_signature) {
        const size_t n_weights = weights.size();
        const auto local_weight_optimizers = weight_optimizers;
        const auto local_groups = optimizer_groups;
        const float local_max_grad_norm = max_grad_norm;
        compiled_mamba3_optimizer_update = mx::compile(
            [n_weights,
             local_weight_optimizers,
             local_groups,
             local_max_grad_norm](const std::vector<mx::array>& fn_args) {
              const size_t m_offset = n_weights;
              const size_t v_offset = n_weights * 2;
              const size_t grad_offset = n_weights * 3;
              const size_t scalar_offset = n_weights * 4;
              const size_t expected_args = scalar_offset + 1 + local_groups.size() * 2;
              if (fn_args.size() != expected_args) {
                throw std::runtime_error("compiled AdamW optimizer update received wrong argument count");
              }

              auto total_norm_sq = mx::array(0.0f, mx::float32);
              for (size_t i = 0; i < n_weights; ++i) {
                total_norm_sq = total_norm_sq + mx::sum(mx::square(fn_args[grad_offset + i]));
              }
              auto clip_scale = mx::array(1.0f, mx::float32);
              if (local_max_grad_norm > 0.0f) {
                clip_scale = mx::minimum(
                    mx::array(1.0f, mx::float32),
                    mx::array(local_max_grad_norm, mx::float32) /
                        (mx::sqrt(total_norm_sq) + mx::array(1e-6f, mx::float32)));
              }

              const auto& lr_scale_arg = fn_args[scalar_offset];
              std::vector<mx::array> updated_weights;
              std::vector<mx::array> updated_m;
              std::vector<mx::array> updated_v;
              updated_weights.reserve(n_weights);
              updated_m.reserve(n_weights);
              updated_v.reserve(n_weights);
              for (size_t i = 0; i < n_weights; ++i) {
                const auto& spec = local_weight_optimizers[i];
                const auto& group = local_groups[spec.group_index];
                const auto& b1t = fn_args[scalar_offset + 1 + static_cast<size_t>(spec.group_index) * 2];
                const auto& b2t = fn_args[scalar_offset + 2 + static_cast<size_t>(spec.group_index) * 2];
                auto grad = fn_args[grad_offset + i] * clip_scale;
                auto m = group.beta1 * fn_args[m_offset + i] + (1.0f - group.beta1) * grad;
                auto v = group.beta2 * fn_args[v_offset + i] + (1.0f - group.beta2) * mx::square(grad);
                auto mhat = m / b1t;
                auto vhat = v / b2t;
                auto effective_lr = group.lr * lr_scale_arg;
                auto weight = fn_args[i];
                if (spec.decay && group.weight_decay > 0.0f) {
                  weight = weight - (effective_lr * group.weight_decay) * weight;
                }
                updated_weights.push_back(
                    weight - effective_lr * mhat / (mx::sqrt(vhat) + group.eps));
                updated_m.push_back(m);
                updated_v.push_back(v);
              }

              std::vector<mx::array> out;
              out.reserve(n_weights * 3);
              out.insert(out.end(), updated_weights.begin(), updated_weights.end());
              out.insert(out.end(), updated_m.begin(), updated_m.end());
              out.insert(out.end(), updated_v.begin(), updated_v.end());
              return out;
            },
            false);
        compiled_mamba3_optimizer_update_signature = update_signature;
      }

      const auto update_t0 = HostClock::now();
      auto update_args = build_compiled_adamw_optimizer_update_args(*this, grads);
      auto update_out = compiled_mamba3_optimizer_update(update_args);
      if (timing_enabled) {
        timing_opt_us = elapsed_us(update_t0, HostClock::now());
      }
      const size_t expected_out = weights.size() * 3;
      if (update_out.size() != expected_out) {
        throw std::runtime_error("compiled AdamW optimizer update output count mismatch");
      }

      std::vector<mx::array> next_weights;
      std::vector<mx::array> next_adam_m;
      std::vector<mx::array> next_adam_v;
      next_weights.reserve(weights.size());
      next_adam_m.reserve(adam_m.size());
      next_adam_v.reserve(adam_v.size());
      size_t offset = 0;
      for (size_t i = 0; i < weights.size(); ++i) {
        next_weights.push_back(update_out[offset + i]);
      }
      offset += weights.size();
      for (size_t i = 0; i < adam_m.size(); ++i) {
        next_adam_m.push_back(update_out[offset + i]);
      }
      offset += adam_m.size();
      for (size_t i = 0; i < adam_v.size(); ++i) {
        next_adam_v.push_back(update_out[offset + i]);
      }

      OptimizerStepTransaction transaction(*this);
      weights = std::move(next_weights);
      adam_m = std::move(next_adam_m);
      adam_v = std::move(next_adam_v);
      std::vector<mx::array> eval_arrays;
      eval_arrays.reserve(1 + outputs.size() + weights.size() + adam_m.size() + adam_v.size() + 3);
      eval_arrays.push_back(loss);
      for (const auto& [_, output] : outputs) {
        eval_arrays.push_back(output);
      }
      collect_state_for_eval(*this, eval_arrays, false);
      transaction.append_validation_arrays(
          loss, grads, eval_arrays, mx::array(0.0f, mx::float32));
      const auto eval_t0 = HostClock::now();
      mx::eval(eval_arrays);
      if (timing_enabled) {
        timing_eval_us = elapsed_us(eval_t0, HostClock::now());
      }
      const bool committed = transaction.finish();
      if (!committed) {
        sanitize_skipped_step_reporting(loss, outputs, transaction.loss_was_nonfinite());
      }
      loss.detach();
      detach_output_map(outputs);
      detach_trainer_state(*this);
      report_gated_delta_timing_summary("step", step_count);
      pending_loss_ = loss;
      pending_outputs_ = std::move(outputs);
      has_pending_step_ = true;
      pending_step_index_ = step_count;
      log_timing("compiled-gradient+compiled-adamw");
      return true;
    } catch (const OptimizerStepCircuitBreaker&) {
      throw;
    } catch (const std::exception& e) {
      log_fused_mamba3_compiled_optimizer_update_fallback_once(*this, e);
      fused_mamba3_compiled_optimizer_update_disabled = true;
      compiled_mamba3_optimizer_update = nullptr;
      compiled_mamba3_optimizer_update_signature.clear();
      return false;
    }
  };
  if (use_compiled_step && run_compiled_fused_mamba3_adamw_optimizer_update()) {
    return;
  }
  auto preclip_grads = grads;
  OptimizerStepTransaction transaction(*this);
  const auto opt_t0 = HostClock::now();
  auto raw_gradient_nonfinite = sanitize_and_clip_gradients(grads, max_grad_norm);
  apply_optimizer_updates(grads);
  if (timing_enabled) {
    timing_opt_us = elapsed_us(opt_t0, HostClock::now());
  }
  log_submit_step_debug(*this, step_count, preclip_grads);

  std::vector<mx::array> eval_arrays;
  eval_arrays.reserve(
      1 + outputs.size() + weights.size() + adam_m.size() * 2 +
      muon_momentum.size() + muon_second_moment.size() + sgd_momentum.size());
  eval_arrays.push_back(loss);
  for (const auto& [_, output] : outputs) {
    eval_arrays.push_back(output);
  }
  collect_state_for_eval(*this, eval_arrays, false);
  transaction.append_validation_arrays(loss, grads, eval_arrays, raw_gradient_nonfinite);
  if (backward_trace) {
    backward_trace->append_evaluation_arrays(eval_arrays);
  }
  const auto eval_t0 = HostClock::now();
  mx::eval(eval_arrays);
  if (backward_trace) {
    const auto trace_summary = backward_trace->summarize_and_log();
    last_backward_trace_bad_edges = trace_summary.bad_edges;
    last_backward_trace_first_forward_bad_op =
        trace_summary.first_forward_bad_op_index;
    last_backward_trace_first_forward_bad_op_type =
        trace_summary.first_forward_bad_op_type;
    last_backward_trace_first_forward_bad_output =
        trace_summary.first_forward_bad_output_index;
    last_backward_trace_first_bad_op = trace_summary.first_bad_op_index;
    last_backward_trace_first_bad_op_type = trace_summary.first_bad_op_type;
    last_backward_trace_first_bad_input = trace_summary.first_bad_input_index;
  }
  if (timing_enabled) {
    timing_eval_us = elapsed_us(eval_t0, HostClock::now());
  }
  const bool committed = transaction.finish();
  if (!committed) {
    sanitize_skipped_step_reporting(loss, outputs, transaction.loss_was_nonfinite());
  }
  loss.detach();
  detach_output_map(outputs);
  detach_trainer_state(*this);
  report_gated_delta_timing_summary("step", step_count);
  pending_loss_ = loss;
  pending_outputs_ = std::move(outputs);
  has_pending_step_ = true;
  pending_step_index_ = step_count;
  log_timing(use_compiled_step ? "compiled-gradient+host-update" : "eager-gradient+host-update");
}

float IRTrainer::collect_loss() {
  const int timing_step =
      has_ready_step_ ? ready_step_index_ : (has_pending_step_ ? pending_step_index_ : step_count);
  const bool timing_enabled =
      program_has_fused_canonical_mamba3_block(program) &&
      should_log_mamba3_host_timing(timing_step);
  const auto collect_t0 = HostClock::now();
  auto log_collect_timing = [&](const char* state_label) {
    if (!timing_enabled) {
      return;
    }
    std::cout << "[mlx_ir] canonical Mamba3 host timing"
              << " step=" << timing_step
              << " path=collect"
              << " state=" << state_label
              << " collect_us=" << elapsed_us(collect_t0, HostClock::now())
              << std::endl;
  };
  if (has_ready_step_) {
    float loss = finalize_ready_step(*this, &last_outputs);
    log_collect_timing("ready");
    return loss;
  }
  last_outputs.clear();
  float loss = finalize_pending_step(*this, &last_outputs);
  log_collect_timing("pending");
  return loss;
}

void IRTrainer::flush() {
  if (has_ready_step_) {
    ready_loss_.item<float>();
    last_outputs = std::move(ready_outputs_);
    ready_outputs_.clear();
    has_ready_step_ = false;
    ready_step_index_ = 0;
  }
  if (has_pending_step_) {
    last_outputs = std::move(pending_outputs_);
    pending_outputs_.clear();
    pending_loss_.item<float>();
    has_pending_step_ = false;
    pending_step_index_ = 0;
  }
}

float IRTrainer::evaluate(const mx::array& tokens, const mx::array& targets) {
  flush();
  if (weights.empty()) {
    throw std::runtime_error("IR trainer has no weights");
  }
  auto effective = effective_compute_weights(weights, compute_dtype);
  auto loss = ir_interpret(program, effective, tokens, targets, false);
  mx::eval(loss);
  report_gated_delta_timing_summary("eval", step_count);
  return loss.item<float>();
}

float IRTrainer::evaluate_named(const TensorMap& inputs) {
  flush();
  if (weights.empty()) {
    throw std::runtime_error("IR trainer has no weights");
  }
  auto loss_name = evaluation_loss_name(program);
  auto output_names = collect_cached_output_names(program, loss_name);
  auto effective = effective_compute_weights(weights, compute_dtype);
  last_outputs = ir_interpret_outputs(program, effective, inputs, output_names);
  auto loss = last_outputs.at(loss_name);
  mx::eval(loss);
  report_gated_delta_timing_summary("eval", step_count);
  return loss.item<float>();
}

float IRTrainer::evaluate_named_with_outputs(
    const TensorMap& inputs,
    const std::vector<std::string>& extra_output_names) {
  flush();
  if (weights.empty()) {
    throw std::runtime_error("IR trainer has no weights");
  }
  auto loss_name = evaluation_loss_name(program);
  std::vector<std::string> output_names{loss_name};
  for (const auto& name : extra_output_names) {
    if (name.empty()) {
      throw std::runtime_error("requested output name is empty");
    }
    if (std::find(output_names.begin(), output_names.end(), name) == output_names.end()) {
      output_names.push_back(name);
    }
  }

  const auto input_names = sorted_input_names(inputs);
  auto input_arrays_by_name = tensor_map_to_arrays(inputs);
  std::vector<mx::array> args;
  args.reserve(weights.size() + input_names.size());
  args.insert(args.end(), weights.begin(), weights.end());
  for (const auto& name : input_names) {
    args.push_back(input_arrays_by_name.at(name));
  }
  const auto signature =
      "named_eval|" + named_step_signature(
          program,
          QATMode::None,
          compute_dtype,
          inputs,
          input_names,
          output_names,
          weights.size());

  if (!compiled_named_eval || compiled_named_eval_signature != signature) {
    auto cached = compiled_named_eval_cache.find(signature);
    if (cached != compiled_named_eval_cache.end()) {
      compiled_named_eval_cache_hits++;
      compiled_named_eval = cached->second;
    } else {
      compiled_named_eval_cache_misses++;
      const auto local_input_names = input_names;
      const auto local_output_names = output_names;
      const auto local_weight_indices = required_weight_indices_for_outputs(
          program,
          output_names,
          weights.size());
      compiled_named_eval = mx::compile(
          [this, local_input_names, local_output_names, local_weight_indices](
              const std::vector<mx::array>& fn_args) {
            const auto n_weights = weights.size();
            if (fn_args.size() != n_weights + local_input_names.size()) {
              throw std::runtime_error("named evaluator argument count mismatch");
            }
            std::vector<mx::array> w;
            w.reserve(n_weights);
            for (size_t i = 0; i < n_weights; ++i) {
              w.push_back(fn_args[i]);
            }
            ArrayMap input_map;
            input_map.reserve(local_input_names.size());
            for (size_t i = 0; i < local_input_names.size(); ++i) {
              input_map.emplace(local_input_names[i], fn_args[n_weights + i]);
            }
            auto effective = effective_compute_weights(
                w,
                compute_dtype,
                local_weight_indices);
            auto outputs = ir_interpret_outputs(
                program,
                effective,
                input_map,
                local_output_names,
                false);
            std::vector<mx::array> values;
            values.reserve(local_output_names.size());
            for (const auto& name : local_output_names) {
              values.push_back(outputs.at(name));
            }
            return values;
          },
          false);
      compiled_named_eval_cache[signature] = compiled_named_eval;
    }
    compiled_named_eval_signature = signature;
  } else {
    compiled_named_eval_cache_hits++;
  }

  auto values = compiled_named_eval(args);
  if (values.size() != output_names.size()) {
    throw std::runtime_error("compiled named evaluator output count mismatch");
  }
  last_outputs.clear();
  for (size_t i = 0; i < output_names.size(); ++i) {
    last_outputs.emplace(output_names[i], values[i]);
  }
  mx::eval(values);
  report_gated_delta_timing_summary("eval", step_count);
  return last_outputs.at(loss_name).item<float>();
}

float IRTrainer::compute_mean_square_grads_named(const TensorMap& inputs, const std::string& output_name) {
  flush();
  if (weights.empty()) {
    throw std::runtime_error("IR trainer has no weights");
  }
  if (output_name.empty()) {
    throw std::runtime_error("output name is required");
  }

  std::vector<int> argnums(weights.size());
  std::iota(argnums.begin(), argnums.end(), 0);
  std::unordered_map<std::string, mx::array> outputs;
  auto fn = mx::value_and_grad(
      [this, inputs, output_name, &outputs](const std::vector<mx::array>& w) {
        auto effective = effective_training_weights(w, qat_mode, compute_dtype);
        outputs = ir_interpret_outputs(program, effective, inputs, {output_name}, true);
        auto it = outputs.find(output_name);
        if (it == outputs.end()) {
          throw std::runtime_error("requested output missing: " + output_name);
        }
        return mx::mean(mx::square(it->second));
      },
      argnums);

  auto out = fn(weights);
  auto loss = out.first;
  last_grads = std::move(out.second);
  last_outputs = std::move(outputs);

  std::vector<mx::array> eval_arrays;
  eval_arrays.reserve(1 + last_outputs.size() + last_grads.size());
  eval_arrays.push_back(loss);
  for (const auto& [_, output] : last_outputs) {
    eval_arrays.push_back(output);
  }
  for (const auto& grad : last_grads) {
    eval_arrays.push_back(grad);
  }
  mx::eval(eval_arrays);
  return loss.item<float>();
}

std::vector<float> IRTrainer::evaluate_per_token(const TensorMap& inputs) {
  flush();
  if (weights.empty()) {
    throw std::runtime_error("IR trainer has no weights");
  }
  auto output_names = collect_cached_output_names(program, evaluation_loss_name(program));
  output_names.push_back("per_token_nll");
  auto effective = effective_compute_weights(weights, compute_dtype);
  last_outputs = ir_interpret_outputs(program, effective, inputs, output_names);
  auto nll = mx::astype(last_outputs.at("per_token_nll"), mx::float32);
  auto flat = mx::reshape(nll, {static_cast<mx::ShapeElem>(nll.size())});
  mx::eval(flat);
  std::vector<float> result(static_cast<size_t>(flat.shape(0)));
  std::memcpy(result.data(), flat.data<float>(), result.size() * sizeof(float));
  return result;
}

std::vector<int32_t> IRTrainer::sample_categorical_output(
    const TensorMap& inputs,
    const std::string& output_name,
    int rows,
    int vocab,
    float temperature,
    uint64_t seed,
    bool allow_compile) {
  flush();
  if (weights.empty()) {
    throw std::runtime_error("IR trainer has no weights");
  }
  if (output_name.empty()) {
    throw std::runtime_error("output name is required");
  }
  if (rows <= 0 || vocab <= 0 || !(temperature > 0.0f) || !std::isfinite(temperature)) {
    throw std::runtime_error("invalid categorical sampler shape or temperature");
  }

  const auto input_names = sorted_input_names(inputs);
  auto input_arrays_by_name = tensor_map_to_arrays(inputs);
  std::vector<mx::array> ordered_input_arrays;
  ordered_input_arrays.reserve(input_names.size());
  for (const auto& name : input_names) {
    ordered_input_arrays.push_back(input_arrays_by_name.at(name));
  }
  auto key = mx::random::key(seed);
  std::vector<mx::array> args;
  args.reserve(weights.size() + ordered_input_arrays.size() + 1);
  args.insert(args.end(), weights.begin(), weights.end());
  args.insert(args.end(), ordered_input_arrays.begin(), ordered_input_arrays.end());
  args.push_back(key);

  const auto signature = categorical_sampler_signature(
      program,
      compute_dtype,
      inputs,
      input_names,
      output_name,
      rows,
      vocab,
      temperature,
      weights.size());
  const auto sampler_weight_indices = required_weight_indices_for_outputs(
      program,
      std::vector<std::string>{output_name},
      weights.size());

  auto run_eager_sampler = [&]() {
    auto effective = effective_compute_weights(
        weights,
        compute_dtype,
        sampler_weight_indices);
    auto logits = mx::astype(ir_interpret(program, effective, inputs, output_name, false), mx::float32);
    try {
      return sample_categorical_logits(logits, rows, vocab, temperature, key);
    } catch (const std::exception& e) {
      throw std::runtime_error(
          std::string(e.what()) + " for " + output_name);
    }
  };

  mx::array sampled = mx::array(0, mx::int32);
  if (allow_compile &&
      !compiled_categorical_sampler_disabled &&
      !env_truthy("MIXLAB_DISABLE_COMPILED_CATEGORICAL_SAMPLER")) {
    try {
      if (!compiled_categorical_sampler ||
          compiled_categorical_sampler_signature != signature) {
        auto cached = compiled_categorical_sampler_cache.find(signature);
        if (cached != compiled_categorical_sampler_cache.end()) {
          compiled_categorical_sampler_cache_hits++;
          log_compile_cache_event(
              "categorical_sampler",
              "hit",
              step_count,
              compiled_categorical_sampler_cache.size(),
              signature);
          compiled_categorical_sampler = cached->second;
        } else {
          compiled_categorical_sampler_cache_misses++;
          log_compile_cache_event(
              "categorical_sampler",
              "miss",
              step_count,
              compiled_categorical_sampler_cache.size(),
              signature);
          const auto local_input_names = input_names;
          const auto local_output_name = output_name;
          const auto local_sampler_weight_indices = sampler_weight_indices;
          compiled_categorical_sampler = mx::compile(
              [this, local_input_names, local_output_name, local_sampler_weight_indices, rows, vocab, temperature](
                  const std::vector<mx::array>& fn_args) {
                const auto n_weights = weights.size();
                const auto n_inputs = local_input_names.size();
                if (fn_args.size() != n_weights + n_inputs + 1) {
                  throw std::runtime_error("categorical sampler argument count mismatch");
                }
                std::vector<mx::array> w;
                w.reserve(n_weights);
                for (size_t i = 0; i < n_weights; ++i) {
                  w.push_back(fn_args[i]);
                }
                ArrayMap input_map;
                input_map.reserve(n_inputs);
                for (size_t i = 0; i < n_inputs; ++i) {
                  input_map.emplace(local_input_names[i], fn_args[n_weights + i]);
                }
                auto effective = effective_compute_weights(
                    w,
                    compute_dtype,
                    local_sampler_weight_indices);
                auto logits = mx::astype(
                    ir_interpret(program, effective, input_map, local_output_name, false),
                    mx::float32);
                auto sampled = sample_categorical_logits(
                    logits,
                    rows,
                    vocab,
                    temperature,
                    fn_args[n_weights + n_inputs]);
                return std::vector<mx::array>{sampled};
              },
              false);
          compiled_categorical_sampler_cache[signature] = compiled_categorical_sampler;
        }
        compiled_categorical_sampler_signature = signature;
      } else {
        compiled_categorical_sampler_cache_hits++;
      }
      auto out = compiled_categorical_sampler(args);
      if (out.size() != 1) {
        throw std::runtime_error("compiled categorical sampler output count mismatch");
      }
      sampled = out[0];
    } catch (const std::exception& e) {
      compiled_categorical_sampler_disabled = true;
      compiled_categorical_sampler = nullptr;
      compiled_categorical_sampler_signature.clear();
      if (!compiled_categorical_sampler_fallback_logged) {
        std::cerr << "[mlx_ir] compiled categorical sampler failed ("
                  << e.what()
                  << "); falling back to eager sampler"
                  << " (set MIXLAB_DISABLE_COMPILED_CATEGORICAL_SAMPLER=1 to skip compiled retry)"
                  << std::endl;
        compiled_categorical_sampler_fallback_logged = true;
      }
      sampled = run_eager_sampler();
    }
  } else {
    sampled = run_eager_sampler();
  }

  mx::eval(sampled);
  std::vector<int32_t> result(static_cast<size_t>(rows));
  std::memcpy(result.data(), sampled.data<int32_t>(), result.size() * sizeof(int32_t));
  return result;
}

float IRTrainer::evaluate_lora_named(const TensorMap& inputs, int rank, int steps, float lr) {
  flush();
  if (weights.empty()) {
    throw std::runtime_error("IR trainer has no weights");
  }
  if (rank <= 0) {
    throw std::runtime_error("LoRA rank must be > 0");
  }
  if (steps < 0) {
    throw std::runtime_error("LoRA TTT steps must be >= 0");
  }

  std::vector<LoRAAdapterState> adapters(weights.size());
  std::vector<size_t> adapter_indices;
  adapter_indices.reserve(weights.size());
  for (size_t i = 0; i < weights.size(); ++i) {
    const auto& w = weights[i];
    if (w.ndim() != 2) {
      continue;
    }
    if (i >= weight_optimizers.size()) {
      throw std::runtime_error("missing weight optimizer spec");
    }
    const auto& spec = weight_optimizers[i];
    if (spec.group_index >= optimizer_groups.size()) {
      throw std::runtime_error("weight optimizer group index out of range");
    }
    const auto rows = static_cast<mx::ShapeElem>(w.shape(0));
    const auto cols = static_cast<mx::ShapeElem>(w.shape(1));
    auto& adapter = adapters[i];
    adapter.enabled = true;
    adapter.spec = spec;
    adapter.group = optimizer_groups[spec.group_index];
    adapter.a = mx::zeros(mx::Shape{rows, static_cast<mx::ShapeElem>(rank)}, mx::float32);
    adapter.b = mx::random::normal(mx::Shape{static_cast<mx::ShapeElem>(rank), cols}, mx::float32, 0.0f, 0.01f);
    init_lora_optimizer_state(adapter.a, adapter.group, adapter.a_state);
    init_lora_optimizer_state(adapter.b, adapter.group, adapter.b_state);
    adapter_indices.push_back(i);
  }
  if (adapter_indices.empty()) {
    return evaluate_named(inputs);
  }

  const float local_lr_scale = default_base_lr > 0.0f ? (lr / default_base_lr) : 1.0f;
  const auto loss_name = evaluation_loss_name(program);
  for (int local_step = 0; local_step < steps; ++local_step) {
    std::vector<int> argnums;
    argnums.reserve(adapter_indices.size() * 2);
    for (size_t i = 0; i < adapter_indices.size()*2; ++i) {
      argnums.push_back(static_cast<int>(i));
    }
    auto fn = mx::value_and_grad(
        [this, inputs, &adapters, &adapter_indices, loss_name](const std::vector<mx::array>& params) {
          auto local_adapters = adapters;
          size_t param_idx = 0;
          for (size_t weight_idx : adapter_indices) {
            local_adapters[weight_idx].a = params[param_idx++];
            local_adapters[weight_idx].b = params[param_idx++];
          }
          auto effective = effective_compute_weights(
              effective_lora_weights(weights, local_adapters),
              compute_dtype);
          return ir_interpret(program, effective, inputs, loss_name, true);
        },
        argnums);

    std::vector<mx::array> params;
    params.reserve(adapter_indices.size() * 2);
    for (size_t weight_idx : adapter_indices) {
      params.push_back(adapters[weight_idx].a);
      params.push_back(adapters[weight_idx].b);
    }
    auto out = fn(params);
    auto grads = std::move(out.second);
    sanitize_and_clip_gradients(grads, max_grad_norm);

    size_t grad_idx = 0;
    for (size_t weight_idx : adapter_indices) {
      auto& adapter = adapters[weight_idx];
      apply_optimizer_update(adapter.a, grads[grad_idx++], adapter.group, false, local_step+1, local_lr_scale, adapter.a_state);
      apply_optimizer_update(adapter.b, grads[grad_idx++], adapter.group, false, local_step+1, local_lr_scale, adapter.b_state);
    }

    std::vector<mx::array> eval_arrays;
    eval_arrays.reserve(adapter_indices.size() * 8);
    for (size_t weight_idx : adapter_indices) {
      const auto& adapter = adapters[weight_idx];
      eval_arrays.push_back(adapter.a);
      eval_arrays.push_back(adapter.b);
      if (adapter.a_state.has_adam_state != 0) {
        eval_arrays.push_back(adapter.a_state.adam_m);
        eval_arrays.push_back(adapter.a_state.adam_v);
        eval_arrays.push_back(adapter.b_state.adam_m);
        eval_arrays.push_back(adapter.b_state.adam_v);
      }
      if (adapter.a_state.has_muon_state != 0) {
        eval_arrays.push_back(adapter.a_state.muon_momentum);
        eval_arrays.push_back(adapter.b_state.muon_momentum);
      }
      if (adapter.a_state.has_muon_second_moment_state != 0) {
        eval_arrays.push_back(adapter.a_state.muon_second_moment);
        eval_arrays.push_back(adapter.b_state.muon_second_moment);
      }
    }
    if (!eval_arrays.empty()) {
      mx::eval(eval_arrays);
    }
  }

  auto output_names = collect_cached_output_names(program, loss_name);
  auto effective = effective_compute_weights(effective_lora_weights(weights, adapters), compute_dtype);
  last_outputs = ir_interpret_outputs(program, effective, inputs, output_names);
  auto loss = last_outputs.at(loss_name);
  std::vector<mx::array> eval_arrays;
  eval_arrays.reserve(1 + last_outputs.size());
  eval_arrays.push_back(loss);
  for (const auto& [_, output] : last_outputs) {
    eval_arrays.push_back(output);
  }
  mx::eval(eval_arrays);
  return loss.item<float>();
}

mx::array IRTrainer::read_output(const std::string& output_name) const {
  auto it = last_outputs.find(output_name);
  if (it == last_outputs.end()) {
    throw std::runtime_error("IR trainer has no cached output: " + output_name);
  }
  return it->second;
}

mx::array IRTrainer::read_grad(int weight_idx) const {
  if (weight_idx < 0 || static_cast<size_t>(weight_idx) >= last_grads.size()) {
    throw std::runtime_error("invalid gradient weight index");
  }
  return last_grads[static_cast<size_t>(weight_idx)];
}

void IRTrainer::set_training_step_extra_output_names(const std::vector<std::string>& output_names) {
  if (has_pending_step_ || has_ready_step_) {
    throw std::runtime_error("cannot change training step outputs while a submitted step is pending");
  }
  training_step_extra_output_names = output_names;
  cached_named_step_metadata_valid = false;
  cached_named_step_argnums.clear();
  cached_named_step_output_names.clear();
  cached_named_step_input_names.clear();
  cached_named_step_input_dtypes.clear();
  cached_named_step_input_shapes.clear();
  cached_named_step_signature.clear();
  compiled_named_step = nullptr;
  compiled_named_step_signature.clear();
  compiled_named_update_step = nullptr;
  compiled_named_update_step_signature.clear();
}

void IRTrainer::set_program(const IRProgram& new_program) {
  if (new_program.n_weights != static_cast<int>(weights.size())) {
    throw std::runtime_error(
        "new IR program weight count " + std::to_string(new_program.n_weights) +
        " does not match trainer weights " + std::to_string(weights.size()));
  }
  if (has_pending_step_ || has_ready_step_) {
    throw std::runtime_error("cannot switch IR program while a submitted step is pending");
  }
  program = new_program;
  last_outputs.clear();
  last_grads.clear();
  cached_named_step_metadata_valid = false;
  cached_named_step_argnums.clear();
  cached_named_step_output_names.clear();
  cached_named_step_input_names.clear();
  cached_named_step_input_dtypes.clear();
  cached_named_step_input_shapes.clear();
  cached_named_step_signature.clear();
  compiled_named_step = nullptr;
  compiled_named_step_signature.clear();
  compiled_named_eval = nullptr;
  compiled_named_eval_signature.clear();
  compiled_categorical_sampler = nullptr;
  compiled_categorical_sampler_signature.clear();
  compiled_named_update_step = nullptr;
  compiled_named_update_step_signature.clear();
  compiled_mamba3_optimizer_update = nullptr;
  compiled_mamba3_optimizer_update_signature.clear();
  compiled_mamba3_grad_chunks.clear();
  compiled_mamba3_grad_chunks_signature.clear();
  compiled_mamba3_grad_chunks_disabled = false;
  compiled_mamba3_grad_chunks_fallback_logged = false;
  adaptive_mamba3_grad_chunk_elements = 0;
  adaptive_mamba3_grad_chunk_fallback_logged = false;
  fused_mamba3_compiled_optimizer_update_disabled = false;
  fused_mamba3_compiled_optimizer_update_notice_logged = false;
  fused_mamba3_compiled_optimizer_update_fallback_logged = false;
  fused_mamba3_compiled_update_step_disabled = false;
  fused_mamba3_compiled_update_step_notice_logged = false;
  fused_mamba3_compiled_update_step_fallback_logged = false;
  fused_mamba3_compiled_step_disabled = false;
  fused_mamba3_compiled_step_notice_logged = false;
  fused_mamba3_compiled_step_fallback_logged = false;
  mamba3_single_backward_disabled = false;
  mamba3_single_backward_fallback_logged = false;
}

std::unique_ptr<IRTrainer> create_ir_trainer(
    const IRProgram& program,
    const std::vector<mx::array>& initial_weights,
    const std::vector<WeightOptimizerSpec>& weight_specs,
    const std::vector<OptimizerGroupConfig>& groups,
    float max_grad_norm,
    float default_base_lr,
    int compute_dtype) {
  if (initial_weights.empty()) {
    throw std::runtime_error("IR trainer requires at least one weight");
  }
  if (weight_specs.size() != initial_weights.size()) {
    throw std::runtime_error("IR optimizer spec count must match weights");
  }
  if (groups.empty()) {
    throw std::runtime_error("IR trainer requires at least one optimizer group");
  }

  auto trainer = std::make_unique<IRTrainer>();
  switch (compute_dtype) {
    case 0:
      trainer->compute_dtype = ComputeDType::Float32;
      break;
    case 1:
      trainer->compute_dtype = ComputeDType::BFloat16;
      break;
    default:
      throw std::runtime_error("unsupported IR trainer compute dtype");
  }
  trainer->program = program;
  trainer->weights = initial_weights;
  trainer->optimizer_groups = groups;
  trainer->weight_optimizers = weight_specs;
  trainer->adam_m.reserve(initial_weights.size());
  trainer->adam_v.reserve(initial_weights.size());
  trainer->has_adam_state.reserve(initial_weights.size());
  trainer->muon_momentum.reserve(initial_weights.size());
  trainer->has_muon_state.reserve(initial_weights.size());
  trainer->muon_second_moment.reserve(initial_weights.size());
  trainer->has_muon_second_moment_state.reserve(initial_weights.size());
  trainer->sgd_momentum.reserve(initial_weights.size());
  trainer->has_sgd_state.reserve(initial_weights.size());
  for (size_t i = 0; i < initial_weights.size(); ++i) {
    const auto& spec = weight_specs[i];
    if (spec.group_index >= groups.size()) {
      throw std::runtime_error("IR optimizer group index out of range");
    }
    const auto& group = groups[spec.group_index];
    switch (group.kind) {
      case OptimizerKind::AdamW:
      case OptimizerKind::Lamb:
        trainer->adam_m.push_back(mx::zeros_like(initial_weights[i]));
        trainer->adam_v.push_back(mx::zeros_like(initial_weights[i]));
        trainer->has_adam_state.push_back(1);
        trainer->muon_momentum.push_back(mx::array(0.0f, mx::float32));
        trainer->has_muon_state.push_back(0);
        trainer->muon_second_moment.push_back(mx::array(0.0f, mx::float32));
        trainer->has_muon_second_moment_state.push_back(0);
        trainer->sgd_momentum.push_back(mx::array(0.0f, mx::float32));
        trainer->has_sgd_state.push_back(0);
        break;
      case OptimizerKind::Muon:
        if (initial_weights[i].ndim() != 2) {
          throw std::runtime_error("Muon only supports rank-2 weights");
        }
        trainer->adam_m.push_back(mx::array(0.0f, mx::float32));
        trainer->adam_v.push_back(mx::array(0.0f, mx::float32));
        trainer->has_adam_state.push_back(0);
        trainer->muon_momentum.push_back(mx::zeros_like(initial_weights[i]));
        trainer->has_muon_state.push_back(1);
        if (group.muon_normalization == MuonNormalization::NorMuon) {
          trainer->muon_second_moment.push_back(init_normuon_second_moment(initial_weights[i]));
          trainer->has_muon_second_moment_state.push_back(1);
        } else {
          trainer->muon_second_moment.push_back(mx::array(0.0f, mx::float32));
          trainer->has_muon_second_moment_state.push_back(0);
        }
        trainer->sgd_momentum.push_back(mx::array(0.0f, mx::float32));
        trainer->has_sgd_state.push_back(0);
        break;
      case OptimizerKind::SGD:
        trainer->adam_m.push_back(mx::array(0.0f, mx::float32));
        trainer->adam_v.push_back(mx::array(0.0f, mx::float32));
        trainer->has_adam_state.push_back(0);
        trainer->muon_momentum.push_back(mx::array(0.0f, mx::float32));
        trainer->has_muon_state.push_back(0);
        trainer->muon_second_moment.push_back(mx::array(0.0f, mx::float32));
        trainer->has_muon_second_moment_state.push_back(0);
        trainer->sgd_momentum.push_back(mx::zeros_like(initial_weights[i]));
        trainer->has_sgd_state.push_back(1);
        break;
      default:
        throw std::runtime_error("unsupported optimizer kind");
    }
  }

  trainer->max_grad_norm = max_grad_norm;
  trainer->default_base_lr = default_base_lr;
  trainer->lr_scale = 1.0f;
  return trainer;
}

} // namespace mlx_ir
