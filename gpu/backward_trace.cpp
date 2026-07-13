#include "backward_trace.h"

#include <mlx/transforms.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace mx = mlx::core;

namespace mlx_ir {
namespace {

thread_local BackwardTraceCollector* current_collector = nullptr;

bool env_truthy(const char* name) {
  const char* raw = std::getenv(name);
  return raw != nullptr && raw[0] != '\0' &&
      std::strcmp(raw, "0") != 0 &&
      std::strcmp(raw, "false") != 0 &&
      std::strcmp(raw, "FALSE") != 0;
}

int env_int(const char* name, int fallback) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') {
    return fallback;
  }
  char* end = nullptr;
  const long parsed = std::strtol(raw, &end, 10);
  if (end == raw || *end != '\0' || parsed < 0 || parsed > std::numeric_limits<int>::max()) {
    throw std::runtime_error(std::string("invalid ") + name + " value: " + raw);
  }
  return static_cast<int>(parsed);
}

bool differentiable_dtype(const mx::array& value) {
  return mx::issubdtype(value.dtype(), mx::floating);
}

std::pair<mx::array, mx::array> finite_stats(const mx::array& value) {
  auto as_f32 = mx::astype(value, mx::float32);
  auto finite = mx::isfinite(as_f32);
  auto bad = mx::sum(mx::astype(mx::logical_not(finite), mx::float32));
  auto finite_abs = mx::where(finite, mx::abs(as_f32), mx::zeros_like(as_f32));
  return {std::move(bad), mx::max(finite_abs)};
}

} // namespace

BackwardTraceCollector::BackwardTraceCollector(int training_step)
    : training_step_(training_step) {}

mx::array BackwardTraceCollector::wrap_input(
    const mx::array& value,
    size_t op_index,
    int op_type,
    int input_index,
    std::string input_name) {
  if (!differentiable_dtype(value)) {
    return value;
  }
  auto [forward_bad, forward_max] = finite_stats(value);
  auto identity = mx::custom_vjp(
      [](const std::vector<mx::array>& args) {
        return std::vector<mx::array>{args[0]};
      },
      [this, op_index, op_type, input_index, input_name = std::move(input_name),
       forward_bad, forward_max](
          const std::vector<mx::array>& args,
          const std::vector<mx::array>& cotangents,
          const std::vector<mx::array>& outputs) {
        (void)args;
        (void)outputs;
        auto [bad, max_abs] = finite_stats(cotangents[0]);
        records_.push_back(Record{
            false,
            op_index,
            op_type,
            input_index,
            input_name,
            forward_bad,
            forward_max,
            std::move(bad),
            std::move(max_abs),
        });
        return std::vector<mx::array>{cotangents[0]};
      });
  return identity({value})[0];
}

mx::array BackwardTraceCollector::wrap_output(
    const mx::array& value,
    size_t op_index,
    int op_type,
    int output_index,
    std::string output_name) {
  if (!differentiable_dtype(value)) {
    return value;
  }
  auto [forward_bad, forward_max] = finite_stats(value);
  auto identity = mx::custom_vjp(
      [](const std::vector<mx::array>& args) {
        return std::vector<mx::array>{args[0]};
      },
      [this, op_index, op_type, output_index, output_name = std::move(output_name),
       forward_bad, forward_max](
          const std::vector<mx::array>& args,
          const std::vector<mx::array>& cotangents,
          const std::vector<mx::array>& outputs) {
        (void)args;
        (void)outputs;
        auto [bad, max_abs] = finite_stats(cotangents[0]);
        records_.push_back(Record{
            true,
            op_index,
            op_type,
            output_index,
            output_name,
            forward_bad,
            forward_max,
            std::move(bad),
            std::move(max_abs),
        });
        return std::vector<mx::array>{cotangents[0]};
      });
  return identity({value})[0];
}

void BackwardTraceCollector::append_evaluation_arrays(
    std::vector<mx::array>& arrays) const {
  arrays.reserve(arrays.size() + records_.size() * 4);
  for (const auto& record : records_) {
    arrays.push_back(record.forward_nonfinite_count);
    arrays.push_back(record.forward_max_abs_finite);
    arrays.push_back(record.nonfinite_count);
    arrays.push_back(record.max_abs_finite);
  }
}

BackwardTraceSummary BackwardTraceCollector::summarize_and_log() const {
  BackwardTraceSummary summary;
  const Record* first_forward = nullptr;
  const Record* fallback = nullptr;
  const Record* first = nullptr;
  for (const auto& record : records_) {
    const float forward_bad = record.forward_nonfinite_count.item<float>();
    if (record.is_output && std::isfinite(forward_bad) && forward_bad > 0.0f &&
        (first_forward == nullptr || record.op_index < first_forward->op_index)) {
      first_forward = &record;
    }
    const float bad = record.nonfinite_count.item<float>();
    if (!std::isfinite(bad) || bad <= 0.0f) {
      continue;
    }
    summary.bad_edges++;
    if (fallback == nullptr || record.op_index > fallback->op_index) {
      fallback = &record;
    }
    if (!record.is_output) {
      bool output_cotangent_finite = true;
      for (const auto& candidate : records_) {
        if (candidate.is_output && candidate.op_index == record.op_index &&
            candidate.nonfinite_count.item<float>() > 0.0f) {
          output_cotangent_finite = false;
          break;
        }
      }
      if (output_cotangent_finite &&
          (first == nullptr || record.op_index > first->op_index)) {
        first = &record;
      }
    }
  }
  if (first == nullptr) {
    first = fallback;
  }
  if (first_forward != nullptr) {
    summary.first_forward_bad_op_index = static_cast<int>(first_forward->op_index);
    summary.first_forward_bad_op_type = first_forward->op_type;
    summary.first_forward_bad_output_index = first_forward->input_index;
    float input_bad = 0.0f;
    float input_max = 0.0f;
    for (const auto& record : records_) {
      if (!record.is_output && record.op_index == first_forward->op_index) {
        input_bad += record.forward_nonfinite_count.item<float>();
        input_max = std::max(
            input_max,
            record.forward_max_abs_finite.item<float>());
      }
    }
    std::cerr << "[mlx_ir] forward non-finite source"
              << " training_step=" << training_step_
              << " op_index=" << first_forward->op_index
              << " op=" << ir_op_type_name(first_forward->op_type)
              << " op_type=" << first_forward->op_type
              << " output_index=" << first_forward->input_index
              << " output=" << first_forward->input_name
              << " input_nonfinite=" << input_bad
              << " input_max_abs=" << input_max
              << " output_nonfinite=" << first_forward->forward_nonfinite_count.item<float>()
              << " output_max_abs_finite=" << first_forward->forward_max_abs_finite.item<float>()
              << std::endl;
  }
  if (first != nullptr) {
    summary.first_bad_op_index = static_cast<int>(first->op_index);
    summary.first_bad_op_type = first->op_type;
    summary.first_bad_input_index = first->input_index;
    float output_bad = 0.0f;
    float output_max = 0.0f;
    for (const auto& record : records_) {
      if (record.is_output && record.op_index == first->op_index) {
        output_bad += record.nonfinite_count.item<float>();
        output_max = std::max(output_max, record.max_abs_finite.item<float>());
      }
    }
    std::cerr << "[mlx_ir] backward non-finite source"
              << " training_step=" << training_step_
              << " op_index=" << first->op_index
              << " op=" << ir_op_type_name(first->op_type)
              << " op_type=" << first->op_type
              << " edge=" << (first->is_output ? "output_cotangent" : "input_gradient")
              << " edge_index=" << first->input_index
              << " edge_name=" << first->input_name
              << " forward_nonfinite=" << first->forward_nonfinite_count.item<float>()
              << " forward_max_abs=" << first->forward_max_abs_finite.item<float>()
              << " nonfinite=" << first->nonfinite_count.item<float>()
              << " max_abs_finite=" << first->max_abs_finite.item<float>()
              << " output_cotangent_nonfinite=" << output_bad
              << " output_cotangent_max_abs=" << output_max
              << " bad_edges=" << summary.bad_edges
              << std::endl;
  } else {
    std::cerr << "[mlx_ir] backward trace finite"
              << " training_step=" << training_step_
              << " traced_edges=" << records_.size()
              << std::endl;
  }
  return summary;
}

BackwardTraceScope::BackwardTraceScope(BackwardTraceCollector* collector)
    : previous_(current_collector) {
  current_collector = collector;
}

BackwardTraceScope::~BackwardTraceScope() {
  current_collector = previous_;
}

bool backward_trace_enabled_for_step(int training_step) {
  if (!env_truthy("MIXLAB_MLX_BACKWARD_TRACE")) {
    return false;
  }
  const int start = env_int("MIXLAB_MLX_BACKWARD_TRACE_START", 1);
  const int end = env_int("MIXLAB_MLX_BACKWARD_TRACE_END", start);
  if (end < start) {
    throw std::runtime_error("MIXLAB_MLX_BACKWARD_TRACE_END must be >= MIXLAB_MLX_BACKWARD_TRACE_START");
  }
  return training_step >= start && training_step <= end;
}

bool backward_trace_active() {
  return current_collector != nullptr;
}

mx::array trace_backward_input(
    const mx::array& value,
    size_t op_index,
    int op_type,
    int input_index,
    const std::string& input_name) {
  if (current_collector == nullptr) {
    return value;
  }
  return current_collector->wrap_input(value, op_index, op_type, input_index, input_name);
}

mx::array trace_backward_output(
    const mx::array& value,
    size_t op_index,
    int op_type,
    int output_index,
    const std::string& output_name) {
  if (current_collector == nullptr) {
    return value;
  }
  return current_collector->wrap_output(
      value, op_index, op_type, output_index, output_name);
}

const char* ir_op_type_name(int op_type) {
  switch (op_type) {
#define MIXLAB_OP_NAME(name) case name: return #name
    MIXLAB_OP_NAME(OP_EMBED); MIXLAB_OP_NAME(OP_MATMUL); MIXLAB_OP_NAME(OP_ADD);
    MIXLAB_OP_NAME(OP_MUL); MIXLAB_OP_NAME(OP_SCALAR_MUL); MIXLAB_OP_NAME(OP_SIGMOID);
    MIXLAB_OP_NAME(OP_SILU); MIXLAB_OP_NAME(OP_SOFTMAX); MIXLAB_OP_NAME(OP_RESHAPE);
    MIXLAB_OP_NAME(OP_TRANSPOSE); MIXLAB_OP_NAME(OP_SLICE); MIXLAB_OP_NAME(OP_CONCAT);
    MIXLAB_OP_NAME(OP_CAUSAL_MASK); MIXLAB_OP_NAME(OP_CROSS_ENTROPY); MIXLAB_OP_NAME(OP_DROPOUT);
    MIXLAB_OP_NAME(OP_SQUARE); MIXLAB_OP_NAME(OP_SUB); MIXLAB_OP_NAME(OP_DIV);
    MIXLAB_OP_NAME(OP_CUMSUM); MIXLAB_OP_NAME(OP_ARGSORT); MIXLAB_OP_NAME(OP_WHERE);
    MIXLAB_OP_NAME(OP_LESS_THAN); MIXLAB_OP_NAME(OP_GREATER_EQ); MIXLAB_OP_NAME(OP_ARANGE);
    MIXLAB_OP_NAME(OP_MEAN_AXIS); MIXLAB_OP_NAME(OP_FULL); MIXLAB_OP_NAME(OP_ASTYPE);
    MIXLAB_OP_NAME(OP_RUNNING_VAR); MIXLAB_OP_NAME(OP_RMSNORM); MIXLAB_OP_NAME(OP_ROPE);
    MIXLAB_OP_NAME(OP_SQRT); MIXLAB_OP_NAME(OP_RSQRT); MIXLAB_OP_NAME(OP_SIN);
    MIXLAB_OP_NAME(OP_COS); MIXLAB_OP_NAME(OP_EXP); MIXLAB_OP_NAME(OP_OUTER);
    MIXLAB_OP_NAME(OP_SQUEEZE); MIXLAB_OP_NAME(OP_GELU); MIXLAB_OP_NAME(OP_RELU);
    MIXLAB_OP_NAME(OP_TANH); MIXLAB_OP_NAME(OP_ROPE_STRIDED); MIXLAB_OP_NAME(OP_PREFIX_CAUSAL_MASK);
    MIXLAB_OP_NAME(OP_SCAN); MIXLAB_OP_NAME(OP_GRADIENT_MAGNITUDES); MIXLAB_OP_NAME(OP_GATHER_POSITIONS);
    MIXLAB_OP_NAME(OP_SCATTER_POSITIONS); MIXLAB_OP_NAME(OP_ROPE_INDEXED); MIXLAB_OP_NAME(OP_LEAKY_RELU);
    MIXLAB_OP_NAME(OP_XSA_PROJECT); MIXLAB_OP_NAME(OP_CROSS_ENTROPY_PER_TOKEN); MIXLAB_OP_NAME(OP_MATRIX_SCAN);
    MIXLAB_OP_NAME(OP_SCAN_TV); MIXLAB_OP_NAME(OP_SOFTPLUS); MIXLAB_OP_NAME(OP_GATED_DELTA_SCAN);
    MIXLAB_OP_NAME(OP_STOP_GRADIENT); MIXLAB_OP_NAME(OP_DEPTHWISE_CONV1D); MIXLAB_OP_NAME(OP_MAMBA3_SELECTIVE_SCAN);
    MIXLAB_OP_NAME(OP_MAMBA3_CANONICAL_BLOCK); MIXLAB_OP_NAME(OP_RANDOM_NORMAL); MIXLAB_OP_NAME(OP_FIRST_BYTE_MASKED_CROSS_ENTROPY);
    MIXLAB_OP_NAME(OP_MASKED_CROSS_ENTROPY); MIXLAB_OP_NAME(OP_MASKED_CROSS_ENTROPY_PER_TOKEN); MIXLAB_OP_NAME(OP_DISTILLATION_KL);
    MIXLAB_OP_NAME(OP_HGRN2_SCAN); MIXLAB_OP_NAME(OP_MLSTM_SCAN); MIXLAB_OP_NAME(OP_DEBERTA_RELATIVE_BIAS);
    MIXLAB_OP_NAME(OP_CHAR_FEATURE_BAG); MIXLAB_OP_NAME(OP_MOE_FEED_FORWARD); MIXLAB_OP_NAME(OP_MASKED_SMOOTH_L1);
    MIXLAB_OP_NAME(OP_Z_LOSS); MIXLAB_OP_NAME(OP_LOG); MIXLAB_OP_NAME(OP_RECIPROCAL);
    MIXLAB_OP_NAME(OP_POW); MIXLAB_OP_NAME(OP_ABS); MIXLAB_OP_NAME(OP_CLAMP);
    MIXLAB_OP_NAME(OP_MINIMUM); MIXLAB_OP_NAME(OP_MAXIMUM); MIXLAB_OP_NAME(OP_GREATER_THAN);
    MIXLAB_OP_NAME(OP_LESS_EQ); MIXLAB_OP_NAME(OP_EQUAL); MIXLAB_OP_NAME(OP_LAYERNORM);
    MIXLAB_OP_NAME(OP_SELECTIVE_CAUSAL_MASK); MIXLAB_OP_NAME(OP_SEGMENT_ATTENTION_MASK); MIXLAB_OP_NAME(OP_BLOCK_DIFFUSION_MASK);
    MIXLAB_OP_NAME(OP_GELU_EXACT); MIXLAB_OP_NAME(OP_MASKED_BCE_WITH_LOGITS); MIXLAB_OP_NAME(OP_MASKED_BINARY_ACCURACY);
    MIXLAB_OP_NAME(OP_ENERGY_PAIRWISE_LOSS); MIXLAB_OP_NAME(OP_ENERGY_SPAN_POOL); MIXLAB_OP_NAME(OP_ENERGY_SPAN_PAIRWISE_LOSS);
    MIXLAB_OP_NAME(OP_SPAN_PLL_POOL); MIXLAB_OP_NAME(OP_SPAN_PLL_PAIRWISE_LOSS); MIXLAB_OP_NAME(OP_MASKED_DISTILLATION_KL);
    MIXLAB_OP_NAME(OP_MASKED_SYMMETRIC_KL); MIXLAB_OP_NAME(OP_MASKED_MARGIN_PLL); MIXLAB_OP_NAME(OP_MASKED_Z_LOSS);
    MIXLAB_OP_NAME(OP_TTT_MLP_SCAN);
#undef MIXLAB_OP_NAME
    default: return "OP_UNKNOWN";
  }
}

} // namespace mlx_ir
