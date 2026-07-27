#include "distributed_step.h"

#include "optimizer_step_guard.h"

#include <cmath>
#include <stdexcept>

namespace mx = mlx::core;

namespace mlx_ir {

namespace {

mx::array nonfinite_count(const std::vector<mx::array>& values) {
  auto any_nonfinite = mx::array(false);
  for (const auto& value : values) {
    any_nonfinite = mx::logical_or(
        any_nonfinite,
        mx::any(mx::logical_not(mx::isfinite(mx::astype(value, mx::float32)))));
  }
  return mx::astype(any_nonfinite, mx::float32);
}

} // namespace

StagedGradientResult prepare_singleton_staged_gradients(
    std::vector<mx::array>& gradients,
    float loss_normalizer,
    float max_grad_norm,
    std::vector<std::string>& stage_trace) {
  if (!std::isfinite(loss_normalizer) || loss_normalizer < 0.0f) {
    throw std::runtime_error("distributed loss_normalizer must be finite and non-negative");
  }
  stage_trace.push_back("numerator_conversion");
  if (loss_normalizer == 0.0f) {
    return {
        mx::array(0.0f, mx::float32),
        true,
    };
  }
  for (auto& gradient : gradients) {
    gradient = mx::astype(gradient, mx::float32) * loss_normalizer;
  }

  stage_trace.push_back("pre_update_finite");
  auto pre_update_nonfinite = nonfinite_count(gradients);

  // Phase 1 intentionally exercises the reduction boundary with a singleton
  // context. Ordered all-sum and global finite decisions land in Phase 2.
  stage_trace.push_back("reduce");

  stage_trace.push_back("divide");
  for (auto& gradient : gradients) {
    gradient = gradient / loss_normalizer;
  }

  stage_trace.push_back("clip");
  auto post_divide_nonfinite = sanitize_and_clip_gradients(gradients, max_grad_norm);
  return {
      pre_update_nonfinite + post_divide_nonfinite,
      false,
  };
}

} // namespace mlx_ir
