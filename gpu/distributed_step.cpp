#include "distributed_step.h"

#include "optimizer_step_guard.h"

#include <mlx/distributed/ops.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace mx = mlx::core;

namespace mlx_ir {

namespace {

mx::array any_nonfinite(const std::vector<mx::array>& values) {
  auto any_nonfinite = mx::array(false);
  for (const auto& value : values) {
    any_nonfinite = mx::logical_or(
        any_nonfinite,
        mx::any(mx::logical_not(mx::isfinite(mx::astype(value, mx::float32)))));
  }
  return mx::astype(any_nonfinite, mx::float32);
}

void hash_word(uint64_t& hash, uint64_t value) {
  constexpr uint64_t kFNVPrime = 1099511628211ull;
  for (int i = 0; i < 8; ++i) {
    hash ^= static_cast<uint8_t>(value >> (i * 8));
    hash *= kFNVPrime;
  }
}

void restore_bucket_views(
    std::vector<mx::array>& gradients,
    const GradientBucket& bucket,
    const mx::array& flat) {
  for (const auto& entry : bucket.entries) {
    auto view = mx::slice(
        flat,
        {static_cast<int>(entry.offset)},
        {static_cast<int>(entry.offset + entry.elements)});
    gradients[entry.weight_index] = mx::reshape(view, entry.shape);
  }
}

} // namespace

GradientBucketPlan build_gradient_bucket_plan(
    const std::vector<mx::array>& values,
    size_t target_bytes) {
  if (target_bytes == 0) {
    throw std::runtime_error("gradient bucket target must be positive");
  }
  GradientBucketPlan plan;
  plan.target_bytes = target_bytes;
  plan.digest = 1469598103934665603ull;
  GradientBucket current;
  for (size_t i = 0; i < values.size(); ++i) {
    const size_t elements = values[i].size();
    if (elements > static_cast<size_t>(std::numeric_limits<int>::max())) {
      throw std::runtime_error("gradient bucket entry exceeds MLX slice index range");
    }
    const size_t bytes = elements * sizeof(float);
    const bool exceeds_target =
        current.bytes > target_bytes ||
        bytes > target_bytes - std::min(current.bytes, target_bytes);
    const bool exceeds_slice_range =
        current.elements >
        static_cast<size_t>(std::numeric_limits<int>::max()) - elements;
    if (!current.entries.empty() && (exceeds_target || exceeds_slice_range)) {
      plan.buckets.push_back(std::move(current));
      current = GradientBucket{};
    }
    GradientBucketEntry entry;
    entry.weight_index = i;
    entry.offset = current.elements;
    entry.elements = elements;
    entry.shape = values[i].shape();
    current.entries.push_back(std::move(entry));
    current.elements += elements;
    current.bytes += bytes;
    plan.total_bytes += bytes;
  }
  if (!current.entries.empty()) {
    plan.buckets.push_back(std::move(current));
  }

  hash_word(plan.digest, plan.target_bytes);
  hash_word(plan.digest, plan.total_bytes);
  hash_word(plan.digest, plan.buckets.size());
  for (const auto& bucket : plan.buckets) {
    hash_word(plan.digest, bucket.bytes);
    hash_word(plan.digest, bucket.entries.size());
    for (const auto& entry : bucket.entries) {
      hash_word(plan.digest, entry.weight_index);
      hash_word(plan.digest, entry.offset);
      hash_word(plan.digest, entry.elements);
      hash_word(plan.digest, entry.shape.size());
      for (int dim : entry.shape) {
        hash_word(plan.digest, static_cast<uint64_t>(dim));
      }
    }
  }
  return plan;
}

StagedGradientResult prepare_distributed_staged_gradients(
    std::vector<mx::array>& gradients,
    const mx::array& loss,
    float loss_normalizer,
    float max_grad_norm,
    const mx::distributed::Group& group,
    const GradientBucketPlan& bucket_plan,
    std::vector<std::string>& stage_trace) {
  if (gradients.empty() || bucket_plan.buckets.empty()) {
    throw std::runtime_error("distributed gradient bucket plan is empty");
  }
  const bool denominator_valid =
      std::isfinite(loss_normalizer) && loss_normalizer >= 0.0f;
  const float safe_normalizer = denominator_valid ? loss_normalizer : 0.0f;

  stage_trace.push_back("numerator_conversion");
  auto raw_gradient_bad = mx::array(0.0f, mx::float32);
  for (auto& gradient : gradients) {
    auto value = mx::astype(gradient, mx::float32);
    auto finite = mx::isfinite(value);
    raw_gradient_bad = mx::maximum(
        raw_gradient_bad,
        mx::astype(mx::any(mx::logical_not(finite)), mx::float32));
    gradient = mx::where(finite, value, mx::zeros_like(value)) *
        safe_normalizer;
  }

  stage_trace.push_back("pre_update_finite");
  auto local_bad = safe_normalizer > 0.0f
      ? mx::maximum(any_nonfinite({loss}), raw_gradient_bad)
      : mx::array(0.0f, mx::float32);
  if (!denominator_valid) {
    local_bad = mx::array(1.0f, mx::float32);
  }
  stage_trace.push_back("all_max_pre_update_bad");
  auto global_bad = mx::distributed::all_max(local_bad, group);
  mx::eval(global_bad);
  if (global_bad.item<float>() > 0.0f) {
    return {
        global_bad,
        false,
        true,
    };
  }

  stage_trace.push_back("all_sum_denominator");
  auto global_denominator = mx::distributed::all_sum(
      mx::array(safe_normalizer, mx::float32),
      group);
  mx::eval(global_denominator);
  const float denominator = global_denominator.item<float>();
  if (!std::isfinite(denominator) || denominator < 0.0f) {
    throw std::runtime_error("distributed global loss denominator is invalid");
  }
  if (denominator == 0.0f) {
    return {
        mx::array(0.0f, mx::float32),
        true,
        false,
    };
  }

  std::vector<mx::array> reduced_buckets;
  reduced_buckets.reserve(bucket_plan.buckets.size());
  for (size_t bucket_index = 0; bucket_index < bucket_plan.buckets.size(); ++bucket_index) {
    const auto& bucket = bucket_plan.buckets[bucket_index];
    std::vector<mx::array> flat_entries;
    flat_entries.reserve(bucket.entries.size());
    for (const auto& entry : bucket.entries) {
      flat_entries.push_back(mx::reshape(
          gradients[entry.weight_index],
          {static_cast<int>(entry.elements)}));
    }
    auto flat = flat_entries.size() == 1
        ? flat_entries.front()
        : mx::concatenate(flat_entries, 0);
    stage_trace.push_back("all_sum_bucket_" + std::to_string(bucket_index));
    reduced_buckets.push_back(mx::distributed::all_sum(flat, group));
  }
  mx::eval(reduced_buckets);
  for (size_t bucket_index = 0; bucket_index < bucket_plan.buckets.size(); ++bucket_index) {
    restore_bucket_views(
        gradients,
        bucket_plan.buckets[bucket_index],
        reduced_buckets[bucket_index] / denominator);
  }

  stage_trace.push_back("clip");
  auto post_reduce_nonfinite = sanitize_and_clip_gradients(gradients, max_grad_norm);
  return {
      post_reduce_nonfinite,
      false,
      false,
  };
}

} // namespace mlx_ir
