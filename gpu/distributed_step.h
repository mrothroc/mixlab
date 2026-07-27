#ifndef MIXLAB_DISTRIBUTED_STEP_H
#define MIXLAB_DISTRIBUTED_STEP_H

#include <mlx/mlx.h>
#include <mlx/distributed/distributed.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mlx_ir {

constexpr size_t kDefaultGradientBucketBytes = 32u * 1024u * 1024u;

struct GradientBucketEntry {
  size_t weight_index = 0;
  size_t offset = 0;
  size_t elements = 0;
  mlx::core::Shape shape;
};

struct GradientBucket {
  size_t elements = 0;
  size_t bytes = 0;
  std::vector<GradientBucketEntry> entries;
};

struct GradientBucketPlan {
  size_t target_bytes = kDefaultGradientBucketBytes;
  size_t total_bytes = 0;
  uint64_t digest = 0;
  std::vector<GradientBucket> buckets;
};

struct StagedGradientResult {
  mlx::core::array gradient_nonfinite;
  bool zero_denominator = false;
  bool globally_bad = false;
  uint64_t collective_us = 0;
  uint64_t wait_us = 0;
  uint64_t gradient_all_reduce_us = 0;
};

GradientBucketPlan build_gradient_bucket_plan(
    const std::vector<mlx::core::array>& values,
    size_t target_bytes = kDefaultGradientBucketBytes);

StagedGradientResult prepare_distributed_staged_gradients(
    std::vector<mlx::core::array>& gradients,
    const mlx::core::array& loss,
    float loss_normalizer,
    bool gradients_are_numerators,
    float max_grad_norm,
    const mlx::core::distributed::Group& group,
    const GradientBucketPlan& bucket_plan,
    std::vector<std::string>& stage_trace);

} // namespace mlx_ir

#endif
