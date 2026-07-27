#ifndef MIXLAB_DISTRIBUTED_STEP_H
#define MIXLAB_DISTRIBUTED_STEP_H

#include <mlx/mlx.h>

#include <string>
#include <vector>

namespace mlx_ir {

struct StagedGradientResult {
  mlx::core::array raw_gradient_nonfinite;
  bool zero_denominator = false;
};

// Phase 1 uses the complete numerator/denominator stage boundary with a
// singleton reduction. Phase 2 replaces the reduction stage with ordered
// group collectives without changing the optimizer boundary.
StagedGradientResult prepare_singleton_staged_gradients(
    std::vector<mlx::core::array>& gradients,
    float loss_normalizer,
    float max_grad_norm,
    std::vector<std::string>& stage_trace);

} // namespace mlx_ir

#endif
