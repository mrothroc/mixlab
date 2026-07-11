#ifndef MLX_OPTIMIZER_STEP_GUARD_H
#define MLX_OPTIMIZER_STEP_GUARD_H

#include "ir_trainer.h"

#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace mlx_ir {

constexpr uint64_t kMaxConsecutiveSkippedOptimizerSteps = 3;

class OptimizerStepCircuitBreaker : public std::runtime_error {
 public:
  explicit OptimizerStepCircuitBreaker(const std::string& message)
      : std::runtime_error(message) {}
};

// Converts gradients to float32, replaces non-finite values with zero, and
// clips only when every raw gradient is finite. The returned scalar counts raw
// non-finite values so the transaction can reject the candidate update even
// though the arrays passed to the optimizer are safe to evaluate.
mlx::core::array sanitize_and_clip_gradients(
    std::vector<mlx::core::array>& gradients,
    float max_grad_norm);

// OptimizerStepTransaction owns the atomicity boundary for one persistent
// full-model optimizer update. Optimizer implementations build candidate
// arrays normally; this guard commits them only when the loss, gradients, and
// complete candidate state are finite.
class OptimizerStepTransaction {
 public:
  explicit OptimizerStepTransaction(IRTrainer& trainer);
  ~OptimizerStepTransaction();

  OptimizerStepTransaction(const OptimizerStepTransaction&) = delete;
  OptimizerStepTransaction& operator=(const OptimizerStepTransaction&) = delete;

  void append_validation_arrays(
      const mlx::core::array& loss,
      const std::vector<mlx::core::array>& gradients,
      std::vector<mlx::core::array>& eval_arrays,
      const mlx::core::array& known_gradient_nonfinite);
  bool finish();
  bool loss_was_nonfinite() const;

 private:
  void restore();

  IRTrainer& trainer_;
  std::vector<mlx::core::array> weights_;
  std::vector<mlx::core::array> adam_m_;
  std::vector<mlx::core::array> adam_v_;
  std::vector<mlx::core::array> muon_momentum_;
  std::vector<mlx::core::array> muon_second_moment_;
  std::vector<mlx::core::array> sgd_momentum_;
  mlx::core::array loss_nonfinite_count_;
  mlx::core::array gradient_nonfinite_count_;
  mlx::core::array state_nonfinite_count_;
  bool validation_prepared_ = false;
  bool finalized_ = false;
  bool loss_was_nonfinite_ = false;
};

void sanitize_skipped_step_reporting(
    mlx::core::array& loss,
    std::unordered_map<std::string, mlx::core::array>& outputs,
    bool loss_was_nonfinite);

} // namespace mlx_ir

#endif
