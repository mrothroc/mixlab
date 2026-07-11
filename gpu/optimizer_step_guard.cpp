#include "optimizer_step_guard.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace mx = mlx::core;

namespace mlx_ir {
namespace {

mx::array nonfinite_count(const mx::array& value) {
  return mx::sum(mx::astype(mx::logical_not(mx::isfinite(value)), mx::float32));
}

void accumulate_nonfinite(mx::array& count, const std::vector<mx::array>& values) {
  for (const auto& value : values) {
    count = count + nonfinite_count(value);
  }
}

void accumulate_active_optimizer_state_nonfinite(
    mx::array& count,
    const IRTrainer& trainer) {
  accumulate_nonfinite(count, trainer.weights);
  for (size_t i = 0; i < trainer.weights.size(); ++i) {
    if (trainer.has_adam_state[i] != 0) {
      count = count + nonfinite_count(trainer.adam_m[i]);
      count = count + nonfinite_count(trainer.adam_v[i]);
    }
    if (trainer.has_muon_state[i] != 0) {
      count = count + nonfinite_count(trainer.muon_momentum[i]);
    }
    if (trainer.has_muon_second_moment_state[i] != 0) {
      count = count + nonfinite_count(trainer.muon_second_moment[i]);
    }
    if (trainer.has_sgd_state[i] != 0) {
      count = count + nonfinite_count(trainer.sgd_momentum[i]);
    }
  }
}

float evaluated_count(const mx::array& count, const char* label) {
  const float value = count.item<float>();
  if (!std::isfinite(value) || value < 0.0f) {
    throw std::runtime_error(std::string("invalid optimizer-step ") + label + " count");
  }
  return value;
}

uint64_t active_optimizer_state_elements(const IRTrainer& trainer) {
  uint64_t count = 0;
  for (size_t i = 0; i < trainer.weights.size(); ++i) {
    count += static_cast<uint64_t>(trainer.weights[i].size());
    if (trainer.has_adam_state[i] != 0) {
      count += static_cast<uint64_t>(trainer.adam_m[i].size());
      count += static_cast<uint64_t>(trainer.adam_v[i].size());
    }
    if (trainer.has_muon_state[i] != 0) {
      count += static_cast<uint64_t>(trainer.muon_momentum[i].size());
    }
    if (trainer.has_muon_second_moment_state[i] != 0) {
      count += static_cast<uint64_t>(trainer.muon_second_moment[i].size());
    }
    if (trainer.has_sgd_state[i] != 0) {
      count += static_cast<uint64_t>(trainer.sgd_momentum[i].size());
    }
  }
  return count;
}

} // namespace

mx::array sanitize_and_clip_gradients(
    std::vector<mx::array>& gradients,
    float max_grad_norm) {
  auto nonfinite = mx::array(0.0f, mx::float32);
  auto norm_sq = mx::array(0.0f, mx::float32);
  for (auto& gradient : gradients) {
    auto value = mx::astype(gradient, mx::float32);
    auto finite = mx::isfinite(value);
    nonfinite = nonfinite + mx::sum(mx::astype(mx::logical_not(finite), mx::float32));
    gradient = mx::where(finite, value, mx::zeros_like(value));
    norm_sq = norm_sq + mx::sum(mx::square(gradient));
  }
  if (max_grad_norm <= 0.0f) {
    return nonfinite;
  }
  auto norm = mx::sqrt(norm_sq);
  auto ordinary_scale = mx::minimum(
      mx::array(1.0f, mx::float32),
      mx::array(max_grad_norm, mx::float32) /
          (norm + mx::array(1e-6f, mx::float32)));
  auto valid = mx::logical_and(
      mx::equal(nonfinite, mx::array(0.0f, mx::float32)),
      mx::isfinite(norm));
  auto scale = mx::where(valid, ordinary_scale, mx::array(0.0f, mx::float32));
  for (auto& gradient : gradients) {
    gradient = gradient * scale;
  }
  return nonfinite;
}

OptimizerStepTransaction::OptimizerStepTransaction(IRTrainer& trainer)
    : trainer_(trainer),
      weights_(trainer.weights),
      adam_m_(trainer.adam_m),
      adam_v_(trainer.adam_v),
      muon_momentum_(trainer.muon_momentum),
      muon_second_moment_(trainer.muon_second_moment),
      sgd_momentum_(trainer.sgd_momentum),
      loss_nonfinite_count_(mx::array(0.0f, mx::float32)),
      gradient_nonfinite_count_(mx::array(0.0f, mx::float32)),
      state_nonfinite_count_(mx::array(0.0f, mx::float32)) {}

OptimizerStepTransaction::~OptimizerStepTransaction() {
  if (!finalized_) {
    restore();
  }
}

void OptimizerStepTransaction::append_validation_arrays(
    const mx::array& loss,
    const std::vector<mx::array>& gradients,
    std::vector<mx::array>& eval_arrays,
    const mx::array& known_gradient_nonfinite) {
  if (validation_prepared_) {
    throw std::runtime_error("optimizer-step validation was prepared more than once");
  }
  loss_nonfinite_count_ = nonfinite_count(loss);
  gradient_nonfinite_count_ = mx::astype(known_gradient_nonfinite, mx::float32);
  accumulate_nonfinite(gradient_nonfinite_count_, gradients);
  state_nonfinite_count_ = mx::array(0.0f, mx::float32);
  accumulate_active_optimizer_state_nonfinite(state_nonfinite_count_, trainer_);
  eval_arrays.push_back(loss_nonfinite_count_);
  eval_arrays.push_back(gradient_nonfinite_count_);
  eval_arrays.push_back(state_nonfinite_count_);
  validation_prepared_ = true;
}

bool OptimizerStepTransaction::finish() {
  if (finalized_) {
    throw std::runtime_error("optimizer-step transaction was finalized more than once");
  }
  if (!validation_prepared_) {
    throw std::runtime_error("optimizer-step transaction has no validation arrays");
  }
  const float loss_bad = evaluated_count(loss_nonfinite_count_, "loss non-finite");
  const float gradient_bad = evaluated_count(gradient_nonfinite_count_, "gradient non-finite");
  const float state_bad = evaluated_count(state_nonfinite_count_, "state non-finite");
  loss_was_nonfinite_ = loss_bad > 0.0f;
  const bool valid = loss_bad == 0.0f && gradient_bad == 0.0f && state_bad == 0.0f;
  if (valid) {
    trainer_.optimizer_step_count++;
    trainer_.consecutive_skipped_optimizer_steps = 0;
    trainer_.last_optimizer_step_skipped = false;
    trainer_.last_optimizer_loss_nonfinite = 0;
    trainer_.last_optimizer_gradient_nonfinite = 0;
    trainer_.last_optimizer_state_nonfinite = 0;
  } else {
    restore();
    trainer_.skipped_optimizer_steps++;
    trainer_.consecutive_skipped_optimizer_steps++;
    trainer_.last_optimizer_step_skipped = true;
    trainer_.last_optimizer_loss_nonfinite = static_cast<uint64_t>(loss_bad);
    trainer_.last_optimizer_gradient_nonfinite = static_cast<uint64_t>(gradient_bad);
    trainer_.last_optimizer_state_nonfinite = static_cast<uint64_t>(state_bad);
    std::cerr << "[mlx_ir] skipped non-finite optimizer update"
              << " training_step=" << trainer_.step_count
              << " committed_optimizer_steps=" << trainer_.optimizer_step_count
              << " loss_nonfinite=" << loss_bad
              << " gradient_nonfinite=" << gradient_bad
              << " state_nonfinite=" << state_bad
              << " consecutive_skips=" << trainer_.consecutive_skipped_optimizer_steps
              << std::endl;
  }
  finalized_ = true;
  const auto state_elements = active_optimizer_state_elements(trainer_);
  const bool fully_nonfinite_state =
      state_elements > 0 && state_bad >= static_cast<float>(state_elements);
  if (!valid &&
      (fully_nonfinite_state ||
       trainer_.consecutive_skipped_optimizer_steps >= kMaxConsecutiveSkippedOptimizerSteps)) {
    throw OptimizerStepCircuitBreaker(
        "optimizer safety circuit breaker: rejected " +
        std::to_string(trainer_.consecutive_skipped_optimizer_steps) +
        " consecutive non-finite updates" +
        (fully_nonfinite_state ? " (candidate optimizer state was fully non-finite)" : "") +
        "; restored the last committed weights and optimizer moments");
  }
  return valid;
}

bool OptimizerStepTransaction::loss_was_nonfinite() const {
  return loss_was_nonfinite_;
}

void OptimizerStepTransaction::restore() {
  trainer_.weights = weights_;
  trainer_.adam_m = adam_m_;
  trainer_.adam_v = adam_v_;
  trainer_.muon_momentum = muon_momentum_;
  trainer_.muon_second_moment = muon_second_moment_;
  trainer_.sgd_momentum = sgd_momentum_;
}

void sanitize_skipped_step_reporting(
    mx::array& loss,
    std::unordered_map<std::string, mx::array>& outputs,
    bool loss_was_nonfinite) {
  if (loss_was_nonfinite) {
    loss = mx::array(0.0f, mx::float32);
  }
  std::vector<mx::array> eval_arrays;
  eval_arrays.reserve(1 + outputs.size());
  eval_arrays.push_back(loss);
  for (auto& [_, value] : outputs) {
    value = mx::where(mx::isfinite(value), value, mx::zeros_like(value));
    eval_arrays.push_back(value);
  }
  mx::eval(eval_arrays);
}

} // namespace mlx_ir
