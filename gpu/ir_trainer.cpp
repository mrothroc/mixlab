#include "ir_trainer.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace mx = mlx::core;

namespace mlx_ir {
namespace {

static mx::array zeropower_via_newtonschulz5(const mx::array& grad, int steps, float eps = 1e-7f) {
  constexpr float a = 3.4445f, b = -4.7750f, c = 2.0315f;
  auto x = mx::astype(grad, mx::bfloat16);
  auto norm = mx::sqrt(mx::sum(mx::square(mx::astype(x, mx::float32))));
  x = x / (norm + mx::array(eps, mx::float32));
  bool transposed = grad.shape(0) > grad.shape(1);
  if (transposed) {
    x = mx::transpose(x, {1, 0});
  }
  for (int i = 0; i < steps; ++i) {
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

void clip_gradients(std::vector<mx::array>& grads, float max_grad_norm) {
  if (max_grad_norm <= 0.0f) {
    return;
  }
  auto total_norm_sq = mx::array(0.0f, mx::float32);
  for (const auto& g : grads) {
    total_norm_sq = total_norm_sq + mx::sum(mx::square(g));
  }
  auto total_norm = mx::sqrt(total_norm_sq);
  auto clip_scale = mx::minimum(
      mx::array(1.0f, mx::float32),
      mx::array(max_grad_norm, mx::float32) / (total_norm + mx::array(1e-6f, mx::float32)));
  for (auto& g : grads) {
    g = g * clip_scale;
  }
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
}

std::vector<std::string> collect_cached_output_names(const IRProgram& program) {
  bool capture_magnitudes = false;
  bool capture_x_hidden = false;
  bool capture_logits = false;
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
    }
  }

  std::vector<std::string> output_names{"loss"};
  if (capture_magnitudes) {
    output_names.push_back("magnitudes");
  }
  if (capture_x_hidden) {
    output_names.push_back("x_hidden");
  }
  if (capture_logits) {
    output_names.push_back("logits");
  }
  return output_names;
}

} // namespace

float IRTrainer::step(const mx::array& tokens, const mx::array& targets) {
  if (weights.empty()) {
    throw std::runtime_error("IR trainer has no weights");
  }
  step_count++;

  std::vector<int> argnums(weights.size());
  std::iota(argnums.begin(), argnums.end(), 0);
  auto fn = mx::value_and_grad(
      [this, tokens, targets](const std::vector<mx::array>& w) {
        return ir_interpret(program, w, tokens, targets);
      },
      argnums);

  auto out = fn(weights);
  auto loss = out.first;
  auto grads = std::move(out.second);
  clip_gradients(grads, max_grad_norm);
  apply_optimizer_updates(grads);

  std::vector<mx::array> eval_arrays;
  eval_arrays.reserve(1 + weights.size() + adam_m.size() * 2 + muon_momentum.size());
  eval_arrays.push_back(loss);
  collect_state_for_eval(*this, eval_arrays, false);
  mx::eval(eval_arrays);
  return loss.item<float>();
}

void IRTrainer::apply_optimizer_updates(const std::vector<mx::array>& grads) {
  if (grads.size() != weights.size()) {
    throw std::runtime_error("gradient/weight size mismatch");
  }

  for (size_t i = 0; i < weights.size(); ++i) {
    if (i >= weight_optimizers.size()) {
      throw std::runtime_error("missing weight optimizer spec");
    }
    const auto& spec = weight_optimizers[i];
    if (spec.group_index >= optimizer_groups.size()) {
      throw std::runtime_error("weight optimizer group index out of range");
    }
    const auto& group = optimizer_groups[spec.group_index];
    const float effective_lr = group.lr * lr_scale;
    auto& w = weights[i];
    const auto& g = grads[i];

    switch (group.kind) {
      case OptimizerKind::AdamW: {
        if (has_adam_state[i] == 0) {
          throw std::runtime_error("AdamW state missing for weight");
        }
        const float b1t = 1.0f - std::pow(group.beta1, static_cast<float>(step_count));
        const float b2t = 1.0f - std::pow(group.beta2, static_cast<float>(step_count));
        const float one_minus_beta1 = 1.0f - group.beta1;
        const float one_minus_beta2 = 1.0f - group.beta2;
        adam_m[i] = group.beta1 * adam_m[i] + one_minus_beta1 * g;
        adam_v[i] = group.beta2 * adam_v[i] + one_minus_beta2 * mx::square(g);

        auto mhat = adam_m[i] / b1t;
        auto vhat = adam_v[i] / b2t;

        if (group.weight_decay > 0.0f && spec.decay) {
          w = w - (effective_lr * group.weight_decay) * w;
        }
        w = w - effective_lr * mhat / (mx::sqrt(vhat) + group.eps);
        break;
      }
      case OptimizerKind::Muon: {
        if (has_muon_state[i] == 0) {
          throw std::runtime_error("Muon state missing for weight");
        }
        if (w.ndim() != 2) {
          throw std::runtime_error("Muon only supports rank-2 weights");
        }
        muon_momentum[i] = group.beta1 * muon_momentum[i] + g;
        mx::array update = group.nesterov ? (g + group.beta1 * muon_momentum[i]) : muon_momentum[i];
        update = zeropower_via_newtonschulz5(update, group.backend_steps);
        const auto rows = static_cast<float>(w.shape(0));
        const auto cols = static_cast<float>(w.shape(1));
        const float aspect = std::sqrt(std::max(1.0f, rows / cols));
        update = update * mx::array(aspect, mx::float32);
        if (group.weight_decay > 0.0f && spec.decay) {
          w = w - (effective_lr * group.weight_decay) * w;
        }
        w = w - effective_lr * update;
        break;
      }
      default:
        throw std::runtime_error("unsupported optimizer kind");
    }
  }
}

float IRTrainer::step_named(const TensorMap& inputs) {
  if (weights.empty()) {
    throw std::runtime_error("IR trainer has no weights");
  }
  step_count++;

  std::vector<int> argnums(weights.size());
  std::iota(argnums.begin(), argnums.end(), 0);
  std::vector<std::string> output_names = collect_cached_output_names(program);
  std::unordered_map<std::string, mx::array> outputs;
  auto fn = mx::value_and_grad(
      [this, inputs, &outputs, &output_names](const std::vector<mx::array>& w) {
        outputs = ir_interpret_outputs(program, w, inputs, output_names);
        return outputs.at("loss");
      },
      argnums);

  auto out = fn(weights);
  auto loss = out.first;
  auto grads = std::move(out.second);
  clip_gradients(grads, max_grad_norm);
  apply_optimizer_updates(grads);

  last_outputs = std::move(outputs);
  std::vector<mx::array> eval_arrays;
  eval_arrays.reserve(1 + last_outputs.size() + weights.size() + adam_m.size() * 2 + muon_momentum.size());
  eval_arrays.push_back(loss);
  collect_state_for_eval(*this, eval_arrays, true);
  mx::eval(eval_arrays);
  return loss.item<float>();
}

float IRTrainer::evaluate(const mx::array& tokens, const mx::array& targets) {
  if (weights.empty()) {
    throw std::runtime_error("IR trainer has no weights");
  }
  auto loss = ir_interpret(program, weights, tokens, targets);
  mx::eval(loss);
  return loss.item<float>();
}

float IRTrainer::evaluate_named(const TensorMap& inputs) {
  if (weights.empty()) {
    throw std::runtime_error("IR trainer has no weights");
  }
  auto output_names = collect_cached_output_names(program);
  last_outputs = ir_interpret_outputs(program, weights, inputs, output_names);
  auto loss = last_outputs.at("loss");
  mx::eval(loss);
  return loss.item<float>();
}

mx::array IRTrainer::read_output(const std::string& output_name) const {
  auto it = last_outputs.find(output_name);
  if (it == last_outputs.end()) {
    throw std::runtime_error("IR trainer has no cached output: " + output_name);
  }
  return it->second;
}

std::unique_ptr<IRTrainer> create_ir_trainer(
    const IRProgram& program,
    const std::vector<mx::array>& initial_weights,
    const std::vector<WeightOptimizerSpec>& weight_specs,
    const std::vector<OptimizerGroupConfig>& groups,
    float max_grad_norm,
    float default_base_lr) {
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
  trainer->program = program;
  trainer->weights = initial_weights;
  trainer->optimizer_groups = groups;
  trainer->weight_optimizers = weight_specs;
  trainer->adam_m.reserve(initial_weights.size());
  trainer->adam_v.reserve(initial_weights.size());
  trainer->has_adam_state.reserve(initial_weights.size());
  trainer->muon_momentum.reserve(initial_weights.size());
  trainer->has_muon_state.reserve(initial_weights.size());
  for (size_t i = 0; i < initial_weights.size(); ++i) {
    const auto& spec = weight_specs[i];
    if (spec.group_index >= groups.size()) {
      throw std::runtime_error("IR optimizer group index out of range");
    }
    const auto& group = groups[spec.group_index];
    switch (group.kind) {
      case OptimizerKind::AdamW:
        trainer->adam_m.push_back(mx::zeros_like(initial_weights[i]));
        trainer->adam_v.push_back(mx::zeros_like(initial_weights[i]));
        trainer->has_adam_state.push_back(1);
        trainer->muon_momentum.push_back(mx::array(0.0f, mx::float32));
        trainer->has_muon_state.push_back(0);
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
