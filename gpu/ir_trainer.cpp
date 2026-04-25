#include "ir_trainer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace mx = mlx::core;

namespace mlx_ir {
namespace {

struct LoraOptimizerState {
  mx::array adam_m = mx::array(0.0f, mx::float32);
  mx::array adam_v = mx::array(0.0f, mx::float32);
  uint8_t has_adam_state = 0;
  mx::array muon_momentum = mx::array(0.0f, mx::float32);
  uint8_t has_muon_state = 0;
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

void init_lora_optimizer_state(const mx::array& weight, const OptimizerGroupConfig& group, LoraOptimizerState& state) {
  switch (group.kind) {
    case OptimizerKind::AdamW:
      state.adam_m = mx::zeros_like(weight);
      state.adam_v = mx::zeros_like(weight);
      state.has_adam_state = 1;
      state.muon_momentum = mx::array(0.0f, mx::float32);
      state.has_muon_state = 0;
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
  switch (group.kind) {
    case OptimizerKind::AdamW: {
      if (state.has_adam_state == 0) {
        throw std::runtime_error("AdamW state missing for LoRA adapter");
      }
      const float b1t = 1.0f - std::pow(group.beta1, static_cast<float>(step_count));
      const float b2t = 1.0f - std::pow(group.beta2, static_cast<float>(step_count));
      const float one_minus_beta1 = 1.0f - group.beta1;
      const float one_minus_beta2 = 1.0f - group.beta2;
      state.adam_m = group.beta1 * state.adam_m + one_minus_beta1 * g;
      state.adam_v = group.beta2 * state.adam_v + one_minus_beta2 * mx::square(g);

      auto mhat = state.adam_m / b1t;
      auto vhat = state.adam_v / b2t;

      if (group.weight_decay > 0.0f && decay) {
        w = w - (effective_lr * group.weight_decay) * w;
      }
      w = w - effective_lr * mhat / (mx::sqrt(vhat) + group.eps);
      break;
    }
    case OptimizerKind::Muon: {
      if (state.has_muon_state == 0) {
        throw std::runtime_error("Muon state missing for LoRA adapter");
      }
      if (w.ndim() != 2) {
        throw std::runtime_error("Muon only supports rank-2 LoRA adapters");
      }
      state.muon_momentum = group.beta1 * state.muon_momentum + g;
      mx::array update = group.nesterov ? (g + group.beta1 * state.muon_momentum) : state.muon_momentum;
      update = zeropower_via_newtonschulz5(update, group.backend_steps);
      const auto rows = static_cast<float>(w.shape(0));
      const auto cols = static_cast<float>(w.shape(1));
      const float aspect = std::sqrt(std::max(1.0f, rows / cols));
      update = update * mx::array(aspect, mx::float32);
      if (group.weight_decay > 0.0f && decay) {
        w = w - (effective_lr * group.weight_decay) * w;
      }
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
    QATMode qat_mode) {
  if (qat_mode == QATMode::None) {
    return base_weights;
  }
  std::vector<mx::array> effective;
  effective.reserve(base_weights.size());
  for (const auto& weight : base_weights) {
    effective.push_back(fake_quantize_weight(weight, qat_mode));
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
}

std::vector<std::string> collect_cached_output_names(const IRProgram& program) {
  bool capture_magnitudes = false;
  bool capture_x_hidden = false;
  bool capture_logits = false;
  bool capture_qr = false;
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
  if (capture_qr) {
    output_names.push_back("qr");
  }
  return output_names;
}

float finalize_pending_step(
    IRTrainer& trainer,
    std::unordered_map<std::string, mx::array>* outputs) {
  if (!trainer.has_pending_step_) {
    throw std::runtime_error("no pending step to collect");
  }
  float loss = trainer.pending_loss_.item<float>();
  if (outputs != nullptr) {
    *outputs = std::move(trainer.pending_outputs_);
  }
  trainer.pending_outputs_.clear();
  trainer.has_pending_step_ = false;
  return loss;
}

} // namespace

IRTrainer::IRTrainer() : pending_loss_(mx::array(0.0f, mx::float32)) {}

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
        auto effective = effective_training_weights(w, qat_mode);
        return ir_interpret(program, effective, tokens, targets, true);
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
  flush();
  submit_step(inputs);
  return collect_loss();
}

void IRTrainer::submit_step(const TensorMap& inputs) {
  if (weights.empty()) {
    throw std::runtime_error("IR trainer has no weights");
  }
  if (has_ready_step_) {
    throw std::runtime_error("previous step loss must be collected before submitting another step");
  }
  if (has_pending_step_) {
    ready_loss_ = finalize_pending_step(*this, &ready_outputs_);
    has_ready_step_ = true;
  }
  step_count++;

  std::vector<int> argnums(weights.size());
  std::iota(argnums.begin(), argnums.end(), 0);
  std::vector<std::string> output_names = collect_cached_output_names(program);
  std::unordered_map<std::string, mx::array> outputs;
  auto fn = mx::value_and_grad(
      [this, inputs, &outputs, &output_names](const std::vector<mx::array>& w) {
        auto effective = effective_training_weights(w, qat_mode);
        outputs = ir_interpret_outputs(program, effective, inputs, output_names, true);
        return outputs.at("loss");
      },
      argnums);

  auto out = fn(weights);
  auto loss = out.first;
  auto grads = std::move(out.second);
  clip_gradients(grads, max_grad_norm);
  apply_optimizer_updates(grads);

  std::vector<mx::array> eval_arrays;
  eval_arrays.reserve(1 + outputs.size() + weights.size() + adam_m.size() * 2 + muon_momentum.size());
  eval_arrays.push_back(loss);
  for (const auto& [_, output] : outputs) {
    eval_arrays.push_back(output);
  }
  collect_state_for_eval(*this, eval_arrays, false);
  mx::eval(eval_arrays);
  pending_loss_ = loss;
  pending_outputs_ = std::move(outputs);
  has_pending_step_ = true;
}

float IRTrainer::collect_loss() {
  if (has_ready_step_) {
    last_outputs = std::move(ready_outputs_);
    ready_outputs_.clear();
    has_ready_step_ = false;
    return ready_loss_;
  }
  last_outputs.clear();
  return finalize_pending_step(*this, &last_outputs);
}

void IRTrainer::flush() {
  if (has_pending_step_) {
    last_outputs = std::move(pending_outputs_);
    pending_outputs_.clear();
    pending_loss_.item<float>();
    has_pending_step_ = false;
  } else if (has_ready_step_) {
    last_outputs = std::move(ready_outputs_);
    ready_outputs_.clear();
  }
  has_ready_step_ = false;
}

float IRTrainer::evaluate(const mx::array& tokens, const mx::array& targets) {
  flush();
  if (weights.empty()) {
    throw std::runtime_error("IR trainer has no weights");
  }
  auto loss = ir_interpret(program, weights, tokens, targets);
  mx::eval(loss);
  return loss.item<float>();
}

float IRTrainer::evaluate_named(const TensorMap& inputs) {
  flush();
  if (weights.empty()) {
    throw std::runtime_error("IR trainer has no weights");
  }
  auto output_names = collect_cached_output_names(program);
  last_outputs = ir_interpret_outputs(program, weights, inputs, output_names);
  auto loss = last_outputs.at("loss");
  mx::eval(loss);
  return loss.item<float>();
}

std::vector<float> IRTrainer::evaluate_per_token(const TensorMap& inputs) {
  flush();
  if (weights.empty()) {
    throw std::runtime_error("IR trainer has no weights");
  }
  auto output_names = collect_cached_output_names(program);
  output_names.push_back("per_token_nll");
  last_outputs = ir_interpret_outputs(program, weights, inputs, output_names);
  auto nll = mx::astype(last_outputs.at("per_token_nll"), mx::float32);
  auto flat = mx::reshape(nll, {static_cast<mx::ShapeElem>(nll.size())});
  mx::eval(flat);
  std::vector<float> result(static_cast<size_t>(flat.shape(0)));
  std::memcpy(result.data(), flat.data<float>(), result.size() * sizeof(float));
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
  for (int local_step = 0; local_step < steps; ++local_step) {
    std::vector<int> argnums;
    argnums.reserve(adapter_indices.size() * 2);
    for (size_t i = 0; i < adapter_indices.size()*2; ++i) {
      argnums.push_back(static_cast<int>(i));
    }
    auto fn = mx::value_and_grad(
        [this, inputs, &adapters, &adapter_indices](const std::vector<mx::array>& params) {
          auto local_adapters = adapters;
          size_t param_idx = 0;
          for (size_t weight_idx : adapter_indices) {
            local_adapters[weight_idx].a = params[param_idx++];
            local_adapters[weight_idx].b = params[param_idx++];
          }
          auto effective = effective_lora_weights(weights, local_adapters);
          return ir_interpret(program, effective, inputs, "", true);
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
    clip_gradients(grads, max_grad_norm);

    size_t grad_idx = 0;
    for (size_t weight_idx : adapter_indices) {
      auto& adapter = adapters[weight_idx];
      apply_optimizer_update(adapter.a, grads[grad_idx++], adapter.group, false, local_step+1, local_lr_scale, adapter.a_state);
      apply_optimizer_update(adapter.b, grads[grad_idx++], adapter.group, false, local_step+1, local_lr_scale, adapter.b_state);
    }

    std::vector<mx::array> eval_arrays;
    eval_arrays.reserve(adapter_indices.size() * 6);
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
    }
    if (!eval_arrays.empty()) {
      mx::eval(eval_arrays);
    }
  }

  auto output_names = collect_cached_output_names(program);
  auto effective = effective_lora_weights(weights, adapters);
  last_outputs = ir_interpret_outputs(program, effective, inputs, output_names);
  auto loss = last_outputs.at("loss");
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
