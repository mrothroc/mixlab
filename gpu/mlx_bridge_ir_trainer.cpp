#include "mlx_bridge_internal.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

std::vector<mx::array> read_weight_arrays(int64_t* weight_handles, int n_weights) {
  std::vector<mx::array> weights;
  weights.reserve(static_cast<size_t>(n_weights));
  for (int i = 0; i < n_weights; ++i) {
    auto* w = get_handle(weight_handles[i]);
    if (!w) {
      throw std::runtime_error("invalid weight handle");
    }
    weights.push_back(*w);
  }
  return weights;
}

std::vector<mlx_ir::WeightOptimizerSpec> read_weight_specs(
    const mlx_ir_weight_optimizer* weight_optimizers,
    int n_weight_optimizers) {
  std::vector<mlx_ir::WeightOptimizerSpec> specs;
  specs.reserve(static_cast<size_t>(n_weight_optimizers));
  for (int i = 0; i < n_weight_optimizers; ++i) {
    if (weight_optimizers[i].group_index < 0) {
      throw std::runtime_error("negative optimizer group index");
    }
    specs.push_back(mlx_ir::WeightOptimizerSpec{
        static_cast<uint32_t>(weight_optimizers[i].group_index),
        weight_optimizers[i].decay != 0,
    });
  }
  return specs;
}

std::vector<mlx_ir::OptimizerGroupConfig> read_optimizer_groups(
    const mlx_ir_optimizer_group* optimizer_groups,
    int n_optimizer_groups) {
  std::vector<mlx_ir::OptimizerGroupConfig> groups;
  groups.reserve(static_cast<size_t>(n_optimizer_groups));
  for (int i = 0; i < n_optimizer_groups; ++i) {
    mlx_ir::OptimizerKind kind;
    switch (optimizer_groups[i].kind) {
      case 0:
        kind = mlx_ir::OptimizerKind::AdamW;
        break;
      case 1:
        kind = mlx_ir::OptimizerKind::Muon;
        break;
      default:
        throw std::runtime_error("unsupported optimizer kind");
    }
    groups.push_back(mlx_ir::OptimizerGroupConfig{
        kind,
        optimizer_groups[i].lr,
        optimizer_groups[i].beta1,
        optimizer_groups[i].beta2,
        optimizer_groups[i].eps,
        optimizer_groups[i].weight_decay,
        optimizer_groups[i].backend_steps,
        optimizer_groups[i].nesterov != 0,
    });
  }
  return groups;
}

float effective_lr_scale(const mlx_ir::IRTrainer& trainer, float lr) {
  if (trainer.default_base_lr <= 0.0f) {
    return 1.0f;
  }
  return lr / trainer.default_base_lr;
}

mlx_ir::QATMode parse_qat_mode(const char* mode) {
  if (!mode || mode[0] == '\0') {
    return mlx_ir::QATMode::None;
  }
  if (std::strcmp(mode, "none") == 0) {
    return mlx_ir::QATMode::None;
  }
  if (std::strcmp(mode, "int8") == 0) {
    return mlx_ir::QATMode::Int8;
  }
  if (std::strcmp(mode, "int6") == 0) {
    return mlx_ir::QATMode::Int6;
  }
  throw std::runtime_error("unsupported QAT mode");
}

} // namespace

extern "C" {

int64_t mlx_ir_create_trainer(
    int64_t program,
    int64_t* weight_handles,
    int n_weights,
    const int* decay_flags,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float wd,
    float max_grad_norm) {
  if (program <= 0 || !weight_handles || n_weights <= 0 || !decay_flags) {
    return 0;
  }
  mlx_ir_optimizer_group group = {
      0,
      lr,
      beta1,
      beta2,
      eps,
      wd,
      5,
      1,
  };
  std::vector<mlx_ir_weight_optimizer> specs(static_cast<size_t>(n_weights));
  for (int i = 0; i < n_weights; ++i) {
    specs[static_cast<size_t>(i)] = mlx_ir_weight_optimizer{0, decay_flags[i] != 0 ? 1 : 0};
  }
  return mlx_ir_create_trainer_v2(
      program,
      weight_handles,
      n_weights,
      specs.data(),
      n_weights,
      &group,
      1,
      max_grad_norm,
      lr);
}

int64_t mlx_ir_create_trainer_v2(
    int64_t program,
    int64_t* weight_handles,
    int n_weights,
    const mlx_ir_weight_optimizer* weight_optimizers,
    int n_weight_optimizers,
    const mlx_ir_optimizer_group* optimizer_groups,
    int n_optimizer_groups,
    float max_grad_norm,
    float default_base_lr) {
  if (program <= 0 || !weight_handles || n_weights <= 0 || !weight_optimizers || !optimizer_groups ||
      n_weight_optimizers != n_weights || n_optimizer_groups <= 0) {
    return 0;
  }
  try {
    if (!g_initialized && mlx_init() != 0) {
      return 0;
    }
    auto* prog = get_ir_program(program);
    if (!prog || prog->n_weights != n_weights) {
      return 0;
    }
    auto weights = read_weight_arrays(weight_handles, n_weights);
    auto specs = read_weight_specs(weight_optimizers, n_weight_optimizers);
    auto groups = read_optimizer_groups(optimizer_groups, n_optimizer_groups);
    auto trainer = mlx_ir::create_ir_trainer(*prog, weights, specs, groups, max_grad_norm, default_base_lr);
    return alloc_ir_trainer(std::move(trainer));
  } catch (const std::exception& e) {
    log_bridge_exception("mlx_ir_create_trainer_v2", e);
    return 0;
  } catch (...) {
    std::cerr << "[mlx_bridge] mlx_ir_create_trainer_v2 unknown exception" << std::endl;
    return 0;
  }
}

float mlx_ir_trainer_step(int64_t trainer, const int* tokens, const int* targets, int B, int T) {
  if (!tokens || !targets || B <= 0 || T <= 0) {
    return std::nanf("");
  }
  const size_t n_tok = static_cast<size_t>(B) * static_cast<size_t>(T);
  const int tok_shape[2] = {B, T};
  const int tgt_shape[1] = {static_cast<int>(n_tok)};
  mlx_tensor_input inputs[2] = {
      mlx_tensor_input{"tokens", 0, tok_shape, 2, tokens, static_cast<int>(n_tok * sizeof(int32_t))},
      mlx_tensor_input{"targets", 0, tgt_shape, 1, targets, static_cast<int>(n_tok * sizeof(int32_t))},
  };
  return mlx_ir_trainer_step_named(trainer, inputs, 2);
}

float mlx_ir_trainer_step_named(int64_t trainer, const mlx_tensor_input* inputs, int n_inputs) {
  if (!inputs || n_inputs <= 0) {
    return std::nanf("");
  }
  try {
    if (!g_initialized && mlx_init() != 0) {
      return std::nanf("");
    }
    auto* t = get_ir_trainer(trainer);
    if (!t) {
      return std::nanf("");
    }
    auto input_map = to_tensor_map(inputs, n_inputs);
    return t->step_named(input_map);
  } catch (const std::exception& e) {
    log_bridge_exception("mlx_ir_trainer_step_named", e);
    return std::nanf("");
  } catch (...) {
    std::cerr << "[mlx_bridge] mlx_ir_trainer_step_named unknown exception" << std::endl;
    return std::nanf("");
  }
}

void mlx_ir_trainer_submit_step(int64_t trainer, const mlx_tensor_input* inputs, int n_inputs) {
  if (!inputs || n_inputs <= 0) {
    return;
  }
  try {
    if (!g_initialized && mlx_init() != 0) {
      return;
    }
    auto* t = get_ir_trainer(trainer);
    if (!t) {
      return;
    }
    auto input_map = to_tensor_map(inputs, n_inputs);
    t->submit_step(input_map);
  } catch (const std::exception& e) {
    log_bridge_exception("mlx_ir_trainer_submit_step", e);
  } catch (...) {
    std::cerr << "[mlx_bridge] mlx_ir_trainer_submit_step unknown exception" << std::endl;
  }
}

float mlx_ir_trainer_collect_loss(int64_t trainer) {
  try {
    if (!g_initialized && mlx_init() != 0) {
      return std::nanf("");
    }
    auto* t = get_ir_trainer(trainer);
    if (!t) {
      return std::nanf("");
    }
    return t->collect_loss();
  } catch (const std::exception& e) {
    log_bridge_exception("mlx_ir_trainer_collect_loss", e);
    return std::nanf("");
  } catch (...) {
    std::cerr << "[mlx_bridge] mlx_ir_trainer_collect_loss unknown exception" << std::endl;
    return std::nanf("");
  }
}

void mlx_ir_trainer_flush(int64_t trainer) {
  try {
    if (!g_initialized && mlx_init() != 0) {
      return;
    }
    auto* t = get_ir_trainer(trainer);
    if (!t) {
      return;
    }
    t->flush();
  } catch (const std::exception& e) {
    log_bridge_exception("mlx_ir_trainer_flush", e);
  } catch (...) {
    std::cerr << "[mlx_bridge] mlx_ir_trainer_flush unknown exception" << std::endl;
  }
}

float mlx_ir_trainer_evaluate_named(int64_t trainer, const mlx_tensor_input* inputs, int n_inputs) {
  try {
    if (!g_initialized && mlx_init() != 0) {
      return std::nanf("");
    }
    auto* t = get_ir_trainer(trainer);
    if (!t || !inputs || n_inputs <= 0) {
      return std::nanf("");
    }
    auto tensor_map = to_tensor_map(inputs, n_inputs);
    return t->evaluate_named(tensor_map);
  } catch (...) {
    return std::nanf("");
  }
}

float mlx_ir_trainer_evaluate_lora_named(
    int64_t trainer,
    const mlx_tensor_input* inputs,
    int n_inputs,
    int rank,
    int steps,
    float lr) {
  try {
    if (!g_initialized && mlx_init() != 0) {
      return std::nanf("");
    }
    auto* t = get_ir_trainer(trainer);
    if (!t || !inputs || n_inputs <= 0 || rank <= 0 || steps < 0) {
      return std::nanf("");
    }
    auto tensor_map = to_tensor_map(inputs, n_inputs);
    return t->evaluate_lora_named(tensor_map, rank, steps, lr);
  } catch (const std::exception& e) {
    log_bridge_exception("mlx_ir_trainer_evaluate_lora_named", e);
    return std::nanf("");
  } catch (...) {
    std::cerr << "[mlx_bridge] mlx_ir_trainer_evaluate_lora_named unknown exception" << std::endl;
    return std::nanf("");
  }
}

int mlx_ir_trainer_read_output(int64_t trainer, const char* output_name, float* out, int out_size) {
  if (!output_name || output_name[0] == '\0' || !out || out_size <= 0) {
    return -1;
  }
  try {
    if (!g_initialized && mlx_init() != 0) {
      return -1;
    }
    auto* t = get_ir_trainer(trainer);
    if (!t) {
      return -1;
    }
    t->flush();
    auto output = t->read_output(output_name);
    if (output.dtype() != mx::float32 || static_cast<int>(output.size()) != out_size) {
      return -1;
    }
    mx::eval(output);
    std::memcpy(out, output.data<float>(), static_cast<size_t>(out_size) * sizeof(float));
    return 0;
  } catch (const std::exception& e) {
    log_bridge_exception("mlx_ir_trainer_read_output", e);
    return -1;
  } catch (...) {
    std::cerr << "[mlx_bridge] mlx_ir_trainer_read_output unknown exception" << std::endl;
    return -1;
  }
}

float mlx_ir_trainer_evaluate(int64_t trainer, const int* tokens, const int* targets, int B, int T) {
  if (!tokens || !targets || B <= 0 || T <= 0) {
    return std::nanf("");
  }
  try {
    if (!g_initialized && mlx_init() != 0) {
      return std::nanf("");
    }
    auto* t = get_ir_trainer(trainer);
    if (!t) {
      return std::nanf("");
    }

    const size_t n_tok = static_cast<size_t>(B) * static_cast<size_t>(T);
    std::vector<int32_t> tok32(n_tok);
    std::vector<int32_t> tgt32(n_tok);
    for (size_t i = 0; i < n_tok; ++i) {
      tok32[i] = static_cast<int32_t>(tokens[i]);
      tgt32[i] = static_cast<int32_t>(targets[i]);
    }

    auto tok_arr = mx::array(
        tok32.begin(),
        {static_cast<mx::ShapeElem>(B), static_cast<mx::ShapeElem>(T)},
        mx::int32);
    auto tgt_arr = mx::array(tgt32.begin(), {static_cast<mx::ShapeElem>(n_tok)}, mx::int32);
    return t->evaluate(tok_arr, tgt_arr);
  } catch (...) {
    return std::nanf("");
  }
}

int mlx_ir_trainer_num_weights(int64_t trainer) {
  try {
    auto* t = get_ir_trainer(trainer);
    if (!t) {
      return -1;
    }
    return static_cast<int>(t->weights.size());
  } catch (...) {
    return -1;
  }
}

int mlx_ir_trainer_weight_size(int64_t trainer, int weight_idx) {
  try {
    auto* t = get_ir_trainer(trainer);
    if (!t || weight_idx < 0 || static_cast<size_t>(weight_idx) >= t->weights.size()) {
      return -1;
    }
    return static_cast<int>(t->weights[static_cast<size_t>(weight_idx)].size());
  } catch (...) {
    return -1;
  }
}

int mlx_ir_trainer_read_weight(int64_t trainer, int weight_idx, float* out, int size) {
  if (!out || size < 0) {
    return -1;
  }
  try {
    auto* t = get_ir_trainer(trainer);
    if (!t || weight_idx < 0 || static_cast<size_t>(weight_idx) >= t->weights.size()) {
      return -1;
    }
    t->flush();
    auto w = mx::astype(t->weights[static_cast<size_t>(weight_idx)], mx::float32);
    const int n = static_cast<int>(w.size());
    if (size != n) {
      return -1;
    }
    auto flat = mx::reshape(w, {static_cast<mx::ShapeElem>(n)});
    mx::eval(flat);
    std::memcpy(out, flat.data<float>(), static_cast<size_t>(n) * sizeof(float));
    return 0;
  } catch (...) {
    return -1;
  }
}

void mlx_ir_trainer_set_lr(int64_t trainer, float lr) {
  auto* t = get_ir_trainer(trainer);
  if (!t) {
    return;
  }
  t->lr_scale = effective_lr_scale(*t, lr);
}

void mlx_ir_trainer_set_lr_scale(int64_t trainer, float lr_scale) {
  auto* t = get_ir_trainer(trainer);
  if (!t) {
    return;
  }
  t->lr_scale = lr_scale;
}

void mlx_ir_trainer_set_qat(int64_t trainer, const char* mode) {
  auto* t = get_ir_trainer(trainer);
  if (!t) {
    return;
  }
  try {
    t->qat_mode = parse_qat_mode(mode);
  } catch (const std::exception& e) {
    log_bridge_exception("mlx_ir_trainer_set_qat", e);
  } catch (...) {
    std::cerr << "[mlx_bridge] mlx_ir_trainer_set_qat unknown exception" << std::endl;
  }
}

void mlx_ir_trainer_destroy(int64_t trainer) {
  if (trainer <= 0) {
    return;
  }
  const size_t idx = static_cast<size_t>(trainer - 1);
  if (idx >= g_ir_trainer_pool.size()) {
    return;
  }
  if (g_ir_trainer_pool[idx]) {
    g_ir_trainer_pool[idx]->flush();
  }
  g_ir_trainer_pool[idx].reset();
}

} // extern "C"
