#include "mlx_bridge_internal.h"

#include <cstring>
#include <limits>
#include <vector>

namespace {

struct OptimizerStateRef {
  int kind;
  int weight_idx;
  mx::array* value;
};

std::vector<OptimizerStateRef> optimizer_state_refs(mlx_ir::IRTrainer& trainer) {
  std::vector<OptimizerStateRef> refs;
  for (size_t i = 0; i < trainer.weights.size(); ++i) {
    if (i < trainer.has_adam_state.size() && trainer.has_adam_state[i] != 0) {
      refs.push_back({0, static_cast<int>(i), &trainer.adam_m[i]});
      refs.push_back({1, static_cast<int>(i), &trainer.adam_v[i]});
    }
    if (i < trainer.has_muon_state.size() && trainer.has_muon_state[i] != 0) {
      refs.push_back({2, static_cast<int>(i), &trainer.muon_momentum[i]});
    }
    if (i < trainer.has_muon_second_moment_state.size() && trainer.has_muon_second_moment_state[i] != 0) {
      refs.push_back({3, static_cast<int>(i), &trainer.muon_second_moment[i]});
    }
    if (i < trainer.has_sgd_state.size() && trainer.has_sgd_state[i] != 0) {
      refs.push_back({4, static_cast<int>(i), &trainer.sgd_momentum[i]});
    }
  }
  return refs;
}

mx::array* optimizer_state_value(mlx_ir::IRTrainer& trainer, int kind, int weight_idx) {
  if (weight_idx < 0 || static_cast<size_t>(weight_idx) >= trainer.weights.size()) {
    return nullptr;
  }
  const auto i = static_cast<size_t>(weight_idx);
  switch (kind) {
    case 0:
      return i < trainer.has_adam_state.size() && trainer.has_adam_state[i] != 0 ? &trainer.adam_m[i] : nullptr;
    case 1:
      return i < trainer.has_adam_state.size() && trainer.has_adam_state[i] != 0 ? &trainer.adam_v[i] : nullptr;
    case 2:
      return i < trainer.has_muon_state.size() && trainer.has_muon_state[i] != 0 ? &trainer.muon_momentum[i] : nullptr;
    case 3:
      return i < trainer.has_muon_second_moment_state.size() && trainer.has_muon_second_moment_state[i] != 0
          ? &trainer.muon_second_moment[i]
          : nullptr;
    case 4:
      return i < trainer.has_sgd_state.size() && trainer.has_sgd_state[i] != 0 ? &trainer.sgd_momentum[i] : nullptr;
    default:
      return nullptr;
  }
}

}  // namespace

extern "C" {

int mlx_ir_trainer_optimizer_state_count(int64_t trainer) {
  try {
    auto* t = get_ir_trainer(trainer);
    if (!t) return -1;
    t->flush();
    return static_cast<int>(optimizer_state_refs(*t).size());
  } catch (...) {
    return -1;
  }
}

int mlx_ir_trainer_optimizer_state_info(
    int64_t trainer, int state_idx, int* kind, int* weight_idx, int* ndim,
    int* shape, int max_ndim, int* size) {
  if (!kind || !weight_idx || !ndim || !shape || max_ndim <= 0 || !size) return -1;
  try {
    auto* t = get_ir_trainer(trainer);
    if (!t) return -1;
    t->flush();
    auto refs = optimizer_state_refs(*t);
    if (state_idx < 0 || static_cast<size_t>(state_idx) >= refs.size()) return -1;
    const auto& ref = refs[static_cast<size_t>(state_idx)];
    const auto dims = static_cast<int>(ref.value->ndim());
    if (dims <= 0 || dims > max_ndim) return -1;
    *kind = ref.kind;
    *weight_idx = ref.weight_idx;
    *ndim = dims;
    *size = static_cast<int>(ref.value->size());
    for (int i = 0; i < dims; ++i) shape[i] = static_cast<int>(ref.value->shape(i));
    return 0;
  } catch (...) {
    return -1;
  }
}

int mlx_ir_trainer_read_optimizer_state(int64_t trainer, int kind, int weight_idx, float* out, int size) {
  if (!out || size <= 0) return -1;
  try {
    auto* t = get_ir_trainer(trainer);
    if (!t) return -1;
    t->flush();
    auto* state = optimizer_state_value(*t, kind, weight_idx);
    if (!state || static_cast<int>(state->size()) != size) return -1;
    auto flat = mx::reshape(mx::astype(*state, mx::float32), {static_cast<mx::ShapeElem>(size)});
    mx::eval(flat);
    std::memcpy(out, flat.data<float>(), static_cast<size_t>(size) * sizeof(float));
    return 0;
  } catch (...) {
    return -1;
  }
}

int mlx_ir_trainer_set_optimizer_state(int64_t trainer, int kind, int weight_idx, const float* data, int size) {
  if (!data || size <= 0) return -1;
  try {
    auto* t = get_ir_trainer(trainer);
    if (!t) return -1;
    t->flush();
    auto* state = optimizer_state_value(*t, kind, weight_idx);
    if (!state || static_cast<int>(state->size()) != size) return -1;
    auto restored = mx::array(data, state->shape(), mx::float32);
    mx::eval(restored);
    *state = restored;
    return 0;
  } catch (...) {
    return -1;
  }
}

int mlx_ir_trainer_set_optimizer_counters(
    int64_t trainer, uint64_t attempted_steps, uint64_t committed_steps,
    uint64_t skipped_steps, uint64_t consecutive_skipped_steps, int last_step_skipped,
    uint64_t last_loss_nonfinite, uint64_t last_gradient_nonfinite, uint64_t last_state_nonfinite) {
  try {
    auto* t = get_ir_trainer(trainer);
    if (!t || attempted_steps > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
        committed_steps > attempted_steps || committed_steps > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
        skipped_steps > attempted_steps || consecutive_skipped_steps > skipped_steps) {
      return -1;
    }
    t->flush();
    t->step_count = static_cast<int>(attempted_steps);
    t->optimizer_step_count = static_cast<int>(committed_steps);
    t->skipped_optimizer_steps = skipped_steps;
    t->consecutive_skipped_optimizer_steps = consecutive_skipped_steps;
    t->last_optimizer_step_skipped = last_step_skipped != 0;
    t->last_optimizer_loss_nonfinite = last_loss_nonfinite;
    t->last_optimizer_gradient_nonfinite = last_gradient_nonfinite;
    t->last_optimizer_state_nonfinite = last_state_nonfinite;
    return 0;
  } catch (...) {
    return -1;
  }
}

}  // extern "C"
