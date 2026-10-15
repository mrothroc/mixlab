#include "mlx_bridge_internal.h"
#include <vector>

extern "C" {

int mlx_ir_trainer_set_local_accumulation(int64_t trainer, int steps) {
  try {
    auto t = get_ir_trainer(trainer);
    if (!t || steps < 1 || t->step_count != 0 || t->uses_staged_updates()) return -1;
    t->flush();
    auto plan = mlx_ir::build_gradient_bucket_plan(t->weights);
    t->distributed_gradient_bucket_plan = std::move(plan);
    t->distributed_accumulation_steps = steps;
    t->local_accumulation_active = true;
    return 0;
  } catch (...) { return -1; }
}

// Validate and materialize the entire candidate before a single vector swap.
// Optimizer tensors/counters and private accumulated state are never replaced.
int mlx_ir_trainer_install_weights(int64_t trainer, int64_t* handles, int count) {
  try {
    auto t = get_ir_trainer(trainer);
    if (!t || !handles || count <= 0 || size_t(count) != t->weights.size() ||
        t->distributed_accumulation_position != 0) return -1;
    t->flush();
    std::vector<mx::array> candidate;
    std::vector<mx::array> finite;
    candidate.reserve(count);
    for (int i = 0; i < count; ++i) {
      auto p = get_handle(handles[i]);
      if (!p || p->shape() != t->weights[i].shape() || p->dtype() != mx::float32) return -1;
      candidate.push_back(*p);
      finite.push_back(mx::all(mx::isfinite(*p)));
    }
    mx::eval(candidate);
    mx::eval(finite);
    for (auto& f : finite) if (!f.item<bool>()) return -1;
    t->weights.swap(candidate);
    return 0;
  } catch (...) { return -1; }
}
}
