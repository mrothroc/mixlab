#ifndef MLX_IR_TRAINER_H
#define MLX_IR_TRAINER_H

#include "ir.h"

#include <mlx/mlx.h>
#include <mlx/random.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlx_ir {

enum class OptimizerKind : uint8_t {
  AdamW = 0,
  Muon = 1,
  SGD = 2,
};

enum class QATMode : uint8_t {
  None = 0,
  Int8 = 1,
  Int6 = 2,
};

enum class NewtonSchulzVariant : uint8_t {
  Fixed = 0,
  PolarExpress = 1,
};

enum class MuonNormalization : uint8_t {
  None = 0,
  RowL2 = 1,
  NorMuon = 2,
};

struct OptimizerGroupConfig {
  OptimizerKind kind = OptimizerKind::AdamW;
  float lr = 0.0f;
  float beta1 = 0.9f;
  float beta2 = 0.95f;
  float eps = 1e-8f;
  float weight_decay = 0.0f;
  bool cautious_weight_decay = false;
  int cautious_weight_decay_activation_step = 0;
  int backend_steps = 5;
  NewtonSchulzVariant newton_schulz_variant = NewtonSchulzVariant::Fixed;
  bool nesterov = true;
  MuonNormalization muon_normalization = MuonNormalization::None;
};

struct WeightOptimizerSpec {
  uint32_t group_index = 0;
  bool decay = false;
};

struct IRTrainer {
  IRTrainer();

  IRProgram program;

  std::vector<mlx::core::array> weights;
  std::vector<OptimizerGroupConfig> optimizer_groups;
  std::vector<WeightOptimizerSpec> weight_optimizers;
  std::vector<mlx::core::array> adam_m;
  std::vector<mlx::core::array> adam_v;
  std::vector<uint8_t> has_adam_state;
  std::vector<mlx::core::array> muon_momentum;
  std::vector<uint8_t> has_muon_state;
  std::vector<mlx::core::array> muon_second_moment;
  std::vector<uint8_t> has_muon_second_moment_state;
  std::vector<mlx::core::array> sgd_momentum;
  std::vector<uint8_t> has_sgd_state;
  int step_count = 0;
  std::unordered_map<std::string, mlx::core::array> last_outputs;
  std::vector<mlx::core::array> last_grads;
  bool has_pending_step_ = false;
  mlx::core::array pending_loss_;
  std::unordered_map<std::string, mlx::core::array> pending_outputs_;
  int pending_step_index_ = 0;
  bool has_ready_step_ = false;
  float ready_loss_ = 0.0f;
  std::unordered_map<std::string, mlx::core::array> ready_outputs_;
  int ready_step_index_ = 0;
  bool memory_safe_step_notice_logged_ = false;
  bool low_memory_update_notice_logged_ = false;
  std::function<std::vector<mlx::core::array>(const std::vector<mlx::core::array>&)> compiled_named_step;
  std::string compiled_named_step_signature;
  std::function<std::vector<mlx::core::array>(const std::vector<mlx::core::array>&)> compiled_named_update_step;
  std::string compiled_named_update_step_signature;
  std::function<std::vector<mlx::core::array>(const std::vector<mlx::core::array>&)> compiled_mamba3_optimizer_update;
  std::string compiled_mamba3_optimizer_update_signature;
  std::vector<std::function<std::vector<mlx::core::array>(const std::vector<mlx::core::array>&)>> compiled_mamba3_grad_chunks;
  std::string compiled_mamba3_grad_chunks_signature;
  bool compiled_mamba3_grad_chunks_disabled = false;
  bool compiled_mamba3_grad_chunks_fallback_logged = false;
  int adaptive_mamba3_grad_chunk_elements = 0;
  bool adaptive_mamba3_grad_chunk_fallback_logged = false;
  bool fused_mamba3_compiled_optimizer_update_disabled = false;
  bool fused_mamba3_compiled_optimizer_update_notice_logged = false;
  bool fused_mamba3_compiled_optimizer_update_fallback_logged = false;
  bool fused_mamba3_compiled_update_step_disabled = false;
  bool fused_mamba3_compiled_update_step_notice_logged = false;
  bool fused_mamba3_compiled_update_step_fallback_logged = false;
  bool fused_mamba3_compiled_step_disabled = false;
  bool fused_mamba3_compiled_step_notice_logged = false;
  bool fused_mamba3_compiled_step_fallback_logged = false;
  bool mamba3_single_backward_disabled = false;
  bool mamba3_single_backward_fallback_logged = false;

  float max_grad_norm = 0.0f;
  float lr_scale = 1.0f;
  float default_base_lr = 0.0f;
  QATMode qat_mode = QATMode::None;

  float step(const mlx::core::array& tokens, const mlx::core::array& targets);
  float step_named(const TensorMap& inputs);
  void submit_step(const TensorMap& inputs);
  float collect_loss();
  void flush();
  float evaluate(const mlx::core::array& tokens, const mlx::core::array& targets);
  float evaluate_named(const TensorMap& inputs);
  std::vector<float> evaluate_per_token(const TensorMap& inputs);
  float evaluate_lora_named(const TensorMap& inputs, int rank, int steps, float lr);
  float compute_mean_square_grads_named(const TensorMap& inputs, const std::string& output_name);
  mlx::core::array read_output(const std::string& output_name) const;
  mlx::core::array read_grad(int weight_idx) const;
  void set_program(const IRProgram& new_program);
  void apply_optimizer_updates(const std::vector<mlx::core::array>& grads);
  void apply_weight_optimizer_update(size_t weight_index, const mlx::core::array& grad);
  void collect_weight_state_for_eval(size_t weight_index, std::vector<mlx::core::array>& eval_arrays) const;
};

std::unique_ptr<IRTrainer> create_ir_trainer(
    const IRProgram& program,
    const std::vector<mlx::core::array>& initial_weights,
    const std::vector<WeightOptimizerSpec>& weight_optimizers,
    const std::vector<OptimizerGroupConfig>& optimizer_groups,
    float max_grad_norm,
    float default_base_lr);

} // namespace mlx_ir

#endif
