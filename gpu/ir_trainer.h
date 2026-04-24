#ifndef MLX_IR_TRAINER_H
#define MLX_IR_TRAINER_H

#include "ir.h"

#include <mlx/mlx.h>
#include <mlx/random.h>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlx_ir {

enum class OptimizerKind : uint8_t {
  AdamW = 0,
  Muon = 1,
};

enum class QATMode : uint8_t {
  None = 0,
  Int8 = 1,
  Int6 = 2,
};

struct OptimizerGroupConfig {
  OptimizerKind kind = OptimizerKind::AdamW;
  float lr = 0.0f;
  float beta1 = 0.9f;
  float beta2 = 0.95f;
  float eps = 1e-8f;
  float weight_decay = 0.0f;
  int backend_steps = 5;
  bool nesterov = true;
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
  int step_count = 0;
  std::unordered_map<std::string, mlx::core::array> last_outputs;
  bool has_pending_step_ = false;
  mlx::core::array pending_loss_;
  std::unordered_map<std::string, mlx::core::array> pending_outputs_;
  bool has_ready_step_ = false;
  float ready_loss_ = 0.0f;
  std::unordered_map<std::string, mlx::core::array> ready_outputs_;

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
  mlx::core::array read_output(const std::string& output_name) const;
  void apply_optimizer_updates(const std::vector<mlx::core::array>& grads);
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
