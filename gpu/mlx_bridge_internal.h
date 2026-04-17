#ifndef MLX_BRIDGE_INTERNAL_H
#define MLX_BRIDGE_INTERNAL_H

#include "ir.h"
#include "ir_trainer.h"
#include "mlx_bridge.h"

#include <memory>
#include <mlx/mlx.h>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace mx = mlx::core;

extern std::string g_device_name;
extern bool g_initialized;
extern std::vector<std::optional<mx::array>> g_handle_pool;
extern std::vector<std::unique_ptr<mlx_ir::IRProgram>> g_ir_program_pool;
extern std::vector<std::unique_ptr<mlx_ir::IRTrainer>> g_ir_trainer_pool;

void log_bridge_exception(const char* fn, const std::exception& e);
mx::array* get_handle(int64_t handle);
int64_t alloc_handle(mx::array&& arr);
mlx_ir::IRProgram* get_ir_program(int64_t handle);
int64_t alloc_ir_program(std::unique_ptr<mlx_ir::IRProgram>&& program);
mlx_ir::IRTrainer* get_ir_trainer(int64_t handle);
int64_t alloc_ir_trainer(std::unique_ptr<mlx_ir::IRTrainer>&& trainer);
std::optional<const mlx_tensor_input*> find_tensor_input(
    const mlx_tensor_input* inputs,
    int n_inputs,
    const char* name);
mlx_ir::TensorMap to_tensor_map(const mlx_tensor_input* inputs, int n_inputs);

#endif
