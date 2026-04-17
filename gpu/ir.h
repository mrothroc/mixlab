#ifndef MLX_IR_H
#define MLX_IR_H

#include <mlx/mlx.h>

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlx_ir {

enum OpType {
  OP_EMBED = 1,
  OP_MATMUL = 2,
  OP_ADD = 3,
  OP_MUL = 4,
  OP_SCALAR_MUL = 5,
  OP_SIGMOID = 6,
  OP_SILU = 7,
  OP_SOFTMAX = 8,
  OP_RESHAPE = 9,
  OP_TRANSPOSE = 10,
  OP_SLICE = 11,
  OP_CONCAT = 12,
  OP_CAUSAL_MASK = 13,
  OP_CROSS_ENTROPY = 14,
  OP_SQUARE = 20,
  OP_SUB = 21,
  OP_DIV = 22,
  OP_CUMSUM = 23,
  OP_ARGSORT = 24,
  OP_WHERE = 25,
  OP_LESS_THAN = 26,
  OP_GREATER_EQ = 27,
  OP_ARANGE = 28,
  OP_MEAN_AXIS = 29,
  OP_FULL = 30,
  OP_ASTYPE = 31,
  OP_RUNNING_VAR = 32,
  OP_RMSNORM = 33,
  OP_ROPE = 34,
  OP_SQRT = 35,
  OP_RSQRT = 36,
  OP_SIN = 37,
  OP_COS = 38,
  OP_EXP = 39,
  OP_OUTER = 40,
  OP_SQUEEZE = 41,
  OP_GELU = 42,
  OP_RELU = 43,
  OP_TANH = 44,
  OP_ROPE_STRIDED = 47,
  OP_PREFIX_CAUSAL_MASK = 48,
  OP_SCAN = 49,
  OP_GRADIENT_MAGNITUDES = 50,
};

struct IRop {
  int type = 0;
  std::string inputs[4];
  int n_inputs = 0;
  std::string outputs[2];
  int n_outputs = 0;

  float float_params[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  int n_float_params = 0;

  int int_params[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  int n_int_params = 0;
};

struct IRProgram {
  std::vector<IRop> ops;
  int n_weights = 0;
};

struct TensorDesc {
  enum DType { INT32, FLOAT32 };
  DType dtype;
  std::vector<int> shape;
  void* data;
  size_t size_bytes;
};

using TensorMap = std::unordered_map<std::string, TensorDesc>;

mlx::core::array ir_interpret(
    const IRProgram& program,
    const std::vector<mlx::core::array>& weights,
    const TensorMap& inputs);

mlx::core::array ir_interpret(
    const IRProgram& program,
    const std::vector<mlx::core::array>& weights,
    const TensorMap& inputs,
    const std::string& output_name);

std::unordered_map<std::string, mlx::core::array> ir_interpret_outputs(
    const IRProgram& program,
    const std::vector<mlx::core::array>& weights,
    const TensorMap& inputs,
    const std::vector<std::string>& output_names);

mlx::core::array ir_interpret(
    const IRProgram& program,
    const std::vector<mlx::core::array>& weights,
    const mlx::core::array& tokens,
    const mlx::core::array& targets);

} // namespace mlx_ir

#endif
