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
  OP_DROPOUT = 15,
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
  OP_GATHER_POSITIONS = 51,
  OP_SCATTER_POSITIONS = 52,
  OP_ROPE_INDEXED = 53,
  OP_LEAKY_RELU = 54,
  OP_XSA_PROJECT = 55,
  OP_CROSS_ENTROPY_PER_TOKEN = 56,
  OP_MATRIX_SCAN = 57,
  OP_SCAN_TV = 58,  // time-varying gated scan
  OP_SOFTPLUS = 59,
  OP_GATED_DELTA_SCAN = 60,
  OP_STOP_GRADIENT = 61,
  OP_DEPTHWISE_CONV1D = 62,
  OP_MAMBA3_SELECTIVE_SCAN = 63,
  OP_MAMBA3_CANONICAL_BLOCK = 64,
  OP_RANDOM_NORMAL = 65,
  OP_FIRST_BYTE_MASKED_CROSS_ENTROPY = 66,
  OP_MASKED_CROSS_ENTROPY = 67,
  OP_MASKED_CROSS_ENTROPY_PER_TOKEN = 68,
  OP_DISTILLATION_KL = 69,
  OP_HGRN2_SCAN = 70,
  OP_MLSTM_SCAN = 71,
  OP_DEBERTA_RELATIVE_BIAS = 72,
  OP_CHAR_FEATURE_BAG = 73,
  OP_MOE_FEED_FORWARD = 74,
  OP_MASKED_SMOOTH_L1 = 75,
  OP_Z_LOSS = 76,
  OP_LOG = 77,
  OP_RECIPROCAL = 78,
  OP_POW = 79,
  OP_ABS = 80,
  OP_CLAMP = 81,
  OP_MINIMUM = 82,
  OP_MAXIMUM = 83,
  OP_GREATER_THAN = 84,
  OP_LESS_EQ = 85,
  OP_EQUAL = 86,
  OP_LAYERNORM = 87,
  OP_SELECTIVE_CAUSAL_MASK = 88,
  OP_SEGMENT_ATTENTION_MASK = 89,
  OP_BLOCK_DIFFUSION_MASK = 90,
  OP_GELU_EXACT = 91,
  OP_MASKED_BCE_WITH_LOGITS = 92,
  OP_MASKED_BINARY_ACCURACY = 93,
  OP_ENERGY_PAIRWISE_LOSS = 94,
  OP_ENERGY_SPAN_POOL = 95,
  OP_ENERGY_SPAN_PAIRWISE_LOSS = 96,
  OP_SPAN_PLL_POOL = 97,
  OP_SPAN_PLL_PAIRWISE_LOSS = 98,
  OP_MASKED_DISTILLATION_KL = 99,
  OP_MASKED_SYMMETRIC_KL = 100,
  OP_MASKED_MARGIN_PLL = 101,
};

struct IRop {
  int type = 0;
  std::string inputs[256];
  int n_inputs = 0;
  std::string outputs[4];
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
using ArrayMap = std::unordered_map<std::string, mlx::core::array>;

ArrayMap tensor_map_to_arrays(const TensorMap& inputs);

mlx::core::array ir_interpret(
    const IRProgram& program,
    const std::vector<mlx::core::array>& weights,
    const TensorMap& inputs);

mlx::core::array ir_interpret(
    const IRProgram& program,
    const std::vector<mlx::core::array>& weights,
    const TensorMap& inputs,
    const std::string& output_name);

mlx::core::array ir_interpret(
    const IRProgram& program,
    const std::vector<mlx::core::array>& weights,
    const TensorMap& inputs,
    const std::string& output_name,
    bool training);

mlx::core::array ir_interpret(
    const IRProgram& program,
    const std::vector<mlx::core::array>& weights,
    const ArrayMap& inputs);

mlx::core::array ir_interpret(
    const IRProgram& program,
    const std::vector<mlx::core::array>& weights,
    const ArrayMap& inputs,
    const std::string& output_name);

mlx::core::array ir_interpret(
    const IRProgram& program,
    const std::vector<mlx::core::array>& weights,
    const ArrayMap& inputs,
    const std::string& output_name,
    bool training);

std::unordered_map<std::string, mlx::core::array> ir_interpret_outputs(
    const IRProgram& program,
    const std::vector<mlx::core::array>& weights,
    const TensorMap& inputs,
    const std::vector<std::string>& output_names);

std::unordered_map<std::string, mlx::core::array> ir_interpret_outputs(
    const IRProgram& program,
    const std::vector<mlx::core::array>& weights,
    const TensorMap& inputs,
    const std::vector<std::string>& output_names,
    bool training);

std::unordered_map<std::string, mlx::core::array> ir_interpret_outputs(
    const IRProgram& program,
    const std::vector<mlx::core::array>& weights,
    const ArrayMap& inputs,
    const std::vector<std::string>& output_names);

std::unordered_map<std::string, mlx::core::array> ir_interpret_outputs(
    const IRProgram& program,
    const std::vector<mlx::core::array>& weights,
    const ArrayMap& inputs,
    const std::vector<std::string>& output_names,
    bool training);

void report_gated_delta_timing_summary(const char* phase, int index);

mlx::core::array ir_interpret(
    const IRProgram& program,
    const std::vector<mlx::core::array>& weights,
    const mlx::core::array& tokens,
    const mlx::core::array& targets);

mlx::core::array ir_interpret(
    const IRProgram& program,
    const std::vector<mlx::core::array>& weights,
    const mlx::core::array& tokens,
    const mlx::core::array& targets,
    bool training);

} // namespace mlx_ir

#endif
