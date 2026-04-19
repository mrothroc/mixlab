#include "mlx_bridge.h"
#include "mlx_bridge_internal.h"
#include "ir.h"
#include "ir_trainer.h"

#include <mlx/mlx.h>

#include <cstring>
#include <optional>
#include <cstdint>
#include <memory>
#include <string>
#include <array>
#include <variant>
#include <vector>
#include <cmath>
#include <iostream>
#include <stdexcept>

std::string g_device_name;
bool g_initialized = false;
std::vector<std::optional<mx::array>> g_handle_pool;
std::vector<std::unique_ptr<mlx_ir::IRProgram>> g_ir_program_pool;
std::vector<std::unique_ptr<mlx_ir::IRTrainer>> g_ir_trainer_pool;

void log_bridge_exception(const char* fn, const std::exception& e) {
  std::cerr << "[mlx_bridge] " << fn << " exception: " << e.what() << std::endl;
}

mx::array* get_handle(int64_t handle) {
  if (handle <= 0) {
    return nullptr;
  }
  const size_t idx = static_cast<size_t>(handle - 1);
  if (idx >= g_handle_pool.size() || !g_handle_pool[idx].has_value()) {
    return nullptr;
  }
  return &g_handle_pool[idx].value();
}

int64_t alloc_handle(mx::array&& arr) {
  for (size_t i = 0; i < g_handle_pool.size(); ++i) {
    if (!g_handle_pool[i].has_value()) {
      g_handle_pool[i] = std::move(arr);
      return static_cast<int64_t>(i + 1);
    }
  }
  g_handle_pool.emplace_back(std::move(arr));
  return static_cast<int64_t>(g_handle_pool.size());
}

mlx_ir::IRProgram* get_ir_program(int64_t handle) {
  if (handle <= 0) {
    return nullptr;
  }
  const size_t idx = static_cast<size_t>(handle - 1);
  if (idx >= g_ir_program_pool.size() || !g_ir_program_pool[idx]) {
    return nullptr;
  }
  return g_ir_program_pool[idx].get();
}

int64_t alloc_ir_program(std::unique_ptr<mlx_ir::IRProgram>&& program) {
  for (size_t i = 0; i < g_ir_program_pool.size(); ++i) {
    if (!g_ir_program_pool[i]) {
      g_ir_program_pool[i] = std::move(program);
      return static_cast<int64_t>(i + 1);
    }
  }
  g_ir_program_pool.emplace_back(std::move(program));
  return static_cast<int64_t>(g_ir_program_pool.size());
}

mlx_ir::IRTrainer* get_ir_trainer(int64_t handle) {
  if (handle <= 0) {
    return nullptr;
  }
  const size_t idx = static_cast<size_t>(handle - 1);
  if (idx >= g_ir_trainer_pool.size() || !g_ir_trainer_pool[idx]) {
    return nullptr;
  }
  return g_ir_trainer_pool[idx].get();
}

int64_t alloc_ir_trainer(std::unique_ptr<mlx_ir::IRTrainer>&& trainer) {
  for (size_t i = 0; i < g_ir_trainer_pool.size(); ++i) {
    if (!g_ir_trainer_pool[i]) {
      g_ir_trainer_pool[i] = std::move(trainer);
      return static_cast<int64_t>(i + 1);
    }
  }
  g_ir_trainer_pool.emplace_back(std::move(trainer));
  return static_cast<int64_t>(g_ir_trainer_pool.size());
}

std::optional<const mlx_tensor_input*> find_tensor_input(
    const mlx_tensor_input* inputs,
    int n_inputs,
    const char* name) {
  if (!inputs || n_inputs <= 0 || !name) {
    return std::nullopt;
  }
  for (int i = 0; i < n_inputs; ++i) {
    if (!inputs[i].name) {
      continue;
    }
    if (std::string(inputs[i].name) == name) {
      return &inputs[i];
    }
  }
  return std::nullopt;
}

mlx_ir::TensorMap to_tensor_map(const mlx_tensor_input* inputs, int n_inputs) {
  if (!inputs || n_inputs <= 0) {
    throw std::runtime_error("tensor inputs are empty");
  }
  mlx_ir::TensorMap out;
  out.reserve(static_cast<size_t>(n_inputs));
  for (int i = 0; i < n_inputs; ++i) {
    const auto& in = inputs[i];
    if (!in.name || !in.shape || in.ndim <= 0 || !in.data || in.size_bytes <= 0) {
      throw std::runtime_error("invalid tensor input");
    }
    std::vector<int> shape;
    shape.reserve(static_cast<size_t>(in.ndim));
    for (int d = 0; d < in.ndim; ++d) {
      shape.push_back(in.shape[d]);
    }
    mlx_ir::TensorDesc::DType dtype;
    if (in.dtype == 0) {
      dtype = mlx_ir::TensorDesc::INT32;
    } else if (in.dtype == 1) {
      dtype = mlx_ir::TensorDesc::FLOAT32;
    } else {
      throw std::runtime_error("unsupported tensor dtype");
    }
    out.emplace(
        std::string(in.name),
        mlx_ir::TensorDesc{
            dtype,
            std::move(shape),
            const_cast<void*>(in.data),
            static_cast<size_t>(in.size_bytes),
        });
  }
  return out;
}

extern "C" {

void mlx_set_cuda_graph_limits(int max_ops, int max_mb) {
  if (max_ops > 0) {
    setenv("MLX_MAX_OPS_PER_BUFFER", std::to_string(max_ops).c_str(), 0);
  }
  if (max_mb > 0) {
    setenv("MLX_MAX_MB_PER_BUFFER", std::to_string(max_mb).c_str(), 0);
  }
}

int mlx_init(void) {
  if (g_initialized) {
    return 0;
  }
  try {
    if (!mx::is_available(mx::Device::gpu)) {
      return -1;
    }
    mx::set_default_device(mx::Device(mx::Device::gpu));

    // Force backend initialization.
    auto x = mx::ones({1});
    mx::eval(x);

    const auto& dev = mx::default_device();
    const auto& info = mx::device_info(dev);
    auto it = info.find("device_name");
    if (it != info.end()) {
      if (auto p = std::get_if<std::string>(&it->second)) {
        g_device_name = *p;
      }
    }
    if (g_device_name.empty()) {
      g_device_name = (dev.type == mx::Device::gpu) ? "MLX GPU" : "MLX CPU";
    }

    g_initialized = true;
    return 0;
  } catch (...) {
    g_device_name.clear();
    return -1;
  }
}

const char* mlx_device_name(void) {
  return g_device_name.c_str();
}

int mlx_sgemm(const float* A, const float* B, float* C, int m, int k, int n) {
  if (!A || !B || !C || m <= 0 || k <= 0 || n <= 0) {
    return -1;
  }
  try {
    if (!g_initialized && mlx_init() != 0) {
      return -1;
    }

    // MLX accepts Shape as int32_t values.
    auto a = mx::array(
        const_cast<float*>(A),
        {static_cast<mx::ShapeElem>(m), static_cast<mx::ShapeElem>(k)},
        mx::float32,
        [](void*) {});
    auto b = mx::array(
        const_cast<float*>(B),
        {static_cast<mx::ShapeElem>(k), static_cast<mx::ShapeElem>(n)},
        mx::float32,
        [](void*) {});

    auto c = mx::matmul(a, b);
    mx::eval(c);

    std::memcpy(C, c.data<float>(), static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(float));
    return 0;
  } catch (...) {
    return -1;
  }
}

int mlx_sgemm_transA(const float* A, const float* B, float* C, int m, int k, int n) {
  if (!A || !B || !C || m <= 0 || k <= 0 || n <= 0) {
    return -1;
  }
  try {
    if (!g_initialized && mlx_init() != 0) {
      return -1;
    }

    // C(k,n) = A(m,k)^T * B(m,n)
    auto a = mx::array(
        const_cast<float*>(A),
        {static_cast<mx::ShapeElem>(m), static_cast<mx::ShapeElem>(k)},
        mx::float32,
        [](void*) {});
    auto b = mx::array(
        const_cast<float*>(B),
        {static_cast<mx::ShapeElem>(m), static_cast<mx::ShapeElem>(n)},
        mx::float32,
        [](void*) {});

    auto c = mx::matmul(mx::transpose(a), b);
    mx::eval(c);

    std::memcpy(C, c.data<float>(), static_cast<size_t>(k) * static_cast<size_t>(n) * sizeof(float));
    return 0;
  } catch (...) {
    return -1;
  }
}

int mlx_sgemm_transB(const float* A, const float* B, float* C, int m, int k, int n) {
  if (!A || !B || !C || m <= 0 || k <= 0 || n <= 0) {
    return -1;
  }
  try {
    if (!g_initialized && mlx_init() != 0) {
      return -1;
    }

    // C(m,n) = A(m,k) * B(n,k)^T
    auto a = mx::array(
        const_cast<float*>(A),
        {static_cast<mx::ShapeElem>(m), static_cast<mx::ShapeElem>(k)},
        mx::float32,
        [](void*) {});
    auto b = mx::array(
        const_cast<float*>(B),
        {static_cast<mx::ShapeElem>(n), static_cast<mx::ShapeElem>(k)},
        mx::float32,
        [](void*) {});

    auto c = mx::matmul(a, mx::transpose(b));
    mx::eval(c);

    std::memcpy(C, c.data<float>(), static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(float));
    return 0;
  } catch (...) {
    return -1;
  }
}

int64_t mlx_from_data(const float* data, int rows, int cols) {
  if (!data || rows <= 0 || cols <= 0) {
    return 0;
  }
  try {
    if (!g_initialized && mlx_init() != 0) {
      return 0;
    }
    const size_t total = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    auto* owned = new float[total];
    std::memcpy(owned, data, total * sizeof(float));

    auto arr = mx::array(
        owned,
        {static_cast<mx::ShapeElem>(rows), static_cast<mx::ShapeElem>(cols)},
        mx::float32,
        [](void* p) {
          delete[] static_cast<float*>(p);
        });
    return alloc_handle(std::move(arr));
  } catch (...) {
    return 0;
  }
}

int64_t mlx_from_data_nocopy(const float* data, int rows, int cols) {
  if (!data || rows <= 0 || cols <= 0) {
    return 0;
  }
  try {
    if (!g_initialized && mlx_init() != 0) {
      return 0;
    }
    auto arr = mx::array(
        const_cast<float*>(data),
        {static_cast<mx::ShapeElem>(rows), static_cast<mx::ShapeElem>(cols)},
        mx::float32,
        [](void*) {});
    return alloc_handle(std::move(arr));
  } catch (...) {
    return 0;
  }
}

int64_t mlx_lazy_matmul(int64_t A, int64_t B) {
  try {
    if (!g_initialized && mlx_init() != 0) {
      return 0;
    }
    auto* a = get_handle(A);
    auto* b = get_handle(B);
    if (!a || !b) {
      return 0;
    }
    auto out = mx::matmul(*a, *b);
    return alloc_handle(std::move(out));
  } catch (...) {
    return 0;
  }
}

int64_t mlx_lazy_add(int64_t A, int64_t B) {
  try {
    if (!g_initialized && mlx_init() != 0) {
      return 0;
    }
    auto* a = get_handle(A);
    auto* b = get_handle(B);
    if (!a || !b) {
      return 0;
    }
    auto out = mx::add(*a, *b);
    return alloc_handle(std::move(out));
  } catch (...) {
    return 0;
  }
}

int64_t mlx_lazy_mul(int64_t A, int64_t B) {
  try {
    if (!g_initialized && mlx_init() != 0) {
      return 0;
    }
    auto* a = get_handle(A);
    auto* b = get_handle(B);
    if (!a || !b) {
      return 0;
    }
    auto out = mx::multiply(*a, *b);
    return alloc_handle(std::move(out));
  } catch (...) {
    return 0;
  }
}

int64_t mlx_lazy_sigmoid(int64_t A) {
  try {
    if (!g_initialized && mlx_init() != 0) {
      return 0;
    }
    auto* a = get_handle(A);
    if (!a) {
      return 0;
    }
    auto out = mx::sigmoid(*a);
    return alloc_handle(std::move(out));
  } catch (...) {
    return 0;
  }
}

void mlx_eval_handles(int64_t* handles, int count) {
  if (!handles || count <= 0) {
    return;
  }
  try {
    if (!g_initialized && mlx_init() != 0) {
      return;
    }
    std::vector<mx::array> eval_arrays;
    eval_arrays.reserve(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
      auto* arr = get_handle(handles[i]);
      if (arr) {
        eval_arrays.push_back(*arr);
      }
    }
    if (!eval_arrays.empty()) {
      mx::eval(eval_arrays);
    }
  } catch (...) {
  }
}

void mlx_read_handle(int64_t handle, float* out, int size) {
  if (!out || size <= 0) {
    return;
  }
  try {
    if (!g_initialized && mlx_init() != 0) {
      return;
    }
    auto* arr = get_handle(handle);
    if (!arr) {
      return;
    }
    std::memcpy(out, arr->data<float>(), static_cast<size_t>(size) * sizeof(float));
  } catch (...) {
  }
}

void mlx_free_handle(int64_t handle) {
  if (handle <= 0) {
    return;
  }
  const size_t idx = static_cast<size_t>(handle - 1);
  if (idx >= g_handle_pool.size()) {
    return;
  }
  g_handle_pool[idx].reset();
}

void mlx_free_handles(int64_t* handles, int count) {
  if (!handles || count <= 0) {
    return;
  }
  for (int i = 0; i < count; ++i) {
    mlx_free_handle(handles[i]);
  }
}

int64_t mlx_ir_program_create(int n_weights) {
  if (n_weights <= 0) {
    return 0;
  }
  try {
    auto program = std::make_unique<mlx_ir::IRProgram>();
    program->n_weights = n_weights;
    return alloc_ir_program(std::move(program));
  } catch (...) {
    return 0;
  }
}

void mlx_ir_program_add_op(
    int64_t prog, int op_type,
    const char** inputs, int n_inputs,
    const char** outputs, int n_outputs,
    const float* float_params, int n_float_params,
    const int* int_params, int n_int_params) {
  auto* p = get_ir_program(prog);
  if (!p) {
    return;
  }
  if (n_inputs < 0 || n_inputs > 4 || n_outputs < 0 || n_outputs > 2) {
    return;
  }
  if (n_float_params < 0 || n_float_params > 4 || n_int_params < 0 || n_int_params > 8) {
    return;
  }
  try {
    mlx_ir::IRop op;
    op.type = op_type;
    op.n_inputs = n_inputs;
    op.n_outputs = n_outputs;
    op.n_float_params = n_float_params;
    op.n_int_params = n_int_params;
    for (int i = 0; i < n_inputs; ++i) {
      op.inputs[i] = (inputs && inputs[i]) ? inputs[i] : "";
    }
    for (int i = 0; i < n_outputs; ++i) {
      op.outputs[i] = (outputs && outputs[i]) ? outputs[i] : "";
    }
    for (int i = 0; i < n_float_params; ++i) {
      op.float_params[i] = float_params ? float_params[i] : 0.0f;
    }
    for (int i = 0; i < n_int_params; ++i) {
      op.int_params[i] = int_params ? int_params[i] : 0;
    }
    p->ops.push_back(std::move(op));
  } catch (...) {
  }
}

void mlx_ir_program_destroy(int64_t prog) {
  if (prog <= 0) {
    return;
  }
  const size_t idx = static_cast<size_t>(prog - 1);
  if (idx >= g_ir_program_pool.size()) {
    return;
  }
  g_ir_program_pool[idx].reset();
}

int mlx_ir_eval_program_output_size(
    int64_t program,
    int64_t* weight_handles,
    int n_weights,
    const int* tokens,
    const int* targets,
    int B,
    int T) {
  if (!tokens || !targets || B <= 0 || T <= 0) {
    return -10;
  }
  const size_t n_tok = static_cast<size_t>(B) * static_cast<size_t>(T);
  const int tok_shape[2] = {B, T};
  const int tgt_shape[1] = {static_cast<int>(n_tok)};
  std::vector<mlx_tensor_input> inputs;
  inputs.reserve(2);
  inputs.push_back(mlx_tensor_input{"tokens", 0, tok_shape, 2, tokens, static_cast<int>(n_tok * sizeof(int32_t))});
  inputs.push_back(mlx_tensor_input{"targets", 0, tgt_shape, 1, targets, static_cast<int>(n_tok * sizeof(int32_t))});
  return mlx_ir_eval_program_output_size_named(program, weight_handles, n_weights, inputs.data(), static_cast<int>(inputs.size()));
}

int mlx_ir_eval_program_output_size_named(
    int64_t program,
    int64_t* weight_handles,
    int n_weights,
    const mlx_tensor_input* inputs,
    int n_inputs) {
  return mlx_ir_eval_program_output_size_named_for_output(
      program, weight_handles, n_weights, inputs, n_inputs, nullptr);
}

int mlx_ir_eval_program_output_size_named_for_output(
    int64_t program,
    int64_t* weight_handles,
    int n_weights,
    const mlx_tensor_input* inputs,
    int n_inputs,
    const char* output_name) {
  if (program <= 0 || !weight_handles || n_weights <= 0 || !inputs || n_inputs <= 0) {
    return -10;
  }
  try {
    if (!g_initialized && mlx_init() != 0) {
      return -11;
    }
    auto* prog = get_ir_program(program);
    if (!prog || prog->n_weights != n_weights) {
      return -12;
    }

    std::vector<mx::array> weights;
    weights.reserve(static_cast<size_t>(n_weights));
    for (int i = 0; i < n_weights; ++i) {
      auto* w = get_handle(weight_handles[i]);
      if (!w) {
        return -13;
      }
      weights.push_back(*w);
    }

    auto tensor_map = to_tensor_map(inputs, n_inputs);
    mx::array out = (output_name && output_name[0] != '\0')
                        ? mlx_ir::ir_interpret(*prog, weights, tensor_map, std::string(output_name))
                        : mlx_ir::ir_interpret(*prog, weights, tensor_map);
    return static_cast<int>(out.size());
  } catch (const std::exception& e) {
    log_bridge_exception("mlx_ir_eval_program_output_size", e);
    return -99;
  } catch (...) {
    std::cerr << "[mlx_bridge] mlx_ir_eval_program_output_size unknown exception" << std::endl;
    return -98;
  }
}

int mlx_ir_eval_program(
    int64_t program,
    int64_t* weight_handles,
    int n_weights,
    const int* tokens,
    const int* targets,
    int B,
    int T,
    float* out,
    int out_size) {
  if (!tokens || !targets || B <= 0 || T <= 0 || !out || out_size <= 0) {
    return -1;
  }
  const size_t n_tok = static_cast<size_t>(B) * static_cast<size_t>(T);
  const int tok_shape[2] = {B, T};
  const int tgt_shape[1] = {static_cast<int>(n_tok)};
  std::vector<mlx_tensor_input> inputs;
  inputs.reserve(2);
  inputs.push_back(mlx_tensor_input{"tokens", 0, tok_shape, 2, tokens, static_cast<int>(n_tok * sizeof(int32_t))});
  inputs.push_back(mlx_tensor_input{"targets", 0, tgt_shape, 1, targets, static_cast<int>(n_tok * sizeof(int32_t))});
  return mlx_ir_eval_program_named(program, weight_handles, n_weights, inputs.data(), static_cast<int>(inputs.size()), out, out_size);
}

int mlx_ir_eval_program_named(
    int64_t program,
    int64_t* weight_handles,
    int n_weights,
    const mlx_tensor_input* inputs,
    int n_inputs,
    float* out,
    int out_size) {
  return mlx_ir_eval_program_named_for_output(
      program, weight_handles, n_weights, inputs, n_inputs, nullptr, out, out_size);
}

int mlx_ir_eval_program_named_for_output(
    int64_t program,
    int64_t* weight_handles,
    int n_weights,
    const mlx_tensor_input* inputs,
    int n_inputs,
    const char* output_name,
    float* out,
    int out_size) {
  if (program <= 0 || !weight_handles || n_weights <= 0 || !inputs || n_inputs <= 0 || !out || out_size <= 0) {
    return -1;
  }
  try {
    if (!g_initialized && mlx_init() != 0) {
      return -1;
    }
    auto* prog = get_ir_program(program);
    if (!prog || prog->n_weights != n_weights) {
      return -1;
    }

    std::vector<mx::array> weights;
    weights.reserve(static_cast<size_t>(n_weights));
    for (int i = 0; i < n_weights; ++i) {
      auto* w = get_handle(weight_handles[i]);
      if (!w) {
        return -1;
      }
      weights.push_back(*w);
    }

    auto tensor_map = to_tensor_map(inputs, n_inputs);
    mx::array out_arr = (output_name && output_name[0] != '\0')
                            ? mlx_ir::ir_interpret(*prog, weights, tensor_map, std::string(output_name))
                            : mlx_ir::ir_interpret(*prog, weights, tensor_map);

    const int flat_size = static_cast<int>(out_arr.size());
    if (flat_size != out_size) {
      return -1;
    }
    auto flat = mx::reshape(mx::astype(out_arr, mx::float32), {static_cast<mx::ShapeElem>(flat_size)});
    mx::eval(flat);
    std::memcpy(out, flat.data<float>(), static_cast<size_t>(flat_size) * sizeof(float));
    return 0;
  } catch (...) {
    return -1;
  }
}

void mlx_shutdown(void) {
  g_ir_trainer_pool.clear();
  g_ir_program_pool.clear();
  g_handle_pool.clear();
  g_initialized = false;
  g_device_name.clear();
}

} // extern "C"
