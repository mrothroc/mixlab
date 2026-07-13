#include "mlx_bridge.h"
#include "mlx_bridge_internal.h"
#include "ir.h"
#include "ir_trainer.h"

#include <mlx/compile.h>
#include <mlx/memory.h>
#include <mlx/mlx.h>

#include <cstdlib>
#include <cstring>
#include <optional>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <array>
#include <sstream>
#include <unordered_map>
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
std::unordered_map<std::string, std::function<std::vector<mx::array>(const std::vector<mx::array>&)>>
    g_compiled_ir_grad_pool;
std::unordered_map<std::string, std::function<std::vector<mx::array>(const std::vector<mx::array>&)>>
    g_compiled_ir_eval_pool;

namespace {

std::string env_value_or_unset(const char* name) {
  const char* value = std::getenv(name);
  if (!value || value[0] == '\0') {
    return "unset";
  }
  return value;
}

bool is_cuda_cache_thrash_message(const std::string& message) {
  return message.find("Cache thrashing is happening") != std::string::npos ||
      message.find("MLX_CUDA_GRAPH_CACHE_SIZE") != std::string::npos;
}

std::string bridge_exception_message(const std::exception& e) {
  std::string message = e.what();
  if (!is_cuda_cache_thrash_message(message)) {
    return message;
  }
  return "Cache thrashing is happening. The MLX CUDA buffer-batching limits are too small "
      "for this workload. Increase MLX_MAX_OPS_PER_BUFFER (current " +
      env_value_or_unset("MLX_MAX_OPS_PER_BUFFER") + ") and/or MLX_MAX_MB_PER_BUFFER (current " +
      env_value_or_unset("MLX_MAX_MB_PER_BUFFER") +
      "), and set MLX_CUDA_GRAPH_CACHE_SIZE above the upstream default (current " +
      env_value_or_unset("MLX_CUDA_GRAPH_CACHE_SIZE") + "). For canonical Mamba3, mixlab "
      "auto-tunes these before MLX initializes unless the environment already overrides them.";
}

}  // namespace

void log_bridge_exception(const char* fn, const std::exception& e) {
  std::cout << "[mlx_bridge] " << fn << " exception: " << bridge_exception_message(e) << std::endl;
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

std::string ir_program_fingerprint(const mlx_ir::IRProgram& program) {
  std::ostringstream oss;
  oss << "nw=" << program.n_weights << ";ops=" << program.ops.size();
  for (const auto& op : program.ops) {
    oss << "|t=" << op.type << ";ni=" << op.n_inputs << ";no=" << op.n_outputs;
    for (int i = 0; i < op.n_inputs; ++i) {
      oss << ";i" << i << "=" << op.inputs[i];
    }
    for (int i = 0; i < op.n_outputs; ++i) {
      oss << ";o" << i << "=" << op.outputs[i];
    }
    oss << ";nf=" << op.n_float_params;
    for (int i = 0; i < op.n_float_params; ++i) {
      oss << ";f" << i << "=" << op.float_params[i];
    }
    oss << ";np=" << op.n_int_params;
    for (int i = 0; i < op.n_int_params; ++i) {
      oss << ";p" << i << "=" << op.int_params[i];
    }
  }
  return oss.str();
}

std::string tensor_input_signature(const mlx_tensor_input* inputs, int n_inputs) {
  std::ostringstream oss;
  for (int i = 0; i < n_inputs; ++i) {
    const auto& in = inputs[i];
    oss << "|in=" << (in.name ? in.name : "") << ";dt=" << in.dtype << ";nd=" << in.ndim;
    for (int d = 0; d < in.ndim; ++d) {
      oss << "," << in.shape[d];
    }
  }
  return oss.str();
}

void erase_compiled_ir_grad_cache_for_program(int64_t program) {
  const std::string prefix = std::to_string(program) + "|";
  for (auto it = g_compiled_ir_grad_pool.begin(); it != g_compiled_ir_grad_pool.end();) {
    if (it->first.rfind(prefix, 0) == 0) {
      it = g_compiled_ir_grad_pool.erase(it);
    } else {
      ++it;
    }
  }
}

void erase_compiled_ir_eval_cache_for_program(int64_t program) {
  const std::string prefix = std::to_string(program) + "|";
  for (auto it = g_compiled_ir_eval_pool.begin(); it != g_compiled_ir_eval_pool.end();) {
    if (it->first.rfind(prefix, 0) == 0) {
      it = g_compiled_ir_eval_pool.erase(it);
    } else {
      ++it;
    }
  }
}

extern "C" {

void mlx_set_cuda_graph_limits(int max_ops, int max_mb) {
  if (max_ops > 0) {
    setenv("MLX_MAX_OPS_PER_BUFFER", std::to_string(max_ops).c_str(), 1);
  }
  if (max_mb > 0) {
    setenv("MLX_MAX_MB_PER_BUFFER", std::to_string(max_mb).c_str(), 1);
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

uint64_t mlx_memory_active(void) {
  try {
    return static_cast<uint64_t>(mx::get_active_memory());
  } catch (...) {
    return 0;
  }
}

uint64_t mlx_memory_cache(void) {
  try {
    return static_cast<uint64_t>(mx::get_cache_memory());
  } catch (...) {
    return 0;
  }
}

uint64_t mlx_memory_peak(void) {
  try {
    return static_cast<uint64_t>(mx::get_peak_memory());
  } catch (...) {
    return 0;
  }
}

void mlx_memory_clear_cache(void) {
  try {
    mx::clear_cache();
  } catch (...) {
  }
}

uint64_t mlx_memory_set_memory_limit(uint64_t limit) {
  try {
    return static_cast<uint64_t>(mx::set_memory_limit(static_cast<size_t>(limit)));
  } catch (...) {
    return 0;
  }
}

uint64_t mlx_memory_set_cache_limit(uint64_t limit) {
  try {
    return static_cast<uint64_t>(mx::set_cache_limit(static_cast<size_t>(limit)));
  } catch (...) {
    return 0;
  }
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

int64_t mlx_from_data_shape(const float* data, const int* shape, int ndim) {
  if (!data || !shape || ndim <= 0) {
    return 0;
  }
  try {
    if (!g_initialized && mlx_init() != 0) {
      return 0;
    }
    mx::Shape mx_shape;
    mx_shape.reserve(static_cast<size_t>(ndim));
    size_t total = 1;
    for (int i = 0; i < ndim; ++i) {
      if (shape[i] <= 0) {
        return 0;
      }
      mx_shape.push_back(static_cast<mx::ShapeElem>(shape[i]));
      total *= static_cast<size_t>(shape[i]);
    }
    auto* owned = new float[total];
    std::memcpy(owned, data, total * sizeof(float));

    auto arr = mx::array(
        owned,
        mx_shape,
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
  if (n_inputs < 0 || n_inputs > 256 || n_outputs < 0 || n_outputs > 8) {
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
    erase_compiled_ir_grad_cache_for_program(prog);
    erase_compiled_ir_eval_cache_for_program(prog);
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
  erase_compiled_ir_grad_cache_for_program(prog);
  erase_compiled_ir_eval_cache_for_program(prog);
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

    auto tensor_map = mlx_ir::tensor_map_to_arrays(to_tensor_map(inputs, n_inputs));
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

int mlx_ir_eval_program_named_outputs(
    int64_t program,
    int64_t* weight_handles,
    int n_weights,
    const mlx_tensor_input* inputs,
    int n_inputs,
    const char** output_names,
    int n_outputs,
    float** outs,
    const int* out_sizes) {
  if (program <= 0 || !weight_handles || n_weights <= 0 || !inputs || n_inputs <= 0 ||
      !output_names || n_outputs <= 0 || !outs || !out_sizes) {
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

    std::vector<std::string> names;
    names.reserve(static_cast<size_t>(n_outputs));
    for (int i = 0; i < n_outputs; ++i) {
      if (!output_names[i] || output_names[i][0] == '\0' || !outs[i] || out_sizes[i] <= 0) {
        return -1;
      }
      names.emplace_back(output_names[i]);
    }

    auto tensor_map = to_tensor_map(inputs, n_inputs);
    auto output_map = mlx_ir::ir_interpret_outputs(*prog, weights, tensor_map, names);
    for (int i = 0; i < n_outputs; ++i) {
      auto it = output_map.find(names[i]);
      if (it == output_map.end()) {
        return -1;
      }
      mx::array out_arr = it->second;
      const int flat_size = static_cast<int>(out_arr.size());
      if (flat_size != out_sizes[i]) {
        return -1;
      }
      auto flat = mx::reshape(mx::astype(out_arr, mx::float32), {static_cast<mx::ShapeElem>(flat_size)});
      mx::eval(flat);
      std::memcpy(outs[i], flat.data<float>(), static_cast<size_t>(flat_size) * sizeof(float));
    }
    return 0;
  } catch (const std::exception& e) {
    log_bridge_exception("mlx_ir_eval_program_named_outputs", e);
    return -1;
  } catch (...) {
    std::cerr << "[mlx_bridge] mlx_ir_eval_program_named_outputs unknown exception" << std::endl;
    return -1;
  }
}

int mlx_ir_eval_program_handle_outputs(
    int64_t program,
    int64_t* weight_handles,
    int n_weights,
    const mlx_tensor_input* inputs,
    int n_inputs,
    const char** handle_input_names,
    const int64_t* handle_inputs,
    int n_handle_inputs,
    const char** output_names,
    int n_outputs,
    int64_t* output_handles) {
  if (program <= 0 || !weight_handles || n_weights <= 0 || !inputs || n_inputs <= 0 ||
      !handle_input_names || !handle_inputs || n_handle_inputs <= 0 ||
      !output_names || n_outputs <= 0 || !output_handles) {
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
      auto* weight = get_handle(weight_handles[i]);
      if (!weight) {
        return -1;
      }
      weights.push_back(*weight);
    }

    auto tensor_map = mlx_ir::tensor_map_to_arrays(to_tensor_map(inputs, n_inputs));
    std::vector<std::string> input_names;
    input_names.reserve(static_cast<size_t>(n_inputs + n_handle_inputs));
    std::vector<mx::array> args;
    args.reserve(static_cast<size_t>(n_weights + n_inputs + n_handle_inputs));
    args.insert(args.end(), weights.begin(), weights.end());
    for (int i = 0; i < n_inputs; ++i) {
      input_names.emplace_back(inputs[i].name);
      args.push_back(tensor_map.at(input_names.back()));
    }
    for (int i = 0; i < n_handle_inputs; ++i) {
      if (!handle_input_names[i] || handle_input_names[i][0] == '\0') {
        return -1;
      }
      auto* input = get_handle(handle_inputs[i]);
      if (!input) {
        return -1;
      }
      tensor_map.insert_or_assign(std::string(handle_input_names[i]), *input);
      input_names.emplace_back(handle_input_names[i]);
      args.push_back(*input);
    }

    std::vector<std::string> names;
    names.reserve(static_cast<size_t>(n_outputs));
    for (int i = 0; i < n_outputs; ++i) {
      if (!output_names[i] || output_names[i][0] == '\0') {
        return -1;
      }
      names.emplace_back(output_names[i]);
      output_handles[i] = 0;
    }
    std::ostringstream key;
    key << program << "|" << ir_program_fingerprint(*prog)
        << "|handle_eval|nw=" << n_weights << tensor_input_signature(inputs, n_inputs);
    for (int i = 0; i < n_handle_inputs; ++i) {
      auto* input = get_handle(handle_inputs[i]);
      key << "|hi=" << handle_input_names[i] << ":";
      for (auto dim : input->shape()) {
        key << dim << ",";
      }
      key << ":dtype=" << input->dtype();
    }
    for (const auto& name : names) {
      key << "|out=" << name;
    }
    auto compiled_it = g_compiled_ir_eval_pool.find(key.str());
    if (compiled_it == g_compiled_ir_eval_pool.end()) {
      auto compiled = mx::compile(
          [prog, input_names, names, n_weights](const std::vector<mx::array>& fn_args) {
            if (fn_args.size() != static_cast<size_t>(n_weights) + input_names.size()) {
              throw std::runtime_error("compiled handle evaluation argument count mismatch");
            }
            std::vector<mx::array> local_weights;
            local_weights.reserve(static_cast<size_t>(n_weights));
            for (int i = 0; i < n_weights; ++i) {
              local_weights.push_back(fn_args[static_cast<size_t>(i)]);
            }
            mlx_ir::ArrayMap local_inputs;
            local_inputs.reserve(input_names.size());
            for (size_t i = 0; i < input_names.size(); ++i) {
              local_inputs.emplace(input_names[i], fn_args[static_cast<size_t>(n_weights) + i]);
            }
            auto local_outputs = mlx_ir::ir_interpret_outputs(*prog, local_weights, local_inputs, names);
            std::vector<mx::array> values;
            values.reserve(names.size());
            for (const auto& name : names) {
              values.push_back(local_outputs.at(name));
            }
            return values;
          },
          false);
      compiled_it = g_compiled_ir_eval_pool.emplace(key.str(), std::move(compiled)).first;
    }
    auto evaluated = compiled_it->second(args);
    if (evaluated.size() != names.size()) {
      return -1;
    }
    mx::eval(evaluated);
    for (int i = 0; i < n_outputs; ++i) {
      output_handles[i] = alloc_handle(std::move(evaluated[static_cast<size_t>(i)]));
      if (output_handles[i] <= 0) {
        for (int j = 0; j < i; ++j) {
          mlx_free_handle(output_handles[j]);
          output_handles[j] = 0;
        }
        return -1;
      }
    }
    return 0;
  } catch (const std::exception& e) {
    log_bridge_exception("mlx_ir_eval_program_handle_outputs", e);
    return -1;
  } catch (...) {
    std::cerr << "[mlx_bridge] mlx_ir_eval_program_handle_outputs unknown exception" << std::endl;
    return -1;
  }
}

int mlx_ir_eval_program_grads_named_for_output(
    int64_t program,
    int64_t* weight_handles,
    int n_weights,
    const mlx_tensor_input* inputs,
    int n_inputs,
    const char* output_name,
    float* loss_out,
    float** grad_out_ptrs,
    int* grad_sizes) {
  if (program <= 0 || !weight_handles || n_weights <= 0 || !inputs || n_inputs <= 0 ||
      !loss_out || !grad_out_ptrs || !grad_sizes) {
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
    auto input_arrays_by_name = mlx_ir::tensor_map_to_arrays(tensor_map);
    std::vector<std::string> input_names;
    input_names.reserve(static_cast<size_t>(n_inputs));
    std::vector<mx::array> input_arrays;
    input_arrays.reserve(static_cast<size_t>(n_inputs));
    for (int i = 0; i < n_inputs; ++i) {
      std::string name(inputs[i].name);
      auto it = input_arrays_by_name.find(name);
      if (it == input_arrays_by_name.end()) {
        return -1;
      }
      input_names.push_back(std::move(name));
      input_arrays.push_back(it->second);
    }

    std::vector<mx::array> args;
    args.reserve(static_cast<size_t>(n_weights + n_inputs));
    args.insert(args.end(), weights.begin(), weights.end());
    args.insert(args.end(), input_arrays.begin(), input_arrays.end());

    std::vector<int> argnums(static_cast<size_t>(n_weights));
    std::iota(argnums.begin(), argnums.end(), 0);
    const std::string requested_output = (output_name && output_name[0] != '\0') ? std::string(output_name) : std::string();

    const std::string cache_key = std::to_string(program) + "|" + ir_program_fingerprint(*prog) +
        "|out=" + requested_output + "|nw=" + std::to_string(n_weights) +
        tensor_input_signature(inputs, n_inputs);
    auto compiled_it = g_compiled_ir_grad_pool.find(cache_key);
    if (compiled_it == g_compiled_ir_grad_pool.end()) {
      auto grad_fn = mx::value_and_grad(
          [prog, input_names, requested_output, n_weights](const std::vector<mx::array>& fn_args) {
            if (static_cast<int>(fn_args.size()) < n_weights + static_cast<int>(input_names.size())) {
              throw std::runtime_error("compiled IR gradient argument count mismatch");
            }
            std::vector<mx::array> w;
            w.reserve(static_cast<size_t>(n_weights));
            for (int i = 0; i < n_weights; ++i) {
              w.push_back(fn_args[static_cast<size_t>(i)]);
            }
            mlx_ir::ArrayMap input_map;
            input_map.reserve(input_names.size());
            for (size_t i = 0; i < input_names.size(); ++i) {
              input_map.emplace(input_names[i], fn_args[static_cast<size_t>(n_weights) + i]);
            }
            auto out = requested_output.empty()
                ? mlx_ir::ir_interpret(*prog, w, input_map)
                : mlx_ir::ir_interpret(*prog, w, input_map, requested_output);
            if (out.size() != 1) {
              throw std::runtime_error("IR gradient output must be scalar");
            }
            return mx::reshape(out, {});
          },
          argnums);

      auto compiled = mx::compile(
          [grad_fn](const std::vector<mx::array>& fn_args) {
            auto result = grad_fn(fn_args);
            std::vector<mx::array> out;
            out.reserve(result.second.size() + 1);
            out.push_back(result.first);
            out.insert(out.end(), result.second.begin(), result.second.end());
            return out;
          },
          false);
      compiled_it = g_compiled_ir_grad_pool.emplace(cache_key, std::move(compiled)).first;
    }

    auto compiled_out = compiled_it->second(args);
    if (static_cast<int>(compiled_out.size()) != n_weights + 1) {
      return -1;
    }
    auto loss = compiled_out[0];

    std::vector<mx::array> eval_arrays;
    eval_arrays.reserve(compiled_out.size());
    eval_arrays.insert(eval_arrays.end(), compiled_out.begin(), compiled_out.end());
    mx::eval(eval_arrays);

    *loss_out = loss.item<float>();
    for (int i = 0; i < n_weights; ++i) {
      const auto& g = compiled_out[static_cast<size_t>(i + 1)];
      if (g.dtype() != mx::float32 || g.size() != static_cast<size_t>(grad_sizes[i]) || grad_out_ptrs[i] == nullptr) {
        return -1;
      }
      std::memcpy(grad_out_ptrs[i], g.data<float>(), g.size() * sizeof(float));
    }
    return 0;
  } catch (...) {
    return -1;
  }
}

void mlx_shutdown(void) {
  g_compiled_ir_grad_pool.clear();
  g_ir_trainer_pool.clear();
  g_ir_program_pool.clear();
  g_handle_pool.clear();
  g_initialized = false;
  g_device_name.clear();
}

} // extern "C"
