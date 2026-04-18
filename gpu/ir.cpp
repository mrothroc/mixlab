#include "ir.h"

#include <mlx/random.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace mx = mlx::core;

namespace {

mx::Shape make_shape(const int* vals, int n) {
  mx::Shape s;
  s.reserve(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    s.push_back(static_cast<mx::ShapeElem>(vals[i]));
  }
  return s;
}

mx::array cross_entropy_mean(const mx::array& logits, const mx::array& targets) {
  auto row_max = mx::max(logits, 1, true);
  auto shifted = logits - row_max;
  auto log_norm = mx::log(mx::sum(mx::exp(shifted), 1, true));
  auto log_probs = shifted - log_norm;
  auto idx = mx::reshape(targets, {targets.shape(0), 1});
  auto chosen = mx::take_along_axis(log_probs, idx, 1);
  return -mx::mean(chosen);
}

mx::array running_variance_raw(const mx::array& x_flat, int B, int T, int D, float alpha) {
  auto x = mx::reshape(x_flat, {B, T, D});
  auto out = mx::zeros({B, T, D}, mx::float32);
  const float one_minus_alpha = 1.0f - alpha;
  for (int b = 0; b < B; ++b) {
    auto x_bt = mx::slice(x, {b, 0, 0}, {b + 1, T, D});
    auto x0 = mx::reshape(mx::slice(x_bt, {0, 0, 0}, {1, 1, D}), {D});
    auto mean = x0;
    auto vari = mx::full({D}, 0.01f, mx::float32);
    for (int t = 0; t < T; ++t) {
      auto xt = mx::reshape(mx::slice(x_bt, {0, t, 0}, {1, t + 1, D}), {D});
      auto diff = xt - mean;
      mean = one_minus_alpha * mean + alpha * xt;
      vari = one_minus_alpha * vari + alpha * mx::square(diff);
      out = mx::slice_update(out, mx::reshape(vari, {1, 1, D}), mx::Shape{b, t, 0}, mx::Shape{b + 1, t + 1, D});
    }
  }
  return out;
}

mx::array tensor_desc_to_array(const mlx_ir::TensorDesc& desc) {
  mx::Shape shape;
  shape.reserve(desc.shape.size());
  size_t elem_count = 1;
  for (int dim : desc.shape) {
    if (dim < 0) {
      throw std::runtime_error("negative tensor dimension");
    }
    shape.push_back(static_cast<mx::ShapeElem>(dim));
    elem_count *= static_cast<size_t>(dim);
  }
  const size_t elem_bytes = (desc.dtype == mlx_ir::TensorDesc::INT32) ? sizeof(int32_t) : sizeof(float);
  const size_t expected_bytes = elem_count * elem_bytes;
  if (desc.size_bytes < expected_bytes) {
    throw std::runtime_error("tensor data size smaller than shape");
  }
  if (desc.data == nullptr && desc.size_bytes > 0) {
    throw std::runtime_error("tensor data is null");
  }

  if (desc.dtype == mlx_ir::TensorDesc::INT32) {
    if ((desc.size_bytes % sizeof(int32_t)) != 0) {
      throw std::runtime_error("int32 tensor data is not aligned to element size");
    }
    const auto* ptr = static_cast<const int32_t*>(desc.data);
    return mx::array(ptr, shape, mx::int32);
  }
  if (desc.dtype == mlx_ir::TensorDesc::FLOAT32) {
    if ((desc.size_bytes % sizeof(float)) != 0) {
      throw std::runtime_error("float32 tensor data is not aligned to element size");
    }
    const auto* ptr = static_cast<const float*>(desc.data);
    return mx::array(ptr, shape, mx::float32);
  }
  throw std::runtime_error("unsupported tensor dtype");
}

std::vector<int> to_shape_vec(const mx::array& arr) {
  std::vector<int> out;
  out.reserve(static_cast<size_t>(arr.ndim()));
  for (int i = 0; i < arr.ndim(); ++i) {
    out.push_back(static_cast<int>(arr.shape(i)));
  }
  return out;
}

mx::array resolve_output(
    const mlx_ir::IRProgram& program,
    const std::unordered_map<std::string, mx::array>& env,
    const std::string* output_name) {
  if (output_name != nullptr && !output_name->empty()) {
    auto out_it = env.find(*output_name);
    if (out_it == env.end()) {
      throw std::runtime_error("IR program did not produce output: " + *output_name);
    }
    return out_it->second;
  }
  auto loss_it = env.find("loss");
  if (loss_it != env.end()) {
    return loss_it->second;
  }
  if (!program.ops.empty() && program.ops.back().n_outputs > 0) {
    auto tail_it = env.find(program.ops.back().outputs[0]);
    if (tail_it != env.end()) {
      return tail_it->second;
    }
  }
  throw std::runtime_error("IR program did not produce `loss`");
}

std::unordered_map<std::string, mx::array> resolve_outputs(
    const std::unordered_map<std::string, mx::array>& env,
    const std::vector<std::string>& output_names) {
  std::unordered_map<std::string, mx::array> out;
  out.reserve(output_names.size());
  for (const auto& output_name : output_names) {
    auto it = env.find(output_name);
    if (it == env.end()) {
      throw std::runtime_error("IR program did not produce output: " + output_name);
    }
    out.emplace(output_name, it->second);
  }
  return out;
}

std::string inferred_output_name(const mlx_ir::IRProgram& program, const std::string& output_name) {
  if (!output_name.empty()) {
    return output_name;
  }
  for (const auto& op : program.ops) {
    for (int i = 0; i < op.n_outputs; ++i) {
      if (op.outputs[i] == "loss") {
        return "loss";
      }
    }
  }
  if (!program.ops.empty() && program.ops.back().n_outputs > 0) {
    return program.ops.back().outputs[0];
  }
  return "";
}

} // namespace

namespace mlx_ir {

mx::array ir_interpret(
    const IRProgram& program,
    const std::vector<mx::array>& weights,
    const TensorMap& inputs) {
  return ir_interpret(program, weights, inputs, "");
}

mx::array ir_interpret(
    const IRProgram& program,
    const std::vector<mx::array>& weights,
    const TensorMap& inputs,
    const std::string& output_name) {
  return ir_interpret(program, weights, inputs, output_name, false);
}

mx::array ir_interpret(
    const IRProgram& program,
    const std::vector<mx::array>& weights,
    const TensorMap& inputs,
    const std::string& output_name,
    bool training) {
  const std::vector<std::string> keep_outputs = output_name.empty()
      ? std::vector<std::string>{}
      : std::vector<std::string>{output_name};
  auto outputs = ir_interpret_outputs(program, weights, inputs, keep_outputs, training);
  if (output_name.empty()) {
    auto loss_it = outputs.find("loss");
    if (loss_it != outputs.end()) {
      return loss_it->second;
    }
    if (!program.ops.empty() && program.ops.back().n_outputs > 0) {
      auto tail_it = outputs.find(program.ops.back().outputs[0]);
      if (tail_it != outputs.end()) {
        return tail_it->second;
      }
    }
    throw std::runtime_error("IR program did not produce `loss`");
  }
  return outputs.at(output_name);
}

std::unordered_map<std::string, mx::array> ir_interpret_outputs(
    const IRProgram& program,
    const std::vector<mx::array>& weights,
    const TensorMap& inputs,
    const std::vector<std::string>& output_names) {
  return ir_interpret_outputs(program, weights, inputs, output_names, false);
}

std::unordered_map<std::string, mx::array> ir_interpret_outputs(
    const IRProgram& program,
    const std::vector<mx::array>& weights,
    const TensorMap& inputs,
    const std::vector<std::string>& output_names,
    bool training) {
  std::unordered_map<std::string, mx::array> env;
  env.reserve(program.ops.size() * 2 + static_cast<size_t>(weights.size()) + 8);
  std::unordered_set<std::string> pinned;
  pinned.reserve(static_cast<size_t>(inputs.size()) + weights.size() + 1);
  for (const auto& kv : inputs) {
    env.emplace(kv.first, tensor_desc_to_array(kv.second));
    pinned.emplace(kv.first);
  }

  for (size_t i = 0; i < weights.size(); ++i) {
    auto name = "w" + std::to_string(i);
    env.emplace(name, weights[i]);
    pinned.emplace(std::move(name));
  }

  if (output_names.empty()) {
    const auto keep_output = inferred_output_name(program, "");
    if (!keep_output.empty()) {
      pinned.emplace(keep_output);
    }
  } else {
    for (const auto& output_name : output_names) {
      if (!output_name.empty()) {
        pinned.emplace(output_name);
      }
    }
  }

  std::unordered_map<std::string, int> remaining_uses;
  remaining_uses.reserve(program.ops.size() * 2 + 8);
  for (const auto& op : program.ops) {
    for (int i = 0; i < op.n_inputs; ++i) {
      if (!op.inputs[i].empty()) {
        remaining_uses[op.inputs[i]]++;
      }
    }
  }

  auto get = [&](const IRop& op, int idx) -> const mx::array& {
    if (idx < 0 || idx >= op.n_inputs) {
      throw std::runtime_error("IR input index out of range");
    }
    auto it = env.find(op.inputs[idx]);
    if (it == env.end()) {
      throw std::runtime_error("IR input not found: " + op.inputs[idx]);
    }
    return it->second;
  };

  auto op_overwrites_name = [&](const IRop& op, const std::string& name) {
    for (int i = 0; i < op.n_outputs; ++i) {
      if (op.outputs[i] == name) {
        return true;
      }
    }
    return false;
  };

  auto set_out = [&](const IRop& op, int idx, mx::array arr) {
    if (idx < 0 || idx >= op.n_outputs) {
      throw std::runtime_error("IR output index out of range");
    }
    auto it = env.find(op.outputs[idx]);
    if (it == env.end()) {
      env.emplace(op.outputs[idx], std::move(arr));
    } else {
      it->second = std::move(arr);
    }
  };

  for (size_t op_idx = 0; op_idx < program.ops.size(); ++op_idx) {
    const auto& op = program.ops[op_idx];
    try {
      switch (op.type) {
      case OP_EMBED: {
        set_out(op, 0, mx::take(get(op, 0), mx::astype(get(op, 1), mx::int32), 0));
        break;
      }
      case OP_MATMUL: {
        set_out(op, 0, mx::matmul(get(op, 0), get(op, 1)));
        break;
      }
      case OP_ADD: {
        set_out(op, 0, get(op, 0) + get(op, 1));
        break;
      }
      case OP_MUL: {
        set_out(op, 0, get(op, 0) * get(op, 1));
        break;
      }
      case OP_SCALAR_MUL: {
        set_out(op, 0, get(op, 0) * op.float_params[0]);
        break;
      }
      case OP_SIGMOID: {
        set_out(op, 0, mx::sigmoid(get(op, 0)));
        break;
      }
      case OP_SILU: {
        auto x = get(op, 0);
        set_out(op, 0, x * mx::sigmoid(x));
        break;
      }
      case OP_GELU: {
        auto x = get(op, 0);
        set_out(op, 0, 0.5f * x * (1.0f + mx::tanh(0.7978845608f * (x + 0.044715f * x * mx::square(x)))));
        break;
      }
      case OP_RELU: {
        set_out(op, 0, mx::maximum(get(op, 0), mx::array(0.0f)));
        break;
      }
      case OP_TANH: {
        set_out(op, 0, mx::tanh(get(op, 0)));
        break;
      }
      case OP_SOFTMAX: {
        int axis = (op.n_int_params > 0) ? op.int_params[0] : -1;
        set_out(op, 0, mx::softmax(get(op, 0), axis));
        break;
      }
      case OP_RESHAPE: {
        if (op.n_int_params <= 0) {
          throw std::runtime_error("OP_RESHAPE missing shape params");
        }
        set_out(op, 0, mx::reshape(get(op, 0), make_shape(op.int_params, op.n_int_params)));
        break;
      }
      case OP_TRANSPOSE: {
        if (op.n_int_params <= 0) {
          throw std::runtime_error("OP_TRANSPOSE missing axes params");
        }
        std::vector<int> axes;
        axes.reserve(static_cast<size_t>(op.n_int_params));
        for (int i = 0; i < op.n_int_params; ++i) {
          axes.push_back(op.int_params[i]);
        }
        set_out(op, 0, mx::transpose(get(op, 0), axes));
        break;
      }
      case OP_SLICE: {
        // int_params: start, end, stride, axis
        if (op.n_int_params < 4) {
          throw std::runtime_error("OP_SLICE requires 4 int params");
        }
        const auto& x = get(op, 0);
        int axis = op.int_params[3];
        if (axis < 0 || axis >= x.ndim()) {
          throw std::runtime_error("OP_SLICE axis out of range");
        }
        mx::Shape starts;
        mx::Shape ends;
        mx::Shape strides;
        starts.reserve(static_cast<size_t>(x.ndim()));
        ends.reserve(static_cast<size_t>(x.ndim()));
        strides.reserve(static_cast<size_t>(x.ndim()));
        for (int d = 0; d < x.ndim(); ++d) {
          starts.push_back(0);
          ends.push_back(x.shape(d));
          strides.push_back(1);
        }
        starts[axis] = op.int_params[0];
        ends[axis] = op.int_params[1];
        strides[axis] = op.int_params[2];
        set_out(op, 0, mx::slice(x, starts, ends, strides));
        break;
      }
      case OP_CONCAT: {
        int axis = (op.n_int_params > 0) ? op.int_params[0] : 0;
        set_out(op, 0, mx::concatenate({get(op, 0), get(op, 1)}, axis));
        break;
      }
      case OP_CAUSAL_MASK: {
        if (op.n_int_params < 1) {
          throw std::runtime_error("OP_CAUSAL_MASK missing T");
        }
        int T = op.int_params[0];
        auto scores = get(op, 0);
        auto mask2d = mx::triu(mx::ones({T, T}, mx::bool_), 1);
        auto mask = mx::expand_dims(mx::expand_dims(mask2d, 0), 0);
        auto masked = mx::where(mask, mx::full_like(scores, -1e9f), scores);
        set_out(op, 0, masked);
        break;
      }
      case OP_PREFIX_CAUSAL_MASK: {
        if (op.n_int_params < 2) {
          throw std::runtime_error("OP_PREFIX_CAUSAL_MASK requires int params: selfT, prefixT");
        }
        int selfT = op.int_params[0];
        int prefixT = op.int_params[1];
        auto scores = get(op, 0);
        mx::array self_pos = mx::astype(mx::arange(selfT), mx::int32);
        mx::array cross_pos = mx::astype(mx::arange(prefixT), mx::int32);
        bool use_position_index = false;
        if (op.n_inputs >= 3) {
          self_pos = mx::astype(get(op, 1), mx::int32);
          cross_pos = mx::astype(get(op, 2), mx::int32);
          if (self_pos.ndim() != 1 || self_pos.shape(0) != selfT) {
            throw std::runtime_error("OP_PREFIX_CAUSAL_MASK self position index must be rank-1 int32 with length selfT");
          }
          if (cross_pos.ndim() != 1 || cross_pos.shape(0) != prefixT) {
            throw std::runtime_error("OP_PREFIX_CAUSAL_MASK cross position index must be rank-1 int32 with length prefixT");
          }
          use_position_index = true;
        } else if (op.n_int_params >= 6) {
          int selfStart = op.int_params[2];
          int selfStride = op.int_params[3];
          int crossStart = op.int_params[4];
          int crossStride = op.int_params[5];
          auto self_idx = mx::astype(mx::arange(selfT), mx::int32);
          auto cross_idx = mx::astype(mx::arange(prefixT), mx::int32);
          self_pos = self_idx * selfStride + selfStart;
          cross_pos = cross_idx * crossStride + crossStart;
          use_position_index = true;
        }
        mx::array causal_self = mx::triu(mx::ones({selfT, selfT}, mx::bool_), 1);
        mx::array cross_mask = mx::zeros({selfT, prefixT}, mx::bool_);
        if (use_position_index) {
          // Position-indexed causal rule: allow attention iff key_pos <= self_pos.
          // Future keys are masked where key_pos > self_pos.
          causal_self = mx::expand_dims(self_pos, 0) > mx::expand_dims(self_pos, 1);
          cross_mask = mx::expand_dims(cross_pos, 0) > mx::expand_dims(self_pos, 1);
        }
        auto mask2d = mx::concatenate({cross_mask, causal_self}, 1);
        auto mask = mx::expand_dims(mx::expand_dims(mask2d, 0), 0);
        auto masked = mx::where(mask, mx::full_like(scores, -1e9f), scores);
        set_out(op, 0, masked);
        break;
      }
      case OP_CROSS_ENTROPY: {
        set_out(op, 0, cross_entropy_mean(get(op, 0), mx::astype(get(op, 1), mx::int32)));
        break;
      }
      case OP_DROPOUT: {
        if (op.n_float_params < 1) {
          throw std::runtime_error("OP_DROPOUT requires rate float param");
        }
        auto x = get(op, 0);
        float rate = op.float_params[0];
        if (rate < 0.0f || rate > 1.0f) {
          throw std::runtime_error("OP_DROPOUT rate must be in [0,1]");
        }
        if (!training || rate == 0.0f) {
          set_out(op, 0, x);
          break;
        }
        if (rate == 1.0f) {
          set_out(op, 0, mx::zeros_like(x));
          break;
        }
        float keep_prob = 1.0f - rate;
        auto mask = mx::astype(mx::random::bernoulli(keep_prob, x.shape()), x.dtype());
        set_out(op, 0, x * mask / keep_prob);
        break;
      }
      case OP_SQUARE: {
        set_out(op, 0, mx::square(get(op, 0)));
        break;
      }
      case OP_SUB: {
        set_out(op, 0, get(op, 0) - get(op, 1));
        break;
      }
      case OP_DIV: {
        set_out(op, 0, get(op, 0) / get(op, 1));
        break;
      }
      case OP_CUMSUM: {
        int axis = (op.n_int_params > 0) ? op.int_params[0] : 0;
        set_out(op, 0, mx::cumsum(get(op, 0), axis));
        break;
      }
      case OP_ARGSORT: {
        int axis = (op.n_int_params > 0) ? op.int_params[0] : -1;
        set_out(op, 0, mx::argsort(get(op, 0), axis));
        break;
      }
      case OP_WHERE: {
        set_out(op, 0, mx::where(get(op, 0), get(op, 1), get(op, 2)));
        break;
      }
      case OP_LESS_THAN: {
        if (op.n_float_params < 1) {
          throw std::runtime_error("OP_LESS_THAN requires scalar");
        }
        set_out(op, 0, get(op, 0) < mx::array(op.float_params[0], mx::float32));
        break;
      }
      case OP_GREATER_EQ: {
        set_out(op, 0, get(op, 0) >= get(op, 1));
        break;
      }
      case OP_ARANGE: {
        if (op.n_int_params < 2) {
          throw std::runtime_error("OP_ARANGE requires start,end");
        }
        set_out(op, 0, mx::arange(op.int_params[0], op.int_params[1]));
        break;
      }
      case OP_MEAN_AXIS: {
        if (op.n_int_params < 1) {
          throw std::runtime_error("OP_MEAN_AXIS requires axis");
        }
        set_out(op, 0, mx::mean(get(op, 0), op.int_params[0]));
        break;
      }
      case OP_FULL: {
        if (op.n_int_params <= 0 || op.n_float_params < 1) {
          throw std::runtime_error("OP_FULL requires shape and value");
        }
        set_out(op, 0, mx::full(make_shape(op.int_params, op.n_int_params), op.float_params[0], mx::float32));
        break;
      }
      case OP_ASTYPE: {
        set_out(op, 0, mx::astype(get(op, 0), mx::float32));
        break;
      }
      case OP_RUNNING_VAR: {
        if (op.n_int_params < 3) {
          throw std::runtime_error("OP_RUNNING_VAR requires B,T,D");
        }
        float alpha = (op.n_float_params > 0) ? op.float_params[0] : 0.1f;
        set_out(op, 0, running_variance_raw(get(op, 0), op.int_params[0], op.int_params[1], op.int_params[2], alpha));
        break;
      }
      case OP_SCAN: {
        if (op.n_int_params < 3) {
          throw std::runtime_error("OP_SCAN requires B,T,D");
        }
        int B = op.int_params[0];
        int T = op.int_params[1];
        int D = op.int_params[2];
        auto x = mx::reshape(get(op, 0), {B, T, D});
        auto decay_raw = get(op, 1);
        auto decay = mx::reshape(decay_raw, {static_cast<mx::ShapeElem>(decay_raw.size())});
        auto gate = mx::sigmoid(decay);
        auto keep = 1.0f - gate;
        auto h = mx::zeros({B, D}, mx::float32);
        auto out = mx::zeros({B, T, D}, mx::float32);
        for (int t = 0; t < T; ++t) {
          auto xt = mx::reshape(mx::slice(x, {0, t, 0}, {B, t + 1, D}), {B, D});
          h = gate * h + keep * xt;
          out = mx::slice_update(out, mx::reshape(h, {B, 1, D}), mx::Shape{0, t, 0}, mx::Shape{B, t + 1, D});
        }
        set_out(op, 0, mx::reshape(out, {B * T, D}));
        break;
      }
      case OP_GRADIENT_MAGNITUDES: {
        auto hidden = mx::stop_gradient(get(op, 0));
        if (hidden.ndim() != 3) {
          throw std::runtime_error("OP_GRADIENT_MAGNITUDES expects input rank-3 [B,T,D]");
        }

        int T = hidden.shape(1);
        auto magnitudes = mx::zeros({T}, mx::float32);
        if (T > 1) {
          auto next = mx::slice(hidden, {0, 1, 0}, {hidden.shape(0), hidden.shape(1), hidden.shape(2)});
          auto prev = mx::slice(hidden, {0, 0, 0}, {hidden.shape(0), hidden.shape(1) - 1, hidden.shape(2)});
          auto diff = mx::subtract(next, prev);
          auto step_magnitudes = mx::mean(mx::sum(mx::multiply(diff, diff), -1), 0);
          auto prefix = mx::zeros({1}, mx::float32);
          magnitudes = mx::concatenate({prefix, step_magnitudes}, 0);
        }
        set_out(op, 0, mx::stop_gradient(magnitudes));
        break;
      }
      case OP_RMSNORM: {
        if (op.n_float_params < 1) {
          throw std::runtime_error("OP_RMSNORM requires eps float param");
        }
        auto x = get(op, 0);
        auto scale = get(op, 1);
        auto ms = mx::mean(mx::square(x), -1, true);
        auto rms_inv = 1.0f / mx::sqrt(ms + op.float_params[0]);
        set_out(op, 0, x * rms_inv * scale);
        break;
      }
      case OP_ROPE: {
        if (op.n_int_params < 2) {
          throw std::runtime_error("OP_ROPE requires int params: T, head_dim");
        }
        int T = op.int_params[0];
        int HD = op.int_params[1];
        if (HD <= 0 || (HD % 2) != 0) {
          throw std::runtime_error("OP_ROPE requires even positive head_dim");
        }
        int start = (op.n_int_params > 2) ? op.int_params[2] : 0;
        int stride = (op.n_int_params > 3) ? op.int_params[3] : 1;
        float base = (op.n_float_params > 0) ? op.float_params[0] : 10000.0f;

        auto dim_idx = mx::astype(mx::arange(0, HD / 2), mx::float32);
        auto freqs = mx::exp(dim_idx * static_cast<float>(-std::log(base) * 2.0 / static_cast<double>(HD)));
        auto positions = [&]() -> mx::array {
          if (op.n_inputs > 2 && !op.inputs[2].empty()) {
            auto positions_in = mx::astype(get(op, 2), mx::int32);
            if (positions_in.ndim() != 1 || positions_in.shape(0) != T) {
              throw std::runtime_error("OP_ROPE positions must be rank-1 int32 with length T");
            }
            return mx::astype(positions_in, mx::float32);
          }
          return mx::astype(mx::arange(0, T) * stride + start, mx::float32);
        }();
        auto angles = mx::reshape(positions, {T, 1}) * mx::reshape(freqs, {1, HD / 2});
        auto cos_t = mx::reshape(mx::cos(angles), {1, 1, T, HD / 2});
        auto sin_t = mx::reshape(mx::sin(angles), {1, 1, T, HD / 2});

        auto apply_rope = [&](const mx::array& x) -> mx::array {
          if (x.ndim() != 4) {
            throw std::runtime_error("OP_ROPE expects rank-4 tensors");
          }
          auto even = mx::slice(x, {0, 0, 0, 0}, {x.shape(0), x.shape(1), x.shape(2), x.shape(3)}, {1, 1, 1, 2});
          auto odd = mx::slice(x, {0, 0, 0, 1}, {x.shape(0), x.shape(1), x.shape(2), x.shape(3)}, {1, 1, 1, 2});
          auto rot_even = even * cos_t - odd * sin_t;
          auto rot_odd = even * sin_t + odd * cos_t;
          return mx::reshape(mx::stack({rot_even, rot_odd}, 4), x.shape());
        };

        set_out(op, 0, apply_rope(get(op, 0)));
        set_out(op, 1, apply_rope(get(op, 1)));
        break;
      }
      case OP_ROPE_STRIDED: {
        if (op.n_int_params < 2) {
          throw std::runtime_error("OP_ROPE_STRIDED requires int params: T, head_dim");
        }
        int T = op.int_params[0];
        int HD = op.int_params[1];
        if (HD <= 0 || (HD % 2) != 0) {
          throw std::runtime_error("OP_ROPE_STRIDED requires even positive head_dim");
        }
        int start = (op.n_int_params > 2) ? op.int_params[2] : 0;
        int stride = (op.n_int_params > 3) ? op.int_params[3] : 1;
        float base = (op.n_float_params > 0) ? op.float_params[0] : 10000.0f;

        auto dim_idx = mx::astype(mx::arange(0, HD / 2), mx::float32);
        auto freqs = mx::exp(dim_idx * static_cast<float>(-std::log(base) * 2.0 / static_cast<double>(HD)));
        auto positions = mx::astype(mx::arange(0, T) * stride + start, mx::float32);
        auto angles = mx::reshape(positions, {T, 1}) * mx::reshape(freqs, {1, HD / 2});
        auto cos_t = mx::reshape(mx::cos(angles), {1, 1, T, HD / 2});
        auto sin_t = mx::reshape(mx::sin(angles), {1, 1, T, HD / 2});

        auto apply_rope = [&](const mx::array& x) -> mx::array {
          if (x.ndim() != 4) {
            throw std::runtime_error("OP_ROPE_STRIDED expects rank-4 tensors");
          }
          auto even = mx::slice(x, {0, 0, 0, 0}, {x.shape(0), x.shape(1), x.shape(2), x.shape(3)}, {1, 1, 1, 2});
          auto odd = mx::slice(x, {0, 0, 0, 1}, {x.shape(0), x.shape(1), x.shape(2), x.shape(3)}, {1, 1, 1, 2});
          auto rot_even = even * cos_t - odd * sin_t;
          auto rot_odd = even * sin_t + odd * cos_t;
          return mx::reshape(mx::stack({rot_even, rot_odd}, 4), x.shape());
        };

        set_out(op, 0, apply_rope(get(op, 0)));
        set_out(op, 1, apply_rope(get(op, 1)));
        break;
      }
      case OP_SQRT: {
        set_out(op, 0, mx::sqrt(get(op, 0)));
        break;
      }
      case OP_RSQRT: {
        set_out(op, 0, 1.0f / mx::sqrt(get(op, 0)));
        break;
      }
      case OP_SIN: {
        set_out(op, 0, mx::sin(get(op, 0)));
        break;
      }
      case OP_COS: {
        set_out(op, 0, mx::cos(get(op, 0)));
        break;
      }
      case OP_EXP: {
        set_out(op, 0, mx::exp(get(op, 0)));
        break;
      }
      case OP_OUTER: {
        auto a = get(op, 0);
        auto b = get(op, 1);
        set_out(op, 0, mx::reshape(a, {static_cast<mx::ShapeElem>(a.size()), 1}) *
                           mx::reshape(b, {1, static_cast<mx::ShapeElem>(b.size())}));
        break;
      }
      case OP_SQUEEZE: {
        if (op.n_int_params < 1) {
          throw std::runtime_error("OP_SQUEEZE requires axis int param");
        }
        auto x = get(op, 0);
        int axis = op.int_params[0];
        if (axis < 0) {
          axis += x.ndim();
        }
        if (axis < 0 || axis >= x.ndim()) {
          throw std::runtime_error("OP_SQUEEZE axis out of range");
        }
        if (x.shape(axis) != 1) {
          throw std::runtime_error("OP_SQUEEZE axis dimension must be 1");
        }
        mx::Shape shape;
        shape.reserve(static_cast<size_t>(x.ndim() - 1));
        for (int i = 0; i < x.ndim(); ++i) {
          if (i != axis) {
            shape.push_back(x.shape(i));
          }
        }
        set_out(op, 0, mx::reshape(x, shape));
        break;
      }
      default:
        throw std::runtime_error("unsupported IR opcode: " + std::to_string(op.type));
      }

      for (int i = 0; i < op.n_inputs; ++i) {
        const auto& name = op.inputs[i];
        if (name.empty() || pinned.find(name) != pinned.end() || op_overwrites_name(op, name)) {
          continue;
        }
        auto it = remaining_uses.find(name);
        if (it == remaining_uses.end()) {
          continue;
        }
        it->second--;
        if (it->second <= 0) {
          env.erase(name);
          remaining_uses.erase(it);
        }
      }

      for (int i = 0; i < op.n_outputs; ++i) {
        const auto& name = op.outputs[i];
        if (name.empty() || pinned.find(name) != pinned.end()) {
          continue;
        }
        auto it = remaining_uses.find(name);
        if (it == remaining_uses.end() || it->second <= 0) {
          env.erase(name);
        }
      }
    } catch (const std::exception& e) {
      throw std::runtime_error(
          "IR op #" + std::to_string(op_idx) + " type=" + std::to_string(op.type) +
          " failed: " + e.what());
    }
  }
  if (output_names.empty()) {
    return {{"loss", resolve_output(program, env, nullptr)}};
  }
  return resolve_outputs(env, output_names);
}

mx::array ir_interpret(
    const IRProgram& program,
    const std::vector<mx::array>& weights,
    const mx::array& tokens,
    const mx::array& targets) {
  return ir_interpret(program, weights, tokens, targets, false);
}

mx::array ir_interpret(
    const IRProgram& program,
    const std::vector<mx::array>& weights,
    const mx::array& tokens,
    const mx::array& targets,
    bool training) {
  auto tokens_i32 = mx::astype(tokens, mx::int32);
  auto targets_i32 = mx::astype(targets, mx::int32);
  mx::eval(tokens_i32, targets_i32);

  std::vector<int32_t> tokens_host(tokens_i32.size());
  std::vector<int32_t> targets_host(targets_i32.size());
  std::memcpy(tokens_host.data(), tokens_i32.data<int32_t>(), tokens_host.size() * sizeof(int32_t));
  std::memcpy(targets_host.data(), targets_i32.data<int32_t>(), targets_host.size() * sizeof(int32_t));

  TensorMap inputs;
  inputs.reserve(2);
  inputs.emplace("tokens", TensorDesc{
                             TensorDesc::INT32,
                             to_shape_vec(tokens_i32),
                             tokens_host.data(),
                             tokens_host.size() * sizeof(int32_t),
                         });
  inputs.emplace("targets", TensorDesc{
                              TensorDesc::INT32,
                              to_shape_vec(targets_i32),
                              targets_host.data(),
                              targets_host.size() * sizeof(int32_t),
                          });
  return ir_interpret(program, weights, inputs, "", training);
}

} // namespace mlx_ir
