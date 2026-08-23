#include "gated_delta_metal_primitive.h"

#include <mlx/allocator.h>
#include <mlx/ops.h>
#include <mlx/primitives.h>
#include <mlx/version.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef __APPLE__
#include <mlx/backend/metal/device.h>
#include <mlx/backend/metal/metal.h>
#endif

namespace mx = mlx::core;

namespace mlx_ir {
namespace {

constexpr int kGatedDeltaMetalThreads = 32;
constexpr int kGatedDeltaMetalMaxDK = 64;
constexpr int kGatedDeltaMetalMaxDV = 256;
constexpr int kGatedDeltaMetalMaxBackwardWindow = 8;
constexpr size_t kGatedDeltaMetalThreadgroupBudget = 24 * 1024;

bool gated_delta_env_is_one(const char* name) {
  const char* raw = std::getenv(name);
  return raw != nullptr && std::string(raw) == "1";
}

int gated_delta_metal_backward_window(int T, int Dk, int Dv, int chunk_size) {
  const size_t bytes_per_step =
      static_cast<size_t>(Dk) *
      static_cast<size_t>(std::min(Dv, kGatedDeltaMetalThreads)) *
      sizeof(float);
  const int memory_limited = bytes_per_step == 0
      ? 1
      : static_cast<int>(kGatedDeltaMetalThreadgroupBudget / bytes_per_step) - 1;
  return std::max(
      1,
      std::min({
          T,
          chunk_size,
          kGatedDeltaMetalMaxBackwardWindow,
          memory_limited}));
}

void validate_gated_delta_scan_shape(
    int B,
    int T,
    int H,
    int Dk,
    int Dv,
    int chunk_size) {
  if (B <= 0 || T <= 0 || H <= 0 || Dk <= 0 || Dv <= 0 || chunk_size <= 0) {
    throw std::runtime_error(
        "GatedDeltaNet Metal scan requires positive B,T,H,Dk,Dv,chunk_size");
  }
  if (Dk > kGatedDeltaMetalMaxDK || Dv > kGatedDeltaMetalMaxDV) {
    throw std::runtime_error(
        "GatedDeltaNet Metal scan supports d_k <= 64 and d_v <= 256");
  }
}

std::vector<mx::array> contiguous_gated_delta_inputs(
    const mx::array& q,
    const mx::array& k,
    const mx::array& v,
    const mx::array& beta,
    const mx::array& gate) {
  return {
      mx::contiguous(mx::astype(q, mx::float32)),
      mx::contiguous(mx::astype(k, mx::float32)),
      mx::contiguous(mx::astype(v, mx::float32)),
      mx::contiguous(mx::astype(beta, mx::float32)),
      mx::contiguous(mx::astype(gate, mx::float32))};
}

void require_gated_delta_float32(
    const std::vector<mx::array>& arrays,
    const char* name) {
  for (const auto& array : arrays) {
    if (array.dtype() != mx::float32) {
      throw std::runtime_error(std::string(name) + " requires float32 arrays");
    }
  }
}

#ifdef __APPLE__

const char* kGatedDeltaMetalSolveSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

kernel void gated_delta_chunk_solve_metal(
    const device float* raw_attn [[buffer(0)]],
    device float* solve_attn [[buffer(1)]],
    constant int& chunk_size [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint col = gid.x;
  const uint matrix = gid.y;
  if (col >= static_cast<uint>(chunk_size)) {
    return;
  }

  const uint n = static_cast<uint>(chunk_size);
  const uint base = matrix * n * n;

  for (uint row = 0; row < col; ++row) {
    solve_attn[base + row * n + col] = 0.0f;
  }
  solve_attn[base + col * n + col] = 1.0f;

  for (uint row = col + 1; row < n; ++row) {
    float acc = raw_attn[base + row * n + col];
    for (uint j = col + 1; j < row; ++j) {
      acc += raw_attn[base + row * n + j] * solve_attn[base + j * n + col];
    }
    solve_attn[base + row * n + col] = acc;
  }
}
)METAL";

const char* kGatedDeltaScanMetalSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

#define GDN_MAX_DK 64
#define GDN_TILE_DV 32

inline void gdn_atomic_add(device float* address, float value) {
  device atomic_uint* atomic_address =
      reinterpret_cast<device atomic_uint*>(address);
  uint expected = atomic_load_explicit(atomic_address, memory_order_relaxed);
  while (true) {
    const uint desired = as_type<uint>(as_type<float>(expected) + value);
    if (atomic_compare_exchange_weak_explicit(
            atomic_address,
            &expected,
            desired,
            memory_order_relaxed,
            memory_order_relaxed)) {
      return;
    }
  }
}

inline int gdn_qk_idx(int b, int t, int h, int d, int T, int H, int Dk) {
  return ((b * T + t) * H + h) * Dk + d;
}

inline int gdn_v_idx(int b, int t, int h, int u, int T, int H, int Dv) {
  return ((b * T + t) * H + h) * Dv + u;
}

inline int gdn_head_idx(int b, int t, int h, int T, int H) {
  return (b * T + t) * H + h;
}

inline int gdn_checkpoint_idx(
    int b,
    int h,
    int window,
    int d,
    int u,
    int H,
    int windows,
    int Dk,
    int Dv) {
  return ((((b * H + h) * windows + window) * Dk + d) * Dv) + u;
}

inline int gdn_history_idx(int local, int d, int lane, int Dk) {
  return (local * Dk + d) * GDN_TILE_DV + lane;
}

kernel void gated_delta_scan_forward_metal(
    const device float* q [[buffer(0)]],
    const device float* k [[buffer(1)]],
    const device float* v [[buffer(2)]],
    const device float* beta [[buffer(3)]],
    const device float* gate [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant int& B [[buffer(6)]],
    constant int& T [[buffer(7)]],
    constant int& H [[buffer(8)]],
    constant int& Dk [[buffer(9)]],
    constant int& Dv [[buffer(10)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
  const int tile = static_cast<int>(group.x) / H;
  const int h = static_cast<int>(group.x) % H;
  const int b = static_cast<int>(group.y);
  const int u = tile * GDN_TILE_DV + static_cast<int>(tid);
  const bool active = b < B && h < H && u < Dv;
  float state[GDN_MAX_DK];
  if (active) {
    for (int d = 0; d < Dk; ++d) {
      state[d] = 0.0f;
    }
  }

  for (int t = 0; t < T; ++t) {
    if (!active) {
      continue;
    }
    const int head_idx = gdn_head_idx(b, t, h, T, H);
    const float decay = max(gate[head_idx], 1e-30f);
    float prediction = 0.0f;
    for (int d = 0; d < Dk; ++d) {
      state[d] *= decay;
      prediction += k[gdn_qk_idx(b, t, h, d, T, H, Dk)] * state[d];
    }
    const int vu = gdn_v_idx(b, t, h, u, T, H, Dv);
    const float error = beta[head_idx] * (v[vu] - prediction);
    float output = 0.0f;
    for (int d = 0; d < Dk; ++d) {
      const int qkd = gdn_qk_idx(b, t, h, d, T, H, Dk);
      state[d] += k[qkd] * error;
      output += q[qkd] * state[d];
    }
    out[vu] = output;
  }
}

kernel void gated_delta_scan_zero_grads_metal(
    device float* grad_q [[buffer(0)]],
    device float* grad_k [[buffer(1)]],
    device float* grad_v [[buffer(2)]],
    device float* grad_beta [[buffer(3)]],
    device float* grad_gate [[buffer(4)]],
    constant int& qk_size [[buffer(5)]],
    constant int& v_size [[buffer(6)]],
    constant int& head_size [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid < static_cast<uint>(qk_size)) {
    grad_q[gid] = 0.0f;
    grad_k[gid] = 0.0f;
  }
  if (gid < static_cast<uint>(v_size)) {
    grad_v[gid] = 0.0f;
  }
  if (gid < static_cast<uint>(head_size)) {
    grad_beta[gid] = 0.0f;
    grad_gate[gid] = 0.0f;
  }
}

kernel void gated_delta_scan_checkpoints_metal(
    const device float* k [[buffer(0)]],
    const device float* v [[buffer(1)]],
    const device float* beta [[buffer(2)]],
    const device float* gate [[buffer(3)]],
    device float* checkpoints [[buffer(4)]],
    constant int& B [[buffer(5)]],
    constant int& T [[buffer(6)]],
    constant int& H [[buffer(7)]],
    constant int& Dk [[buffer(8)]],
    constant int& Dv [[buffer(9)]],
    constant int& window_size [[buffer(10)]],
    constant int& windows [[buffer(11)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
  const int tile = static_cast<int>(group.x) / H;
  const int h = static_cast<int>(group.x) % H;
  const int b = static_cast<int>(group.y);
  const int u = tile * GDN_TILE_DV + static_cast<int>(tid);
  const bool active = b < B && h < H && u < Dv;
  float state[GDN_MAX_DK];
  if (active) {
    for (int d = 0; d < Dk; ++d) {
      state[d] = 0.0f;
    }
  }

  for (int t = 0; t < T; ++t) {
    if (!active) {
      continue;
    }
    if ((t % window_size) == 0) {
      const int window = t / window_size;
      for (int d = 0; d < Dk; ++d) {
        checkpoints[gdn_checkpoint_idx(
            b, h, window, d, u, H, windows, Dk, Dv)] = state[d];
      }
    }
    const int head_idx = gdn_head_idx(b, t, h, T, H);
    const float decay = max(gate[head_idx], 1e-30f);
    float prediction = 0.0f;
    for (int d = 0; d < Dk; ++d) {
      state[d] *= decay;
      prediction += k[gdn_qk_idx(b, t, h, d, T, H, Dk)] * state[d];
    }
    const int vu = gdn_v_idx(b, t, h, u, T, H, Dv);
    const float error = beta[head_idx] * (v[vu] - prediction);
    for (int d = 0; d < Dk; ++d) {
      state[d] += k[gdn_qk_idx(b, t, h, d, T, H, Dk)] * error;
    }
  }
}

kernel void gated_delta_scan_backward_metal(
    const device float* q [[buffer(0)]],
    const device float* k [[buffer(1)]],
    const device float* v [[buffer(2)]],
    const device float* beta [[buffer(3)]],
    const device float* gate [[buffer(4)]],
    const device float* grad_out [[buffer(5)]],
    const device float* checkpoints [[buffer(6)]],
    device float* grad_q [[buffer(7)]],
    device float* grad_k [[buffer(8)]],
    device float* grad_v [[buffer(9)]],
    device float* grad_beta [[buffer(10)]],
    device float* grad_gate [[buffer(11)]],
    constant int& B [[buffer(12)]],
    constant int& T [[buffer(13)]],
    constant int& H [[buffer(14)]],
    constant int& Dk [[buffer(15)]],
    constant int& Dv [[buffer(16)]],
    constant int& window_size [[buffer(17)]],
    constant int& windows [[buffer(18)]],
    threadgroup float* history [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
  const int tile = static_cast<int>(group.x) / H;
  const int h = static_cast<int>(group.x) % H;
  const int b = static_cast<int>(group.y);
  const int lane = static_cast<int>(tid);
  const int u = tile * GDN_TILE_DV + lane;
  const bool active = b < B && h < H && u < Dv;
  float dstate[GDN_MAX_DK];
  if (active) {
    for (int d = 0; d < Dk; ++d) {
      dstate[d] = 0.0f;
    }
  }

  for (int window = windows - 1; window >= 0; --window) {
    const int start = window * window_size;
    const int limit = min(start + window_size, T);
    if (active) {
      for (int d = 0; d < Dk; ++d) {
        history[gdn_history_idx(0, d, lane, Dk)] =
            checkpoints[gdn_checkpoint_idx(
                b, h, window, d, u, H, windows, Dk, Dv)];
      }
      for (int t = start; t < limit; ++t) {
        const int local = t - start;
        const int head_idx = gdn_head_idx(b, t, h, T, H);
        const float decay = max(gate[head_idx], 1e-30f);
        float prediction = 0.0f;
        for (int d = 0; d < Dk; ++d) {
          const float decayed =
              history[gdn_history_idx(local, d, lane, Dk)] * decay;
          history[gdn_history_idx(local + 1, d, lane, Dk)] = decayed;
          prediction += k[gdn_qk_idx(b, t, h, d, T, H, Dk)] * decayed;
        }
        const int vu = gdn_v_idx(b, t, h, u, T, H, Dv);
        const float error = beta[head_idx] * (v[vu] - prediction);
        for (int d = 0; d < Dk; ++d) {
          history[gdn_history_idx(local + 1, d, lane, Dk)] +=
              k[gdn_qk_idx(b, t, h, d, T, H, Dk)] * error;
        }
      }
    }

    for (int t = limit - 1; t >= start; --t) {
      const int local = t - start;
      const int head_idx = gdn_head_idx(b, t, h, T, H);
      const float decay = max(gate[head_idx], 1e-30f);
      const int vu = gdn_v_idx(b, t, h, u, T, H, Dv);
      const float dout = active ? grad_out[vu] : 0.0f;
      float prediction = 0.0f;
      if (active) {
        for (int d = 0; d < Dk; ++d) {
          prediction +=
              k[gdn_qk_idx(b, t, h, d, T, H, Dk)] *
              (history[gdn_history_idx(local, d, lane, Dk)] * decay);
        }
      }
      const float residual = active ? v[vu] - prediction : 0.0f;
      const float error = active ? beta[head_idx] * residual : 0.0f;
      float derror = 0.0f;
      if (active) {
        for (int d = 0; d < Dk; ++d) {
          const int qkd = gdn_qk_idx(b, t, h, d, T, H, Dk);
          const float state_after =
              history[gdn_history_idx(local + 1, d, lane, Dk)];
          const float upstream = dstate[d] + q[qkd] * dout;
          derror += upstream * k[qkd];
        }
      }

      float dgate_partial = 0.0f;
      for (int d = 0; d < Dk; ++d) {
        const int qkd = gdn_qk_idx(b, t, h, d, T, H, Dk);
        const float state_before = active
            ? history[gdn_history_idx(local, d, lane, Dk)]
            : 0.0f;
        const float state_after = active
            ? history[gdn_history_idx(local + 1, d, lane, Dk)]
            : 0.0f;
        const float decayed_state = state_before * decay;
        const float upstream = active ? dstate[d] + q[qkd] * dout : 0.0f;
        const float dq_partial = active ? dout * state_after : 0.0f;
        const float dprediction = active ? -derror * beta[head_idx] : 0.0f;
        const float dk_partial =
            active ? upstream * error + dprediction * decayed_state : 0.0f;
        const float dq_total = simd_sum(dq_partial);
        const float dk_total = simd_sum(dk_partial);
        if (tid == 0 && b < B && h < H) {
          gdn_atomic_add(&grad_q[qkd], dq_total);
          gdn_atomic_add(&grad_k[qkd], dk_total);
        }
        if (active) {
          const float ddecayed = upstream + k[qkd] * dprediction;
          dgate_partial += ddecayed * state_before;
          dstate[d] = ddecayed * decay;
        }
      }

      const float dbeta_partial = active ? derror * residual : 0.0f;
      if (active) {
        grad_v[vu] = derror * beta[head_idx];
      }
      const float dbeta_total = simd_sum(dbeta_partial);
      const float dgate_total = simd_sum(dgate_partial);
      if (tid == 0 && b < B && h < H) {
        gdn_atomic_add(&grad_beta[head_idx], dbeta_total);
        gdn_atomic_add(
            &grad_gate[head_idx],
            gate[head_idx] > 1e-30f ? dgate_total : 0.0f);
      }
    }
  }
}
)METAL";

class SolveStrictlyLowerMetalPrimitive : public mx::UnaryPrimitive {
 public:
  SolveStrictlyLowerMetalPrimitive(
      mx::Stream stream,
      int matrix_count,
      int chunk_size)
      : mx::UnaryPrimitive(stream),
        matrix_count_(matrix_count),
        chunk_size_(chunk_size) {}

  void eval_cpu(const std::vector<mx::array>&, mx::array&) override {
    throw std::runtime_error("SolveStrictlyLowerMetalPrimitive is Metal-only");
  }

  void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override {
    if (inputs.size() != 1) {
      throw std::runtime_error("SolveStrictlyLowerMetalPrimitive expects 1 input");
    }
    if (inputs[0].dtype() != mx::float32 || out.dtype() != mx::float32) {
      throw std::runtime_error("SolveStrictlyLowerMetalPrimitive requires float32 tensors");
    }
    if (!mx::metal::is_available()) {
      throw std::runtime_error("SolveStrictlyLowerMetalPrimitive requires Metal");
    }

    auto& device = mx::metal::device(stream().device);
#if MLX_VERSION_NUMERIC >= 31002
    auto& encoder = mx::metal::get_command_encoder(stream());
#else
    auto& encoder = device.get_command_encoder(stream().index);
#endif
    out.set_data(mx::allocator::malloc(out.nbytes()));

    auto* library = device.get_library(
        "mixlab_gated_delta_metal_solve",
        []() { return std::string(kGatedDeltaMetalSolveSource); });
    auto* kernel = device.get_kernel(
        "gated_delta_chunk_solve_metal",
        library);

    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(inputs[0], 0);
    encoder.set_output_array(out, 1);
    encoder.set_bytes(chunk_size_, 2);
    encoder.dispatch_threads(
        MTL::Size::Make(
            static_cast<NS::UInteger>(chunk_size_),
            static_cast<NS::UInteger>(matrix_count_),
            1),
        MTL::Size::Make(
            static_cast<NS::UInteger>(std::min(chunk_size_, 64)),
            1,
            1));
  }

  std::vector<mx::array> vjp(
      const std::vector<mx::array>&,
      const std::vector<mx::array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<mx::array>& outputs) override {
    if (cotangents.size() != 1 || outputs.size() != 1) {
      throw std::runtime_error("SolveStrictlyLowerMetalPrimitive vjp expects one cotangent and one output");
    }
    auto solve_t = mx::transpose(outputs[0], {0, 2, 1});
    auto grad = mx::matmul(mx::matmul(solve_t, cotangents[0]), solve_t);
    grad = mx::tril(grad, -1);

    std::vector<mx::array> grads;
    grads.reserve(argnums.size());
    for (int argnum : argnums) {
      if (argnum != 0) {
        throw std::runtime_error("SolveStrictlyLowerMetalPrimitive vjp argnum out of range");
      }
      grads.push_back(grad);
    }
    return grads;
  }

  const char* name() const override {
    return "SolveStrictlyLowerMetalPrimitive";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    auto* rhs = dynamic_cast<const SolveStrictlyLowerMetalPrimitive*>(&other);
    return rhs != nullptr &&
        matrix_count_ == rhs->matrix_count_ &&
        chunk_size_ == rhs->chunk_size_;
  }

  std::vector<mx::Shape> output_shapes(const std::vector<mx::array>&) override {
    return {mx::Shape{
        static_cast<mx::ShapeElem>(matrix_count_),
        static_cast<mx::ShapeElem>(chunk_size_),
        static_cast<mx::ShapeElem>(chunk_size_)}};
  }

 private:
  int matrix_count_;
  int chunk_size_;
};

class GatedDeltaScanMetalForwardPrimitive : public mx::Primitive {
 public:
  GatedDeltaScanMetalForwardPrimitive(
      mx::Stream stream,
      int B,
      int T,
      int H,
      int Dk,
      int Dv,
      int chunk_size)
      : mx::Primitive(stream),
        B_(B),
        T_(T),
        H_(H),
        Dk_(Dk),
        Dv_(Dv),
        chunk_size_(chunk_size) {}

  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("GatedDeltaScanMetalForwardPrimitive is Metal-only");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    if (inputs.size() != 5 || outputs.size() != 1) {
      throw std::runtime_error(
          "GatedDeltaScanMetalForwardPrimitive expects 5 inputs and 1 output");
    }
    require_gated_delta_float32(inputs, "GatedDeltaScanMetalForwardPrimitive");
    require_gated_delta_float32(outputs, "GatedDeltaScanMetalForwardPrimitive");
    validate_gated_delta_scan_shape(B_, T_, H_, Dk_, Dv_, chunk_size_);

    auto& device = mx::metal::device(stream().device);
#if MLX_VERSION_NUMERIC >= 31002
    auto& encoder = mx::metal::get_command_encoder(stream());
#else
    auto& encoder = device.get_command_encoder(stream().index);
#endif
    outputs[0].set_data(mx::allocator::malloc(outputs[0].nbytes()));
    auto* library = device.get_library(
        "mixlab_gated_delta_scan_metal",
        []() { return std::string(kGatedDeltaScanMetalSource); });
    auto* kernel = device.get_kernel("gated_delta_scan_forward_metal", library);
    encoder.set_compute_pipeline_state(kernel);
    for (int i = 0; i < 5; ++i) {
      encoder.set_input_array(inputs[static_cast<size_t>(i)], i);
    }
    encoder.set_output_array(outputs[0], 5);
    encoder.set_bytes(B_, 6);
    encoder.set_bytes(T_, 7);
    encoder.set_bytes(H_, 8);
    encoder.set_bytes(Dk_, 9);
    encoder.set_bytes(Dv_, 10);
    const int value_tiles =
        (Dv_ + kGatedDeltaMetalThreads - 1) / kGatedDeltaMetalThreads;
    encoder.dispatch_threadgroups(
        MTL::Size::Make(
            static_cast<NS::UInteger>(H_ * value_tiles),
            static_cast<NS::UInteger>(B_),
            1),
        MTL::Size::Make(kGatedDeltaMetalThreads, 1, 1));
  }

  const char* name() const override {
    return "GatedDeltaScanMetalForwardPrimitive";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    auto* rhs = dynamic_cast<const GatedDeltaScanMetalForwardPrimitive*>(&other);
    return rhs != nullptr &&
        B_ == rhs->B_ && T_ == rhs->T_ && H_ == rhs->H_ &&
        Dk_ == rhs->Dk_ && Dv_ == rhs->Dv_ &&
        chunk_size_ == rhs->chunk_size_;
  }

  std::vector<mx::Shape> output_shapes(const std::vector<mx::array>&) override {
    return {mx::Shape{
        static_cast<mx::ShapeElem>(B_ * T_ * H_),
        static_cast<mx::ShapeElem>(Dv_)}};
  }

 private:
  int B_;
  int T_;
  int H_;
  int Dk_;
  int Dv_;
  int chunk_size_;
};

class GatedDeltaScanMetalBackwardPrimitive : public mx::Primitive {
 public:
  GatedDeltaScanMetalBackwardPrimitive(
      mx::Stream stream,
      int B,
      int T,
      int H,
      int Dk,
      int Dv,
      int chunk_size)
      : mx::Primitive(stream),
        B_(B),
        T_(T),
        H_(H),
        Dk_(Dk),
        Dv_(Dv),
        chunk_size_(chunk_size),
        window_size_(gated_delta_metal_backward_window(T, Dk, Dv, chunk_size)),
        windows_((T + window_size_ - 1) / window_size_) {}

  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("GatedDeltaScanMetalBackwardPrimitive is Metal-only");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    if (inputs.size() != 6 || outputs.size() != 6) {
      throw std::runtime_error(
          "GatedDeltaScanMetalBackwardPrimitive expects 6 inputs and 6 outputs");
    }
    require_gated_delta_float32(inputs, "GatedDeltaScanMetalBackwardPrimitive");
    require_gated_delta_float32(outputs, "GatedDeltaScanMetalBackwardPrimitive");
    validate_gated_delta_scan_shape(B_, T_, H_, Dk_, Dv_, chunk_size_);

    auto& device = mx::metal::device(stream().device);
#if MLX_VERSION_NUMERIC >= 31002
    auto& encoder = mx::metal::get_command_encoder(stream());
#else
    auto& encoder = device.get_command_encoder(stream().index);
#endif
    for (auto& output : outputs) {
      output.set_data(mx::allocator::malloc(output.nbytes()));
    }
    auto* library = device.get_library(
        "mixlab_gated_delta_scan_metal",
        []() { return std::string(kGatedDeltaScanMetalSource); });

    auto* zero_kernel = device.get_kernel(
        "gated_delta_scan_zero_grads_metal", library);
    encoder.set_compute_pipeline_state(zero_kernel);
    for (int i = 1; i < 6; ++i) {
      encoder.set_output_array(outputs[static_cast<size_t>(i)], i - 1);
    }
    const int qk_size = B_ * T_ * H_ * Dk_;
    const int v_size = B_ * T_ * H_ * Dv_;
    const int head_size = B_ * T_ * H_;
    encoder.set_bytes(qk_size, 5);
    encoder.set_bytes(v_size, 6);
    encoder.set_bytes(head_size, 7);
    encoder.dispatch_threads(
        MTL::Size::Make(
            static_cast<NS::UInteger>(std::max({qk_size, v_size, head_size})),
            1,
            1),
        MTL::Size::Make(256, 1, 1));

    auto* checkpoint_kernel = device.get_kernel(
        "gated_delta_scan_checkpoints_metal", library);
    encoder.set_compute_pipeline_state(checkpoint_kernel);
    encoder.set_input_array(inputs[1], 0);
    encoder.set_input_array(inputs[2], 1);
    encoder.set_input_array(inputs[3], 2);
    encoder.set_input_array(inputs[4], 3);
    encoder.set_output_array(outputs[0], 4);
    encoder.set_bytes(B_, 5);
    encoder.set_bytes(T_, 6);
    encoder.set_bytes(H_, 7);
    encoder.set_bytes(Dk_, 8);
    encoder.set_bytes(Dv_, 9);
    encoder.set_bytes(window_size_, 10);
    encoder.set_bytes(windows_, 11);
    const int value_tiles =
        (Dv_ + kGatedDeltaMetalThreads - 1) / kGatedDeltaMetalThreads;
    encoder.dispatch_threadgroups(
        MTL::Size::Make(
            static_cast<NS::UInteger>(H_ * value_tiles),
            static_cast<NS::UInteger>(B_),
            1),
        MTL::Size::Make(kGatedDeltaMetalThreads, 1, 1));

    auto* backward_kernel = device.get_kernel(
        "gated_delta_scan_backward_metal", library);
    encoder.set_compute_pipeline_state(backward_kernel);
    for (int i = 0; i < 6; ++i) {
      encoder.set_input_array(inputs[static_cast<size_t>(i)], i);
    }
    encoder.set_input_array(outputs[0], 6);
    for (int i = 1; i < 6; ++i) {
      encoder.set_output_array(outputs[static_cast<size_t>(i)], i + 6);
    }
    encoder.set_bytes(B_, 12);
    encoder.set_bytes(T_, 13);
    encoder.set_bytes(H_, 14);
    encoder.set_bytes(Dk_, 15);
    encoder.set_bytes(Dv_, 16);
    encoder.set_bytes(window_size_, 17);
    encoder.set_bytes(windows_, 18);
    const size_t history_bytes =
        static_cast<size_t>(window_size_ + 1) *
        static_cast<size_t>(Dk_) *
        static_cast<size_t>(kGatedDeltaMetalThreads) *
        sizeof(float);
    encoder.set_threadgroup_memory_length(history_bytes, 0);
    encoder.dispatch_threadgroups(
        MTL::Size::Make(
            static_cast<NS::UInteger>(H_ * value_tiles),
            static_cast<NS::UInteger>(B_),
            1),
        MTL::Size::Make(kGatedDeltaMetalThreads, 1, 1));
  }

  const char* name() const override {
    return "GatedDeltaScanMetalBackwardPrimitive";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    auto* rhs = dynamic_cast<const GatedDeltaScanMetalBackwardPrimitive*>(&other);
    return rhs != nullptr &&
        B_ == rhs->B_ && T_ == rhs->T_ && H_ == rhs->H_ &&
        Dk_ == rhs->Dk_ && Dv_ == rhs->Dv_ &&
        chunk_size_ == rhs->chunk_size_ &&
        window_size_ == rhs->window_size_;
  }

  std::vector<mx::Shape> output_shapes(const std::vector<mx::array>&) override {
    return {
        mx::Shape{
            static_cast<mx::ShapeElem>(B_ * H_ * windows_ * Dk_),
            static_cast<mx::ShapeElem>(Dv_)},
        mx::Shape{
            static_cast<mx::ShapeElem>(B_),
            static_cast<mx::ShapeElem>(T_),
            static_cast<mx::ShapeElem>(H_),
            static_cast<mx::ShapeElem>(Dk_)},
        mx::Shape{
            static_cast<mx::ShapeElem>(B_),
            static_cast<mx::ShapeElem>(T_),
            static_cast<mx::ShapeElem>(H_),
            static_cast<mx::ShapeElem>(Dk_)},
        mx::Shape{
            static_cast<mx::ShapeElem>(B_),
            static_cast<mx::ShapeElem>(T_),
            static_cast<mx::ShapeElem>(H_),
            static_cast<mx::ShapeElem>(Dv_)},
        mx::Shape{
            static_cast<mx::ShapeElem>(B_),
            static_cast<mx::ShapeElem>(T_),
            static_cast<mx::ShapeElem>(H_)},
        mx::Shape{
            static_cast<mx::ShapeElem>(B_),
            static_cast<mx::ShapeElem>(T_),
            static_cast<mx::ShapeElem>(H_)}};
  }

 private:
  int B_;
  int T_;
  int H_;
  int Dk_;
  int Dv_;
  int chunk_size_;
  int window_size_;
  int windows_;
};

#endif

} // namespace

bool gated_delta_metal_primitive_available() {
#ifdef __APPLE__
  return mx::metal::is_available();
#else
  return false;
#endif
}

bool gated_delta_scan_metal_primitive_available(
    int d_k,
    int d_v,
    int chunk_size) {
  if (gated_delta_env_is_one("MIXLAB_DISABLE_GATED_DELTA_METAL_SCAN")) {
    return false;
  }
  if (d_k <= 0 || d_k > kGatedDeltaMetalMaxDK ||
      d_v <= 0 || d_v > kGatedDeltaMetalMaxDV ||
      chunk_size <= 0) {
    return false;
  }
#ifdef __APPLE__
  return mx::metal::is_available();
#else
  return false;
#endif
}

mx::array gated_delta_scan_metal_forward(
    const mx::array& q,
    const mx::array& k,
    const mx::array& v,
    const mx::array& beta,
    const mx::array& gate,
    int B,
    int T,
    int H,
    int Dk,
    int Dv,
    int chunk_size) {
#ifdef __APPLE__
  validate_gated_delta_scan_shape(B, T, H, Dk, Dv, chunk_size);
  static std::atomic<bool> logged{false};
  if (!logged.exchange(true)) {
    const int window =
        gated_delta_metal_backward_window(T, Dk, Dv, chunk_size);
    std::cerr << "[mlx_ir] GatedDeltaNet scan using native Metal forward/backward"
              << " (B=" << B << " T=" << T << " H=" << H
              << " d_k=" << Dk << " d_v=" << Dv
              << " backward=window-checkpointed window=" << window
              << " output_accumulation=direct"
              << "; set MIXLAB_DISABLE_GATED_DELTA_METAL_SCAN=1"
              << " to use the checkpointed MLX fallback)"
              << std::endl;
  }
  auto stream = mx::default_stream(mx::default_device());
  auto primitive = std::make_shared<GatedDeltaScanMetalForwardPrimitive>(
      stream, B, T, H, Dk, Dv, chunk_size);
  auto outputs = mx::array::make_arrays(
      {mx::Shape{
          static_cast<mx::ShapeElem>(B * T * H),
          static_cast<mx::ShapeElem>(Dv)}},
      {mx::float32},
      primitive,
      contiguous_gated_delta_inputs(q, k, v, beta, gate));
  return outputs[0];
#else
  (void)q;
  (void)k;
  (void)v;
  (void)beta;
  (void)gate;
  (void)B;
  (void)T;
  (void)H;
  (void)Dk;
  (void)Dv;
  (void)chunk_size;
  throw std::runtime_error("GatedDeltaNet Metal scan primitive is unavailable");
#endif
}

std::vector<mx::array> gated_delta_scan_metal_vjp(
    const std::vector<mx::array>& args,
    const std::vector<mx::array>& cotangents,
    int B,
    int T,
    int H,
    int Dk,
    int Dv,
    int chunk_size) {
#ifdef __APPLE__
  if (args.size() != 5 || cotangents.size() != 1) {
    throw std::runtime_error(
        "gated_delta_scan_metal_vjp expects 5 args and 1 cotangent");
  }
  validate_gated_delta_scan_shape(B, T, H, Dk, Dv, chunk_size);
  auto inputs = contiguous_gated_delta_inputs(
      args[0], args[1], args[2], args[3], args[4]);
  inputs.push_back(mx::contiguous(mx::astype(cotangents[0], mx::float32)));
  auto stream = mx::default_stream(mx::default_device());
  auto primitive = std::make_shared<GatedDeltaScanMetalBackwardPrimitive>(
      stream, B, T, H, Dk, Dv, chunk_size);
  const int window_size =
      gated_delta_metal_backward_window(T, Dk, Dv, chunk_size);
  const int windows = (T + window_size - 1) / window_size;
  auto outputs = mx::array::make_arrays(
      {
          mx::Shape{
              static_cast<mx::ShapeElem>(B * H * windows * Dk),
              static_cast<mx::ShapeElem>(Dv)},
          mx::Shape{
              static_cast<mx::ShapeElem>(B),
              static_cast<mx::ShapeElem>(T),
              static_cast<mx::ShapeElem>(H),
              static_cast<mx::ShapeElem>(Dk)},
          mx::Shape{
              static_cast<mx::ShapeElem>(B),
              static_cast<mx::ShapeElem>(T),
              static_cast<mx::ShapeElem>(H),
              static_cast<mx::ShapeElem>(Dk)},
          mx::Shape{
              static_cast<mx::ShapeElem>(B),
              static_cast<mx::ShapeElem>(T),
              static_cast<mx::ShapeElem>(H),
              static_cast<mx::ShapeElem>(Dv)},
          mx::Shape{
              static_cast<mx::ShapeElem>(B),
              static_cast<mx::ShapeElem>(T),
              static_cast<mx::ShapeElem>(H)},
          mx::Shape{
              static_cast<mx::ShapeElem>(B),
              static_cast<mx::ShapeElem>(T),
              static_cast<mx::ShapeElem>(H)}},
      std::vector<mx::Dtype>(6, mx::float32),
      primitive,
      inputs);
  return {outputs[1], outputs[2], outputs[3], outputs[4], outputs[5]};
#else
  (void)args;
  (void)cotangents;
  (void)B;
  (void)T;
  (void)H;
  (void)Dk;
  (void)Dv;
  (void)chunk_size;
  throw std::runtime_error("GatedDeltaNet Metal scan primitive is unavailable");
#endif
}

mx::array solve_strictly_lower_metal_primitive(
    const mx::array& raw_attn,
    int matrix_count,
    int chunk_size) {
#ifdef __APPLE__
  auto stream = mx::default_stream(mx::default_device());
  auto primitive = std::make_shared<SolveStrictlyLowerMetalPrimitive>(
      stream,
      matrix_count,
      chunk_size);

  return mx::array(
      mx::Shape{
          static_cast<mx::ShapeElem>(matrix_count),
          static_cast<mx::ShapeElem>(chunk_size),
          static_cast<mx::ShapeElem>(chunk_size)},
      mx::float32,
      primitive,
      std::vector<mx::array>{raw_attn});
#else
  (void)raw_attn;
  (void)matrix_count;
  (void)chunk_size;
  throw std::runtime_error("Metal gated delta solve primitive is unavailable on this platform");
#endif
}

} // namespace mlx_ir
