#include "gated_delta_chunk_metal_primitive.h"

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

constexpr int kThreads = 32;
constexpr int kMaxDK = 32;
constexpr int kMaxDV = 32;
constexpr int kMaxChunk = 64;
constexpr size_t kThreadgroupBudget = 24 * 1024;

bool env_is_one(const char* name) {
  const char* raw = std::getenv(name);
  return raw != nullptr && std::string(raw) == "1";
}

void validate_shape(
    int B,
    int T,
    int H,
    int Dk,
    int Dv,
    int chunk_size) {
  if (B <= 0 || T <= 0 || H <= 0 || Dk <= 0 || Dv <= 0 ||
      chunk_size <= 0) {
    throw std::runtime_error(
        "chunk-parallel Gated DeltaNet requires positive dimensions");
  }
  if (Dk > kMaxDK || Dv > kMaxDV || chunk_size > kMaxChunk) {
    throw std::runtime_error(
        "chunk-parallel Gated DeltaNet supports d_k <= 32, d_v <= 32, "
        "and scan_chunk_size <= 64");
  }
}

int backward_window(int Dk, int chunk_size) {
  for (int candidate : {8, 4, 2, 1}) {
    const size_t bytes =
        static_cast<size_t>(candidate + 1) *
        static_cast<size_t>(Dk) * kThreads * sizeof(float);
    if (candidate <= chunk_size &&
        (chunk_size % candidate) == 0 &&
        bytes <= kThreadgroupBudget) {
      return candidate;
    }
  }
  return 1;
}

std::vector<mx::array> contiguous_inputs(
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

void require_float32(
    const std::vector<mx::array>& arrays,
    const char* name) {
  for (const auto& array : arrays) {
    if (array.dtype() != mx::float32) {
      throw std::runtime_error(std::string(name) + " requires float32 arrays");
    }
  }
}

#ifdef __APPLE__

const char* kChunkMetalSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

#define GDN_CHUNK_MAX_DK 32
#define GDN_CHUNK_THREADS 32

inline int gdn_qk_idx(int b, int t, int h, int d, int T, int H, int Dk) {
  return ((b * T + t) * H + h) * Dk + d;
}

inline int gdn_v_idx(int b, int t, int h, int u, int T, int H, int Dv) {
  return ((b * T + t) * H + h) * Dv + u;
}

inline int gdn_head_idx(int b, int t, int h, int T, int H) {
  return (b * T + t) * H + h;
}

inline int gdn_chunk_matrix_idx(
    int b, int h, int chunk, int row, int col,
    int H, int chunks, int Dk) {
  return ((((b * H + h) * chunks + chunk) * Dk + row) * Dk) + col;
}

inline int gdn_chunk_state_idx(
    int b, int h, int chunk, int d, int u,
    int H, int chunks, int Dk, int Dv) {
  return ((((b * H + h) * chunks + chunk) * Dk + d) * Dv) + u;
}

inline int gdn_micro_checkpoint_idx(
    int b, int h, int chunk, int micro, int d, int u,
    int H, int chunks, int micros, int Dk, int Dv) {
  return ((((((b * H + h) * chunks + chunk) * micros + micro) * Dk + d) * Dv) + u);
}

inline int gdn_history_idx(int local, int d, int lane, int Dk) {
  return (local * Dk + d) * GDN_CHUNK_THREADS + lane;
}

kernel void gated_delta_chunk_summaries_metal(
    const device float* k [[buffer(0)]],
    const device float* v [[buffer(1)]],
    const device float* beta [[buffer(2)]],
    const device float* gate [[buffer(3)]],
    device float* chunk_a [[buffer(4)]],
    device float* chunk_b [[buffer(5)]],
    constant int& B [[buffer(6)]],
    constant int& T [[buffer(7)]],
    constant int& H [[buffer(8)]],
    constant int& Dk [[buffer(9)]],
    constant int& Dv [[buffer(10)]],
    constant int& chunk_size [[buffer(11)]],
    constant int& chunks [[buffer(12)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
  const int chunk = static_cast<int>(group.x) / H;
  const int h = static_cast<int>(group.x) % H;
  const int b = static_cast<int>(group.y);
  const int lane = static_cast<int>(tid);
  const bool a_active = b < B && h < H && chunk < chunks && lane < Dk;
  const bool b_active = b < B && h < H && chunk < chunks && lane < Dv;
  float a_col[GDN_CHUNK_MAX_DK];
  float b_col[GDN_CHUNK_MAX_DK];
  for (int d = 0; d < Dk; ++d) {
    if (a_active) {
      a_col[d] = d == lane ? 1.0f : 0.0f;
    }
    if (b_active) {
      b_col[d] = 0.0f;
    }
  }

  const int start = chunk * chunk_size;
  const int limit = min(start + chunk_size, T);
  for (int t = start; t < limit; ++t) {
    const int head_idx = gdn_head_idx(b, t, h, T, H);
    const float decay = max(gate[head_idx], 1e-30f);
    if (a_active) {
      float prediction = 0.0f;
      for (int d = 0; d < Dk; ++d) {
        a_col[d] *= decay;
        prediction += k[gdn_qk_idx(b, t, h, d, T, H, Dk)] * a_col[d];
      }
      const float error = -beta[head_idx] * prediction;
      for (int d = 0; d < Dk; ++d) {
        a_col[d] += k[gdn_qk_idx(b, t, h, d, T, H, Dk)] * error;
      }
    }
    if (b_active) {
      float prediction = 0.0f;
      for (int d = 0; d < Dk; ++d) {
        b_col[d] *= decay;
        prediction += k[gdn_qk_idx(b, t, h, d, T, H, Dk)] * b_col[d];
      }
      const float error = beta[head_idx] *
          (v[gdn_v_idx(b, t, h, lane, T, H, Dv)] - prediction);
      for (int d = 0; d < Dk; ++d) {
        b_col[d] += k[gdn_qk_idx(b, t, h, d, T, H, Dk)] * error;
      }
    }
  }

  if (a_active) {
    for (int d = 0; d < Dk; ++d) {
      chunk_a[gdn_chunk_matrix_idx(b, h, chunk, d, lane, H, chunks, Dk)] =
          a_col[d];
    }
  }
  if (b_active) {
    for (int d = 0; d < Dk; ++d) {
      chunk_b[gdn_chunk_state_idx(b, h, chunk, d, lane, H, chunks, Dk, Dv)] =
          b_col[d];
    }
  }
}

kernel void gated_delta_chunk_prefix_metal(
    const device float* chunk_a [[buffer(0)]],
    const device float* chunk_b [[buffer(1)]],
    device float* chunk_starts [[buffer(2)]],
    constant int& B [[buffer(3)]],
    constant int& H [[buffer(4)]],
    constant int& Dk [[buffer(5)]],
    constant int& Dv [[buffer(6)]],
    constant int& chunks [[buffer(7)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
  const int h = static_cast<int>(group.x);
  const int b = static_cast<int>(group.y);
  const int u = static_cast<int>(tid);
  if (b >= B || h >= H || u >= Dv) {
    return;
  }
  float state[GDN_CHUNK_MAX_DK];
  float next[GDN_CHUNK_MAX_DK];
  for (int d = 0; d < Dk; ++d) {
    state[d] = 0.0f;
  }
  for (int chunk = 0; chunk < chunks; ++chunk) {
    for (int d = 0; d < Dk; ++d) {
      chunk_starts[gdn_chunk_state_idx(
          b, h, chunk, d, u, H, chunks, Dk, Dv)] = state[d];
      float value = chunk_b[gdn_chunk_state_idx(
          b, h, chunk, d, u, H, chunks, Dk, Dv)];
      for (int j = 0; j < Dk; ++j) {
        value += chunk_a[gdn_chunk_matrix_idx(
            b, h, chunk, d, j, H, chunks, Dk)] * state[j];
      }
      next[d] = value;
    }
    for (int d = 0; d < Dk; ++d) {
      state[d] = next[d];
    }
  }
}

kernel void gated_delta_chunk_replay_metal(
    const device float* q [[buffer(0)]],
    const device float* k [[buffer(1)]],
    const device float* v [[buffer(2)]],
    const device float* beta [[buffer(3)]],
    const device float* gate [[buffer(4)]],
    const device float* chunk_starts [[buffer(5)]],
    device float* out [[buffer(6)]],
    constant int& B [[buffer(7)]],
    constant int& T [[buffer(8)]],
    constant int& H [[buffer(9)]],
    constant int& Dk [[buffer(10)]],
    constant int& Dv [[buffer(11)]],
    constant int& chunk_size [[buffer(12)]],
    constant int& chunks [[buffer(13)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
  const int chunk = static_cast<int>(group.x) / H;
  const int h = static_cast<int>(group.x) % H;
  const int b = static_cast<int>(group.y);
  const int u = static_cast<int>(tid);
  if (b >= B || h >= H || chunk >= chunks || u >= Dv) {
    return;
  }
  float state[GDN_CHUNK_MAX_DK];
  for (int d = 0; d < Dk; ++d) {
    state[d] = chunk_starts[gdn_chunk_state_idx(
        b, h, chunk, d, u, H, chunks, Dk, Dv)];
  }
  const int start = chunk * chunk_size;
  const int limit = min(start + chunk_size, T);
  for (int t = start; t < limit; ++t) {
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

kernel void gated_delta_chunk_local_carry_metal(
    const device float* q [[buffer(0)]],
    const device float* k [[buffer(1)]],
    const device float* beta [[buffer(2)]],
    const device float* gate [[buffer(3)]],
    const device float* grad_out [[buffer(4)]],
    device float* local_carry [[buffer(5)]],
    constant int& B [[buffer(6)]],
    constant int& T [[buffer(7)]],
    constant int& H [[buffer(8)]],
    constant int& Dk [[buffer(9)]],
    constant int& Dv [[buffer(10)]],
    constant int& chunk_size [[buffer(11)]],
    constant int& chunks [[buffer(12)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
  const int chunk = static_cast<int>(group.x) / H;
  const int h = static_cast<int>(group.x) % H;
  const int b = static_cast<int>(group.y);
  const int u = static_cast<int>(tid);
  if (b >= B || h >= H || chunk >= chunks || u >= Dv) {
    return;
  }
  float dstate[GDN_CHUNK_MAX_DK];
  for (int d = 0; d < Dk; ++d) {
    dstate[d] = 0.0f;
  }
  const int start = chunk * chunk_size;
  const int limit = min(start + chunk_size, T);
  for (int t = limit - 1; t >= start; --t) {
    const int head_idx = gdn_head_idx(b, t, h, T, H);
    const float decay = max(gate[head_idx], 1e-30f);
    const float dout = grad_out[gdn_v_idx(b, t, h, u, T, H, Dv)];
    float k_dot = 0.0f;
    for (int d = 0; d < Dk; ++d) {
      const int qkd = gdn_qk_idx(b, t, h, d, T, H, Dk);
      k_dot += k[qkd] * (dstate[d] + q[qkd] * dout);
    }
    for (int d = 0; d < Dk; ++d) {
      const int qkd = gdn_qk_idx(b, t, h, d, T, H, Dk);
      dstate[d] = decay *
          (dstate[d] + q[qkd] * dout - beta[head_idx] * k[qkd] * k_dot);
    }
  }
  for (int d = 0; d < Dk; ++d) {
    local_carry[gdn_chunk_state_idx(
        b, h, chunk, d, u, H, chunks, Dk, Dv)] = dstate[d];
  }
}

kernel void gated_delta_chunk_reverse_prefix_metal(
    const device float* chunk_a [[buffer(0)]],
    const device float* local_carry [[buffer(1)]],
    device float* end_carry [[buffer(2)]],
    constant int& B [[buffer(3)]],
    constant int& H [[buffer(4)]],
    constant int& Dk [[buffer(5)]],
    constant int& Dv [[buffer(6)]],
    constant int& chunks [[buffer(7)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
  const int h = static_cast<int>(group.x);
  const int b = static_cast<int>(group.y);
  const int u = static_cast<int>(tid);
  if (b >= B || h >= H || u >= Dv) {
    return;
  }
  float carry[GDN_CHUNK_MAX_DK];
  float previous[GDN_CHUNK_MAX_DK];
  for (int d = 0; d < Dk; ++d) {
    carry[d] = 0.0f;
  }
  for (int chunk = chunks - 1; chunk >= 0; --chunk) {
    for (int d = 0; d < Dk; ++d) {
      end_carry[gdn_chunk_state_idx(
          b, h, chunk, d, u, H, chunks, Dk, Dv)] = carry[d];
    }
    for (int j = 0; j < Dk; ++j) {
      float value = local_carry[gdn_chunk_state_idx(
          b, h, chunk, j, u, H, chunks, Dk, Dv)];
      for (int d = 0; d < Dk; ++d) {
        value += chunk_a[gdn_chunk_matrix_idx(
            b, h, chunk, d, j, H, chunks, Dk)] * carry[d];
      }
      previous[j] = value;
    }
    for (int d = 0; d < Dk; ++d) {
      carry[d] = previous[d];
    }
  }
}

kernel void gated_delta_chunk_checkpoints_metal(
    const device float* k [[buffer(0)]],
    const device float* v [[buffer(1)]],
    const device float* beta [[buffer(2)]],
    const device float* gate [[buffer(3)]],
    const device float* chunk_starts [[buffer(4)]],
    device float* checkpoints [[buffer(5)]],
    constant int& B [[buffer(6)]],
    constant int& T [[buffer(7)]],
    constant int& H [[buffer(8)]],
    constant int& Dk [[buffer(9)]],
    constant int& Dv [[buffer(10)]],
    constant int& chunk_size [[buffer(11)]],
    constant int& chunks [[buffer(12)]],
    constant int& window_size [[buffer(13)]],
    constant int& micros [[buffer(14)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
  const int chunk = static_cast<int>(group.x) / H;
  const int h = static_cast<int>(group.x) % H;
  const int b = static_cast<int>(group.y);
  const int u = static_cast<int>(tid);
  if (b >= B || h >= H || chunk >= chunks || u >= Dv) {
    return;
  }
  float state[GDN_CHUNK_MAX_DK];
  for (int d = 0; d < Dk; ++d) {
    state[d] = chunk_starts[gdn_chunk_state_idx(
        b, h, chunk, d, u, H, chunks, Dk, Dv)];
  }
  const int start = chunk * chunk_size;
  const int limit = min(start + chunk_size, T);
  for (int t = start; t < limit; ++t) {
    const int local = t - start;
    if ((local % window_size) == 0) {
      const int micro = local / window_size;
      for (int d = 0; d < Dk; ++d) {
        checkpoints[gdn_micro_checkpoint_idx(
            b, h, chunk, micro, d, u,
            H, chunks, micros, Dk, Dv)] = state[d];
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

kernel void gated_delta_chunk_backward_metal(
    const device float* q [[buffer(0)]],
    const device float* k [[buffer(1)]],
    const device float* v [[buffer(2)]],
    const device float* beta [[buffer(3)]],
    const device float* gate [[buffer(4)]],
    const device float* grad_out [[buffer(5)]],
    const device float* checkpoints [[buffer(6)]],
    const device float* end_carry [[buffer(7)]],
    device float* grad_q [[buffer(8)]],
    device float* grad_k [[buffer(9)]],
    device float* grad_v [[buffer(10)]],
    device float* grad_beta [[buffer(11)]],
    device float* grad_gate [[buffer(12)]],
    constant int& B [[buffer(13)]],
    constant int& T [[buffer(14)]],
    constant int& H [[buffer(15)]],
    constant int& Dk [[buffer(16)]],
    constant int& Dv [[buffer(17)]],
    constant int& chunk_size [[buffer(18)]],
    constant int& chunks [[buffer(19)]],
    constant int& window_size [[buffer(20)]],
    constant int& micros [[buffer(21)]],
    threadgroup float* history [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
  const int chunk = static_cast<int>(group.x) / H;
  const int h = static_cast<int>(group.x) % H;
  const int b = static_cast<int>(group.y);
  const int lane = static_cast<int>(tid);
  const int u = lane;
  const bool active = b < B && h < H && chunk < chunks && u < Dv;
  float dstate[GDN_CHUNK_MAX_DK];
  if (active) {
    for (int d = 0; d < Dk; ++d) {
      dstate[d] = end_carry[gdn_chunk_state_idx(
          b, h, chunk, d, u, H, chunks, Dk, Dv)];
    }
  }

  const int chunk_start = chunk * chunk_size;
  const int chunk_limit = min(chunk_start + chunk_size, T);
  for (int micro = micros - 1; micro >= 0; --micro) {
    const int start = chunk_start + micro * window_size;
    const int limit = min(start + window_size, chunk_limit);
    if (start >= limit) {
      continue;
    }
    if (active) {
      for (int d = 0; d < Dk; ++d) {
        history[gdn_history_idx(0, d, lane, Dk)] =
            checkpoints[gdn_micro_checkpoint_idx(
                b, h, chunk, micro, d, u,
                H, chunks, micros, Dk, Dv)];
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
          prediction += k[gdn_qk_idx(b, t, h, d, T, H, Dk)] *
              history[gdn_history_idx(local, d, lane, Dk)] * decay;
        }
      }
      const float residual = active ? v[vu] - prediction : 0.0f;
      const float error = active ? beta[head_idx] * residual : 0.0f;
      float derror = 0.0f;
      if (active) {
        for (int d = 0; d < Dk; ++d) {
          const int qkd = gdn_qk_idx(b, t, h, d, T, H, Dk);
          const float upstream = dstate[d] + q[qkd] * dout;
          derror += upstream * k[qkd];
        }
      }

      float dgate_partial = 0.0f;
      for (int d = 0; d < Dk; ++d) {
        const int qkd = gdn_qk_idx(b, t, h, d, T, H, Dk);
        const float state_before = active
            ? history[gdn_history_idx(local, d, lane, Dk)] : 0.0f;
        const float state_after = active
            ? history[gdn_history_idx(local + 1, d, lane, Dk)] : 0.0f;
        const float upstream = active ? dstate[d] + q[qkd] * dout : 0.0f;
        const float dprediction = active ? -derror * beta[head_idx] : 0.0f;
        const float dq_partial = active ? dout * state_after : 0.0f;
        const float dk_partial = active
            ? upstream * error + dprediction * state_before * decay : 0.0f;
        const float dq_total = simd_sum(dq_partial);
        const float dk_total = simd_sum(dk_partial);
        if (tid == 0 && b < B && h < H && chunk < chunks) {
          grad_q[qkd] = dq_total;
          grad_k[qkd] = dk_total;
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
      if (tid == 0 && b < B && h < H && chunk < chunks) {
        grad_beta[head_idx] = dbeta_total;
        grad_gate[head_idx] =
            gate[head_idx] > 1e-30f ? dgate_total : 0.0f;
      }
    }
  }
}
)METAL";

class ChunkForwardPrimitive : public mx::Primitive {
 public:
  ChunkForwardPrimitive(
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
        chunks_((T + chunk_size - 1) / chunk_size) {}

  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("ChunkForwardPrimitive is Metal-only");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    if (inputs.size() != 5 || outputs.size() != 4) {
      throw std::runtime_error(
          "ChunkForwardPrimitive expects 5 inputs and 4 outputs");
    }
    require_float32(inputs, "ChunkForwardPrimitive");
    require_float32(outputs, "ChunkForwardPrimitive");
    validate_shape(B_, T_, H_, Dk_, Dv_, chunk_size_);
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
        "mixlab_gated_delta_chunk_metal",
        []() { return std::string(kChunkMetalSource); });

    auto* summaries = device.get_kernel(
        "gated_delta_chunk_summaries_metal", library);
    encoder.set_compute_pipeline_state(summaries);
    encoder.set_input_array(inputs[1], 0);
    encoder.set_input_array(inputs[2], 1);
    encoder.set_input_array(inputs[3], 2);
    encoder.set_input_array(inputs[4], 3);
    encoder.set_output_array(outputs[1], 4);
    encoder.set_output_array(outputs[2], 5);
    bind_summary_constants(encoder, 6);
    dispatch_chunks(encoder);

    auto* prefix = device.get_kernel("gated_delta_chunk_prefix_metal", library);
    encoder.set_compute_pipeline_state(prefix);
    encoder.set_input_array(outputs[1], 0);
    encoder.set_input_array(outputs[2], 1);
    encoder.set_output_array(outputs[3], 2);
    bind_prefix_constants(encoder, 3);
    dispatch_heads(encoder);

    auto* replay = device.get_kernel("gated_delta_chunk_replay_metal", library);
    encoder.set_compute_pipeline_state(replay);
    for (int i = 0; i < 5; ++i) {
      encoder.set_input_array(inputs[static_cast<size_t>(i)], i);
    }
    encoder.set_input_array(outputs[3], 5);
    encoder.set_output_array(outputs[0], 6);
    bind_summary_constants(encoder, 7);
    dispatch_chunks(encoder);
  }

  const char* name() const override {
    return "GatedDeltaChunkMetalForwardPrimitive";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    auto* rhs = dynamic_cast<const ChunkForwardPrimitive*>(&other);
    return rhs != nullptr && B_ == rhs->B_ && T_ == rhs->T_ &&
        H_ == rhs->H_ && Dk_ == rhs->Dk_ && Dv_ == rhs->Dv_ &&
        chunk_size_ == rhs->chunk_size_;
  }

  std::vector<mx::Shape> output_shapes(const std::vector<mx::array>&) override {
    return {
        mx::Shape{B_ * T_ * H_, Dv_},
        mx::Shape{B_ * H_ * chunks_ * Dk_, Dk_},
        mx::Shape{B_ * H_ * chunks_ * Dk_, Dv_},
        mx::Shape{B_ * H_ * chunks_ * Dk_, Dv_}};
  }

 private:
  void bind_summary_constants(mx::metal::CommandEncoder& encoder, int base) {
    encoder.set_bytes(B_, base + 0);
    encoder.set_bytes(T_, base + 1);
    encoder.set_bytes(H_, base + 2);
    encoder.set_bytes(Dk_, base + 3);
    encoder.set_bytes(Dv_, base + 4);
    encoder.set_bytes(chunk_size_, base + 5);
    encoder.set_bytes(chunks_, base + 6);
  }

  void bind_prefix_constants(mx::metal::CommandEncoder& encoder, int base) {
    encoder.set_bytes(B_, base + 0);
    encoder.set_bytes(H_, base + 1);
    encoder.set_bytes(Dk_, base + 2);
    encoder.set_bytes(Dv_, base + 3);
    encoder.set_bytes(chunks_, base + 4);
  }

  void dispatch_chunks(mx::metal::CommandEncoder& encoder) {
    encoder.dispatch_threadgroups(
        MTL::Size::Make(H_ * chunks_, B_, 1),
        MTL::Size::Make(kThreads, 1, 1));
  }

  void dispatch_heads(mx::metal::CommandEncoder& encoder) {
    encoder.dispatch_threadgroups(
        MTL::Size::Make(H_, B_, 1),
        MTL::Size::Make(kThreads, 1, 1));
  }

  int B_;
  int T_;
  int H_;
  int Dk_;
  int Dv_;
  int chunk_size_;
  int chunks_;
};

class ChunkBackwardPrimitive : public mx::Primitive {
 public:
  ChunkBackwardPrimitive(
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
        chunks_((T + chunk_size - 1) / chunk_size),
        window_size_(backward_window(Dk, chunk_size)),
        micros_((chunk_size + window_size_ - 1) / window_size_) {}

  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("ChunkBackwardPrimitive is Metal-only");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    if (inputs.size() != 6 || outputs.size() != 11) {
      throw std::runtime_error(
          "ChunkBackwardPrimitive expects 6 inputs and 11 outputs");
    }
    require_float32(inputs, "ChunkBackwardPrimitive");
    require_float32(outputs, "ChunkBackwardPrimitive");
    validate_shape(B_, T_, H_, Dk_, Dv_, chunk_size_);
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
        "mixlab_gated_delta_chunk_metal",
        []() { return std::string(kChunkMetalSource); });

    encode_summaries(device, encoder, library, inputs, outputs);
    encode_prefix(device, encoder, library, outputs);
    encode_carries(device, encoder, library, inputs, outputs);
    encode_checkpoints(device, encoder, library, inputs, outputs);
    encode_backward(device, encoder, library, inputs, outputs);
  }

  const char* name() const override {
    return "GatedDeltaChunkMetalBackwardPrimitive";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    auto* rhs = dynamic_cast<const ChunkBackwardPrimitive*>(&other);
    return rhs != nullptr && B_ == rhs->B_ && T_ == rhs->T_ &&
        H_ == rhs->H_ && Dk_ == rhs->Dk_ && Dv_ == rhs->Dv_ &&
        chunk_size_ == rhs->chunk_size_ && window_size_ == rhs->window_size_;
  }

  std::vector<mx::Shape> output_shapes(const std::vector<mx::array>&) override {
    const mx::Shape chunk_state{B_ * H_ * chunks_ * Dk_, Dv_};
    return {
        mx::Shape{B_ * H_ * chunks_ * Dk_, Dk_},
        chunk_state,
        chunk_state,
        chunk_state,
        chunk_state,
        mx::Shape{B_ * H_ * chunks_ * micros_ * Dk_, Dv_},
        mx::Shape{B_, T_, H_, Dk_},
        mx::Shape{B_, T_, H_, Dk_},
        mx::Shape{B_, T_, H_, Dv_},
        mx::Shape{B_, T_, H_},
        mx::Shape{B_, T_, H_}};
  }

 private:
  void bind_summary_constants(mx::metal::CommandEncoder& encoder, int base) {
    encoder.set_bytes(B_, base + 0);
    encoder.set_bytes(T_, base + 1);
    encoder.set_bytes(H_, base + 2);
    encoder.set_bytes(Dk_, base + 3);
    encoder.set_bytes(Dv_, base + 4);
    encoder.set_bytes(chunk_size_, base + 5);
    encoder.set_bytes(chunks_, base + 6);
  }

  void bind_prefix_constants(mx::metal::CommandEncoder& encoder, int base) {
    encoder.set_bytes(B_, base + 0);
    encoder.set_bytes(H_, base + 1);
    encoder.set_bytes(Dk_, base + 2);
    encoder.set_bytes(Dv_, base + 3);
    encoder.set_bytes(chunks_, base + 4);
  }

  void bind_window_constants(mx::metal::CommandEncoder& encoder, int base) {
    bind_summary_constants(encoder, base);
    encoder.set_bytes(window_size_, base + 7);
    encoder.set_bytes(micros_, base + 8);
  }

  void dispatch_chunks(mx::metal::CommandEncoder& encoder) {
    encoder.dispatch_threadgroups(
        MTL::Size::Make(H_ * chunks_, B_, 1),
        MTL::Size::Make(kThreads, 1, 1));
  }

  void dispatch_heads(mx::metal::CommandEncoder& encoder) {
    encoder.dispatch_threadgroups(
        MTL::Size::Make(H_, B_, 1),
        MTL::Size::Make(kThreads, 1, 1));
  }

  void encode_summaries(
      mx::metal::Device& device,
      mx::metal::CommandEncoder& encoder,
      MTL::Library* library,
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) {
    auto* kernel = device.get_kernel("gated_delta_chunk_summaries_metal", library);
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(inputs[1], 0);
    encoder.set_input_array(inputs[2], 1);
    encoder.set_input_array(inputs[3], 2);
    encoder.set_input_array(inputs[4], 3);
    encoder.set_output_array(outputs[0], 4);
    encoder.set_output_array(outputs[1], 5);
    bind_summary_constants(encoder, 6);
    dispatch_chunks(encoder);
  }

  void encode_prefix(
      mx::metal::Device& device,
      mx::metal::CommandEncoder& encoder,
      MTL::Library* library,
      std::vector<mx::array>& outputs) {
    auto* kernel = device.get_kernel("gated_delta_chunk_prefix_metal", library);
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(outputs[0], 0);
    encoder.set_input_array(outputs[1], 1);
    encoder.set_output_array(outputs[2], 2);
    bind_prefix_constants(encoder, 3);
    dispatch_heads(encoder);
  }

  void encode_carries(
      mx::metal::Device& device,
      mx::metal::CommandEncoder& encoder,
      MTL::Library* library,
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) {
    auto* local = device.get_kernel("gated_delta_chunk_local_carry_metal", library);
    encoder.set_compute_pipeline_state(local);
    encoder.set_input_array(inputs[0], 0);
    encoder.set_input_array(inputs[1], 1);
    encoder.set_input_array(inputs[3], 2);
    encoder.set_input_array(inputs[4], 3);
    encoder.set_input_array(inputs[5], 4);
    encoder.set_output_array(outputs[3], 5);
    bind_summary_constants(encoder, 6);
    dispatch_chunks(encoder);

    auto* reverse = device.get_kernel(
        "gated_delta_chunk_reverse_prefix_metal", library);
    encoder.set_compute_pipeline_state(reverse);
    encoder.set_input_array(outputs[0], 0);
    encoder.set_input_array(outputs[3], 1);
    encoder.set_output_array(outputs[4], 2);
    bind_prefix_constants(encoder, 3);
    dispatch_heads(encoder);
  }

  void encode_checkpoints(
      mx::metal::Device& device,
      mx::metal::CommandEncoder& encoder,
      MTL::Library* library,
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) {
    auto* kernel = device.get_kernel(
        "gated_delta_chunk_checkpoints_metal", library);
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(inputs[1], 0);
    encoder.set_input_array(inputs[2], 1);
    encoder.set_input_array(inputs[3], 2);
    encoder.set_input_array(inputs[4], 3);
    encoder.set_input_array(outputs[2], 4);
    encoder.set_output_array(outputs[5], 5);
    bind_window_constants(encoder, 6);
    dispatch_chunks(encoder);
  }

  void encode_backward(
      mx::metal::Device& device,
      mx::metal::CommandEncoder& encoder,
      MTL::Library* library,
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) {
    auto* kernel = device.get_kernel("gated_delta_chunk_backward_metal", library);
    encoder.set_compute_pipeline_state(kernel);
    for (int i = 0; i < 6; ++i) {
      encoder.set_input_array(inputs[static_cast<size_t>(i)], i);
    }
    encoder.set_input_array(outputs[5], 6);
    encoder.set_input_array(outputs[4], 7);
    for (int i = 6; i < 11; ++i) {
      encoder.set_output_array(outputs[static_cast<size_t>(i)], i + 2);
    }
    bind_window_constants(encoder, 13);
    const size_t history_bytes =
        static_cast<size_t>(window_size_ + 1) * Dk_ * kThreads * sizeof(float);
    encoder.set_threadgroup_memory_length(history_bytes, 0);
    dispatch_chunks(encoder);
  }

  int B_;
  int T_;
  int H_;
  int Dk_;
  int Dv_;
  int chunk_size_;
  int chunks_;
  int window_size_;
  int micros_;
};

#endif

} // namespace

#include "gated_delta_chunk_metal_api.inc"

} // namespace mlx_ir
