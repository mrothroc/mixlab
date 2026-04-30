#include "gated_delta_cuda_primitive.h"
#include "cuda_kernel_dispatch.h"

#include <mlx/ops.h>
#include <mlx/primitives.h>
#include <mlx/transforms.h>

#include <functional>
#include <cstdlib>
#include <iostream>
#include <ios>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mx = mlx::core;

namespace mlx_ir {
namespace {

struct ForceUnbufferedStderr {
  ForceUnbufferedStderr() { std::cerr.setf(std::ios::unitbuf); }
} _force_unbuffered_stderr;

constexpr float kGatedDeltaGateFloor = 1e-30f;
constexpr float kGatedDeltaExpClampMin = -80.0f;
constexpr float kGatedDeltaExpClampMax = 0.0f;

mx::array as_float32(const mx::array& x) {
  return mx::astype(x, mx::float32);
}

mx::array clamp_float32(const mx::array& x, float lo, float hi) {
  auto lo_arr = mx::array(lo, mx::float32);
  auto hi_arr = mx::array(hi, mx::float32);
  return mx::minimum(mx::maximum(as_float32(x), lo_arr), hi_arr);
}

mx::array stable_exp_nonpos(const mx::array& x) {
  return mx::exp(clamp_float32(x, kGatedDeltaExpClampMin, kGatedDeltaExpClampMax));
}

bool use_experimental_gated_delta_cuda_kernel() {
  const char* override = std::getenv("MIXLAB_GATED_DELTA_USE_CUDA_KERNEL");
  return override != nullptr && std::string(override) == "1";
}

mx::array solve_strictly_lower_cuda(
    const mx::array& raw_attn,
    int matrix_count,
    int chunk_size,
    mx::Stream stream) {
#ifdef __linux__
  std::cerr << "[gated_delta_cuda] solve_strictly_lower_cuda enter matrix_count="
            << matrix_count << " chunk_size=" << chunk_size << std::endl;
  const int threads = 128;
  const int blocks = (chunk_size + threads - 1) / threads;
  try {
    std::cerr << "[gated_delta_cuda] before precompiled_cuda_kernel" << std::endl;
    auto outputs = launch_precompiled_cuda_kernel(
        "gated_delta_chunk_solve",
        {raw_attn},
        {mx::Shape{
            static_cast<mx::ShapeElem>(matrix_count),
            static_cast<mx::ShapeElem>(chunk_size),
            static_cast<mx::ShapeElem>(chunk_size)}},
        {mx::float32},
        {chunk_size},
        std::make_tuple(blocks, matrix_count, 1),
        std::make_tuple(threads, 1, 1),
        stream);
    std::cerr << "[gated_delta_cuda] precompiled_cuda_kernel returned, outputs="
              << outputs.size() << std::endl;
    if (outputs.size() != 1) {
      throw std::runtime_error("gated_delta_chunk_solve returned unexpected output count");
    }
    return outputs[0];
  } catch (const std::exception& e) {
    std::cerr << "[gated_delta_cuda] EXCEPTION: " << e.what() << std::endl;
    throw;
  }
#else
  (void)matrix_count;
  (void)chunk_size;
  (void)stream;
  return raw_attn;
#endif
}

class GatedDeltaScanCUDAPrimitive : public mx::UnaryPrimitive {
 public:
  GatedDeltaScanCUDAPrimitive(
      mx::Stream stream,
      std::function<std::vector<mx::array>(std::vector<mx::array>)> fallback,
      int B,
      int T,
      int H,
      int Dk,
      int Dv,
      int chunk_size)
      : mx::UnaryPrimitive(stream),
        fallback_(std::move(fallback)),
        B_(B),
        T_(T),
        H_(H),
        Dk_(Dk),
        Dv_(Dv),
        chunk_size_(chunk_size) {}

  void eval_cpu(const std::vector<mx::array>& inputs, mx::array& out) override {
    auto outputs = fallback_(std::vector<mx::array>(inputs.begin(), inputs.end()));
    if (outputs.size() != 1) {
      throw std::runtime_error("GatedDeltaScanCUDAPrimitive fallback returned unexpected output count");
    }
    out = outputs[0];
  }

  void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override {
    std::cerr << "[gated_delta_cuda] eval_gpu ENTERED env="
              << (std::getenv("MIXLAB_GATED_DELTA_USE_CUDA_KERNEL")
                      ? std::getenv("MIXLAB_GATED_DELTA_USE_CUDA_KERNEL")
                      : "(unset)")
              << std::endl;
    if (!use_experimental_gated_delta_cuda_kernel()) {
      eval_cpu(inputs, out);
      return;
    }
    if (inputs.size() != 5) {
      throw std::runtime_error("GatedDeltaScanCUDAPrimitive expects 5 inputs");
    }

    const int pad_len = (chunk_size_ - (T_ % chunk_size_)) % chunk_size_;
    const int T_pad = T_ + pad_len;
    const int n_chunks = T_pad / chunk_size_;
    const int matrix_count = B_ * H_ * n_chunks;

    auto q = as_float32(mx::transpose(inputs[0], {0, 2, 1, 3}));
    auto k = as_float32(mx::transpose(inputs[1], {0, 2, 1, 3}));
    auto v = as_float32(mx::transpose(inputs[2], {0, 2, 1, 3}));
    auto beta = as_float32(mx::transpose(inputs[3], {0, 2, 1}));
    auto gate = as_float32(mx::transpose(inputs[4], {0, 2, 1}));

    if (pad_len > 0) {
      auto q_pad = mx::zeros({B_, H_, pad_len, Dk_}, mx::float32);
      auto k_pad = mx::zeros({B_, H_, pad_len, Dk_}, mx::float32);
      auto v_pad = mx::zeros({B_, H_, pad_len, Dv_}, mx::float32);
      auto beta_pad = mx::zeros({B_, H_, pad_len}, mx::float32);
      auto gate_pad = mx::ones({B_, H_, pad_len}, mx::float32);
      q = mx::concatenate({q, q_pad}, 2);
      k = mx::concatenate({k, k_pad}, 2);
      v = mx::concatenate({v, v_pad}, 2);
      beta = mx::concatenate({beta, beta_pad}, 2);
      gate = mx::concatenate({gate, gate_pad}, 2);
    }

    gate = mx::maximum(gate, mx::array(kGatedDeltaGateFloor, mx::float32));
    q = mx::reshape(q, {B_, H_, n_chunks, chunk_size_, Dk_});
    k = mx::reshape(k, {B_, H_, n_chunks, chunk_size_, Dk_});
    v = mx::reshape(v, {B_, H_, n_chunks, chunk_size_, Dv_});
    beta = mx::reshape(beta, {B_, H_, n_chunks, chunk_size_});
    gate = mx::reshape(gate, {B_, H_, n_chunks, chunk_size_});

    auto v_beta = as_float32(v * mx::expand_dims(beta, -1));
    auto k_beta = as_float32(k * mx::expand_dims(beta, -1));
    auto log_decay = as_float32(mx::cumsum(mx::log(gate), 3));
    auto decay_exp = stable_exp_nonpos(log_decay);

    auto time = mx::astype(mx::arange(chunk_size_), mx::int32);
    auto row_idx = mx::expand_dims(time, 1);
    auto col_idx = mx::expand_dims(time, 0);
    auto strict_lower = row_idx > col_idx;
    auto lower_inclusive = row_idx >= col_idx;
    auto strict_lower_f = mx::astype(
        mx::expand_dims(mx::expand_dims(mx::expand_dims(strict_lower, 0), 0), 0),
        mx::float32);
    auto lower_inclusive_f = mx::astype(
        mx::expand_dims(mx::expand_dims(mx::expand_dims(lower_inclusive, 0), 0), 0),
        mx::float32);

    auto decay_i = mx::expand_dims(log_decay, 4);
    auto decay_j = mx::expand_dims(log_decay, 3);
    auto decay_delta = stable_exp_nonpos(decay_i - decay_j);

    auto raw_attn = as_float32(
        -mx::matmul(k_beta, mx::transpose(k, {0, 1, 2, 4, 3})) * decay_delta * strict_lower_f);
    std::cerr << "[gated_delta_cuda] before solve_strictly_lower_cuda" << std::endl;
    auto solve_attn = solve_strictly_lower_cuda(
        mx::contiguous(mx::reshape(raw_attn, {matrix_count, chunk_size_, chunk_size_})),
        matrix_count,
        chunk_size_,
        stream());
    solve_attn = mx::reshape(solve_attn, {B_, H_, n_chunks, chunk_size_, chunk_size_});
    mx::eval(solve_attn);

    auto k_cumsum = as_float32(mx::matmul(solve_attn, v_beta));
    auto k_cumdecay = as_float32(mx::matmul(
        solve_attn,
        k_beta * mx::expand_dims(decay_exp, -1)));

    auto causal_attn = as_float32(
        mx::matmul(q, mx::transpose(k, {0, 1, 2, 4, 3})) * decay_delta * lower_inclusive_f);
    auto state = mx::zeros({B_, H_, Dk_, Dv_}, mx::float32);
    auto out_buf = mx::zeros({B_, H_, n_chunks, chunk_size_, Dv_}, mx::float32);
    for (int chunk = 0; chunk < n_chunks; ++chunk) {
      auto q_i = mx::reshape(
          mx::slice(q, {0, 0, chunk, 0, 0}, {B_, H_, chunk + 1, chunk_size_, Dk_}),
          {B_, H_, chunk_size_, Dk_});
      auto k_i = mx::reshape(
          mx::slice(k, {0, 0, chunk, 0, 0}, {B_, H_, chunk + 1, chunk_size_, Dk_}),
          {B_, H_, chunk_size_, Dk_});
      auto v_i = mx::reshape(
          mx::slice(k_cumsum, {0, 0, chunk, 0, 0}, {B_, H_, chunk + 1, chunk_size_, Dv_}),
          {B_, H_, chunk_size_, Dv_});
      auto k_decay_i = mx::reshape(
          mx::slice(k_cumdecay, {0, 0, chunk, 0, 0}, {B_, H_, chunk + 1, chunk_size_, Dk_}),
          {B_, H_, chunk_size_, Dk_});
      auto decay_i_chunk = mx::reshape(
          mx::slice(log_decay, {0, 0, chunk, 0}, {B_, H_, chunk + 1, chunk_size_}),
          {B_, H_, chunk_size_});
      auto attn_i = mx::reshape(
          mx::slice(
              causal_attn,
              {0, 0, chunk, 0, 0},
              {B_, H_, chunk + 1, chunk_size_, chunk_size_}),
          {B_, H_, chunk_size_, chunk_size_});

      auto v_prime = as_float32(mx::matmul(k_decay_i, state));
      auto v_new = as_float32(v_i - v_prime);
      auto decay_chunk_exp = stable_exp_nonpos(decay_i_chunk);
      auto o_inter = as_float32(mx::matmul(q_i * mx::expand_dims(decay_chunk_exp, -1), state));
      auto o_chunk = as_float32(o_inter + mx::matmul(attn_i, v_new));
      out_buf = mx::slice_update(
          out_buf,
          mx::reshape(o_chunk, {B_, H_, 1, chunk_size_, Dv_}),
          mx::Shape{0, 0, chunk, 0, 0},
          mx::Shape{B_, H_, chunk + 1, chunk_size_, Dv_});

      auto decay_last = mx::reshape(
          mx::slice(decay_i_chunk, {0, 0, chunk_size_ - 1}, {B_, H_, chunk_size_}),
          {B_, H_});
      auto carry = stable_exp_nonpos(mx::expand_dims(decay_last, -1) - decay_i_chunk);
      auto state_update = as_float32(mx::matmul(
          mx::transpose(k_i * mx::expand_dims(carry, -1), {0, 1, 3, 2}),
          v_new));
      state = as_float32(
          state * mx::reshape(stable_exp_nonpos(decay_last), {B_, H_, 1, 1}) + state_update);
    }

    auto out_seq = mx::reshape(out_buf, {B_, H_, T_pad, Dv_});
    if (pad_len > 0) {
      out_seq = mx::slice(out_seq, {0, 0, 0, 0}, {B_, H_, T_, Dv_});
    }
    out_seq = mx::transpose(out_seq, {0, 2, 1, 3});
    out = mx::reshape(out_seq, {B_ * T_ * H_, Dv_});
    mx::eval(out);
  }

  std::vector<mx::array> vjp(
      const std::vector<mx::array>& primals,
      const std::vector<mx::array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<mx::array>&) override {
    auto fun = [this](const std::vector<mx::array>& inputs) {
      return fallback_(std::vector<mx::array>(inputs.begin(), inputs.end()));
    };
    auto result = mx::vjp(fun, primals, cotangents).second;
    std::vector<mx::array> grads;
    grads.reserve(argnums.size());
    for (int argnum : argnums) {
      if (argnum < 0 || static_cast<size_t>(argnum) >= result.size()) {
        throw std::runtime_error("GatedDeltaScanCUDAPrimitive vjp argnum out of range");
      }
      grads.push_back(result[argnum]);
    }
    return grads;
  }

  const char* name() const override {
    return "GatedDeltaScanCUDAPrimitive";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    auto* rhs = dynamic_cast<const GatedDeltaScanCUDAPrimitive*>(&other);
    return rhs != nullptr &&
        B_ == rhs->B_ &&
        T_ == rhs->T_ &&
        H_ == rhs->H_ &&
        Dk_ == rhs->Dk_ &&
        Dv_ == rhs->Dv_ &&
        chunk_size_ == rhs->chunk_size_;
  }

  std::vector<mx::Shape> output_shapes(const std::vector<mx::array>&) override {
    return {mx::Shape{
        static_cast<mx::ShapeElem>(B_ * T_ * H_),
        static_cast<mx::ShapeElem>(Dv_)}};
  }

 private:
  std::function<std::vector<mx::array>(std::vector<mx::array>)> fallback_;
  int B_;
  int T_;
  int H_;
  int Dk_;
  int Dv_;
  int chunk_size_;
};

} // namespace

mx::array gated_delta_scan_cuda_primitive(
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
    int chunk_size,
    std::function<mx::array(
        const mx::array&,
        const mx::array&,
        const mx::array&,
        const mx::array&,
        const mx::array&)> fallback) {
  auto stream = mx::default_stream(mx::default_device());
  auto primitive = std::make_shared<GatedDeltaScanCUDAPrimitive>(
      stream,
      [fallback = std::move(fallback)](std::vector<mx::array> inputs) {
        if (inputs.size() != 5) {
          throw std::runtime_error("GatedDeltaScanCUDAPrimitive expects 5 inputs");
        }
        return std::vector<mx::array>{fallback(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4])};
      },
      B,
      T,
      H,
      Dk,
      Dv,
      chunk_size);

  return mx::array(
      mx::Shape{
          static_cast<mx::ShapeElem>(B * T * H),
          static_cast<mx::ShapeElem>(Dv)},
      mx::float32,
      primitive,
      std::vector<mx::array>{q, k, v, beta, gate});
}

} // namespace mlx_ir
