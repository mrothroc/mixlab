#include "grid_spatial.h"

#include <array>
#include <cstdint>
#include <stdexcept>

namespace mx = mlx::core;
namespace mlx_ir {
namespace {
struct ConvParameters {
  std::array<int, 2> kernel, stride, padding, dilation;
};

ConvParameters convolution_parameters(const IRop& op) {
  if (op.n_float_params != 0 || (op.n_inputs != 2 && op.n_inputs != 3) || op.n_outputs != 1)
    throw std::runtime_error("convolution requires input,weight[,bias] and one output");
  const int* p = op.int_params;
  if (op.type == OP_CONV2D && op.n_int_params == 8)
    return {{p[0], p[1]}, {p[2], p[3]}, {p[4], p[5]}, {p[6], p[7]}};
  if (op.n_int_params != 3 && !(op.type == OP_CONV2D && op.n_int_params == 4))
    throw std::runtime_error("convolution requires 3 legacy params, 4 dilated params, or 8 conv2d per-axis params");
  int d = op.n_int_params == 4 ? p[3] : 1;
  return {{p[0], p[0]}, {p[1], p[1]}, {p[2], p[2]}, {d, d}};
}

void checked_shape(const mx::Shape& shape) {
  int64_t size = 1;
  for (int d : shape) {
    if (d <= 0 || size > INT32_MAX / int64_t(d))
      throw std::runtime_error("spatial output exceeds positive int32 shape/product bounds");
    size *= d;
  }
}
}  // namespace

mx::array spatial_convolution(const IRop& op, const std::vector<mx::array>& inputs) {
  auto p = convolution_parameters(op);
  auto x = inputs.at(0), w = inputs.at(1);
  bool transpose = op.type == OP_CONV_TRANSPOSE2D;
  if (x.ndim() != 4 || w.ndim() != 4)
    throw std::runtime_error("convolution requires rank-4 input and weight");
  int cin = w.shape(transpose ? 0 : 3), cout = w.shape(transpose ? 3 : 0);
  if (x.shape(3) != cin) throw std::runtime_error("convolution input channels mismatch");
  mx::Shape output = {x.shape(0), 0, 0, cout};
  for (int a = 0; a < 2; ++a) {
    int k = p.kernel[a], s = p.stride[a], pad = p.padding[a], d = p.dilation[a];
    if (k <= 0 || s <= 0 || pad < 0 || d <= 0 || w.shape(a + 1) != k)
      throw std::runtime_error("invalid convolution kernel/stride/padding/dilation");
    int64_t effective = int64_t(d) * (k - 1) + 1;
    int64_t padded = int64_t(x.shape(a + 1)) + 2 * int64_t(pad);
    int64_t n = 0;
    if (transpose) {
      n = int64_t(x.shape(a + 1) - 1) * s - 2 * int64_t(pad) + k;
    } else {
      if (effective > INT32_MAX || padded > INT32_MAX || padded < effective)
        throw std::runtime_error("invalid convolution effective kernel or padded input");
      n = (padded - effective) / s + 1;
    }
    if (n <= 0 || n > INT32_MAX) throw std::runtime_error("invalid convolution output dimension");
    output[a + 1] = int(n);
  }
  checked_shape(output);
  // Canonical transposed weights are IHWO; MLX expects OHWI.
  if (transpose) w = mx::transpose(w, {3, 1, 2, 0});
  std::pair<int, int> stride = {p.stride[0], p.stride[1]};
  std::pair<int, int> padding = {p.padding[0], p.padding[1]};
  std::pair<int, int> dilation = {p.dilation[0], p.dilation[1]};
  auto y = transpose ? mx::conv_transpose2d(x, w, stride, padding)
                     : mx::conv2d(x, w, stride, padding, dilation);
  if (inputs.size() == 3) {
    auto bias = inputs[2];
    if (!((bias.ndim() == 1 && bias.shape(0) == cout) ||
          (bias.ndim() == 2 && bias.shape(0) == 1 && bias.shape(1) == cout)))
      throw std::runtime_error("convolution bias must be a channel vector");
    y = y + mx::reshape(bias, {cout});
  }
  return y;
}

mx::array spatial_pad(const IRop& op, const mx::array& input) {
  if (op.n_inputs != 1 || op.n_outputs != 1 || op.n_int_params != 5 ||
      op.n_float_params != 0 || input.ndim() != 4)
    throw std::runtime_error("pad requires one rank-4 input and five integer params");
  auto p = op.int_params;
  int mode = p[4];
  if (mode < 0 || mode > 2) throw std::runtime_error("invalid spatial pad mode");
  checked_shape(input.shape());
  auto shape = input.shape();
  for (int axis = 1; axis <= 2; ++axis) {
    int lo = p[2 * (axis - 1)], hi = p[2 * (axis - 1) + 1];
    int64_t n = int64_t(shape[axis]) + lo + int64_t(hi);
    if (lo < 0 || hi < 0 || n > INT32_MAX ||
        (mode == 2 && (lo > shape[axis] || hi > shape[axis])))
      throw std::runtime_error("invalid spatial pad extent");
    shape[axis] = int(n);
  }
  checked_shape(shape);
  if (mode == 0) {
    // MLX 0.32.0 Pad::vjp indexes padding vectors by absolute axis. Supply
    // full-rank vectors so its backward does not index past a two-axis vector.
    return mx::pad(input, {0, 1, 2, 3}, {0, p[0], p[2], 0},
                   {0, p[1], p[3], 0}, mx::array(0, input.dtype()));
  }
  auto result = input;
  for (int axis = 1; axis <= 2; ++axis) {
    int lo = p[2 * (axis - 1)], hi = p[2 * (axis - 1) + 1], n = input.shape(axis);
    if (lo == 0 && hi == 0) continue;
    auto indices = mx::arange(-lo, n + hi, 1, mx::int32);
    if (mode == 1) {
      indices = mx::minimum(mx::maximum(indices, mx::array(0, mx::int32)), mx::array(n - 1, mx::int32));
    } else {
      // Each extent is <= n, so one suffix/prefix suffices. Avoid integer
      // remainder: its fused CUDA path fails to compile on the pinned MLX.
      std::vector<mx::array> segments;
      if (lo > 0) segments.push_back(mx::arange(n - lo, n, 1, mx::int32));
      segments.push_back(mx::arange(0, n, 1, mx::int32));
      if (hi > 0) segments.push_back(mx::arange(0, hi, 1, mx::int32));
      indices = mx::concatenate(segments, 0);
    }
    // take's VJP sums repeated border/wrap indices; no host copy severs gradients.
    result = mx::take(result, indices, axis);
  }
  return result;
}
}  // namespace mlx_ir
