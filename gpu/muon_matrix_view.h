#pragma once

#include <mlx/mlx.h>
#include <stdexcept>

namespace mlx_ir {
// Optimizer-only reversible views. Parameters and momentum retain storage shape.
inline mlx::core::array muon_matrix_view(const mlx::core::array& x, int view) {
  namespace mx = mlx::core;
  if (view == 0 && x.ndim() == 2) return x;
  if (x.ndim() != 4 || (view != 1 && view != 2)) {
    throw std::runtime_error("Muon requires rank-2 weights or an explicit rank-4 kernel view");
  }
  if (view == 1) return mx::reshape(x, {x.shape(0), -1});
  return mx::reshape(mx::transpose(x, {3, 1, 2, 0}), {x.shape(3), -1});
}

inline mlx::core::array muon_storage_view(
    const mlx::core::array& x, const mlx::core::Shape& shape, int view) {
  namespace mx = mlx::core;
  if (view == 0) return x;
  if (view == 1) return mx::reshape(x, shape);
  return mx::transpose(mx::reshape(x, {shape[3], shape[1], shape[2], shape[0]}), {3, 1, 2, 0});
}
} // namespace mlx_ir
