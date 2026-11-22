#pragma once
#include "ir.h"
#include <array>

namespace mlx_ir {
struct PolarGeometry { int k, j, h, w, row_column, col_column; float dr; };
struct BilinearPlan { std::array<mlx::core::array, 4> indices, weights; };
PolarGeometry polar_geometry(const IRop& op);
// Shared with independent coordinate and fixed-plan adjoint tests.
std::array<mlx::core::array, 2> polar_coordinates(const PolarGeometry&, const mlx::core::array& vectors, bool inverse);
BilinearPlan polar_bilinear_plan(const mlx::core::array& row, const mlx::core::array& col, int h, int w, bool wrap);
mlx::core::array polar_sample(const mlx::core::array& input, const BilinearPlan& plan);
mlx::core::array polar_resample(const IRop&, const mlx::core::array& input, const mlx::core::array& vectors);
void validate_polar_inputs(const IRProgram&, const TensorMap&);
}
