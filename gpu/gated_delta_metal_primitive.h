#ifndef MLX_GATED_DELTA_METAL_PRIMITIVE_H
#define MLX_GATED_DELTA_METAL_PRIMITIVE_H

#include <mlx/array.h>

#include <vector>

namespace mlx_ir {

mlx::core::array solve_strictly_lower_metal_primitive(
    const mlx::core::array& raw_attn,
    int matrix_count,
    int chunk_size);

bool gated_delta_metal_primitive_available();

bool gated_delta_scan_metal_primitive_available(
    int d_k,
    int d_v,
    int chunk_size);

mlx::core::array gated_delta_scan_metal_forward(
    const mlx::core::array& q,
    const mlx::core::array& k,
    const mlx::core::array& v,
    const mlx::core::array& beta,
    const mlx::core::array& gate,
    int B,
    int T,
    int H,
    int Dk,
    int Dv,
    int chunk_size);

std::vector<mlx::core::array> gated_delta_scan_metal_vjp(
    const std::vector<mlx::core::array>& args,
    const std::vector<mlx::core::array>& cotangents,
    int B,
    int T,
    int H,
    int Dk,
    int Dv,
    int chunk_size);

} // namespace mlx_ir

#endif
