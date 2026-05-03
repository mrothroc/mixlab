#ifndef MLX_GATED_DELTA_METAL_PRIMITIVE_H
#define MLX_GATED_DELTA_METAL_PRIMITIVE_H

#include <mlx/array.h>

namespace mlx_ir {

mlx::core::array solve_strictly_lower_metal_primitive(
    const mlx::core::array& raw_attn,
    int matrix_count,
    int chunk_size);

bool gated_delta_metal_primitive_available();

} // namespace mlx_ir

#endif
