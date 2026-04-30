#ifndef MLX_GATED_DELTA_CUDA_PRIMITIVE_H
#define MLX_GATED_DELTA_CUDA_PRIMITIVE_H

#include <mlx/array.h>

namespace mlx_ir {

mlx::core::array solve_strictly_lower_cuda_primitive(
    const mlx::core::array& raw_attn,
    int matrix_count,
    int chunk_size);

} // namespace mlx_ir

#endif
