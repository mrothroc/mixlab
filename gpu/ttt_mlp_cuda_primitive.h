#ifndef MLX_TTT_MLP_CUDA_PRIMITIVE_H
#define MLX_TTT_MLP_CUDA_PRIMITIVE_H

#include <mlx/array.h>

namespace mlx_ir {

bool ttt_mlp_causal_conv_cuda_primitive_available();

mlx::core::array ttt_mlp_causal_conv_cuda(
    const mlx::core::array& x,
    const mlx::core::array& history,
    const mlx::core::array& weight,
    int batch,
    int token_count,
    int width);

} // namespace mlx_ir

#endif
