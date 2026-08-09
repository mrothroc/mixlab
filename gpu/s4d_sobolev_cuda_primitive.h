#ifndef MLX_S4D_SOBOLEV_CUDA_PRIMITIVE_H
#define MLX_S4D_SOBOLEV_CUDA_PRIMITIVE_H

#include <mlx/array.h>

namespace mlx_ir {

bool s4d_sobolev_cuda_primitive_available();

mlx::core::array s4d_sobolev_filter_cuda_primitive(
    const mlx::core::array& product,
    const mlx::core::array& beta,
    int fft_len);

} // namespace mlx_ir

#endif
