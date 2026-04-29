#ifndef MLX_GATED_DELTA_CUDA_PRIMITIVE_H
#define MLX_GATED_DELTA_CUDA_PRIMITIVE_H

#include <mlx/array.h>

#include <functional>

namespace mlx_ir {

mlx::core::array gated_delta_scan_cuda_primitive(
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
    int chunk_size,
    std::function<mlx::core::array(
        const mlx::core::array&,
        const mlx::core::array&,
        const mlx::core::array&,
        const mlx::core::array&,
        const mlx::core::array&)> fallback);

} // namespace mlx_ir

#endif
