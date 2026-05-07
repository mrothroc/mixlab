#ifndef MLX_MAMBA3_CUDA_PRIMITIVE_H
#define MLX_MAMBA3_CUDA_PRIMITIVE_H

#include <mlx/array.h>

#include <vector>

namespace mlx_ir {

bool mamba3_selective_scan_cuda_primitive_available(int state_size);

mlx::core::array mamba3_selective_scan_cuda_forward(
    const mlx::core::array& x_flat,
    const mlx::core::array& dt_flat,
    const mlx::core::array& lambda_flat,
    const mlx::core::array& theta_flat,
    const mlx::core::array& a_log,
    const mlx::core::array& b_proj_flat,
    const mlx::core::array& c_proj_flat,
    int B,
    int T,
    int D,
    int N,
    int G);

std::vector<mlx::core::array> mamba3_selective_scan_cuda_vjp(
    const std::vector<mlx::core::array>& args,
    const std::vector<mlx::core::array>& cotangents,
    int B,
    int T,
    int D,
    int N,
    int G);

} // namespace mlx_ir

#endif
