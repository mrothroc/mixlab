#ifndef MLX_S4D_KERNEL_METAL_PRIMITIVE_H
#define MLX_S4D_KERNEL_METAL_PRIMITIVE_H

#include <mlx/array.h>

namespace mlx_ir {

bool s4d_bidirectional_kernel_metal_primitive_available();

mlx::core::array s4d_bidirectional_kernel_metal_primitive(
    const mlx::core::array& b_real,
    const mlx::core::array& b_imag,
    const mlx::core::array& log_magnitude,
    const mlx::core::array& phase,
    const mlx::core::array& c_forward_real,
    const mlx::core::array& c_forward_imag,
    const mlx::core::array& c_backward_real,
    const mlx::core::array& c_backward_imag,
    int D,
    int T,
    int state_pairs);

} // namespace mlx_ir

#endif
