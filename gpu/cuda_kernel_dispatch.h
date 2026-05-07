#ifndef MLX_IR_CUDA_KERNEL_DISPATCH_H
#define MLX_IR_CUDA_KERNEL_DISPATCH_H

#include <mlx/array.h>
#include <mlx/fast.h>

#include <string>
#include <tuple>
#include <vector>

namespace mlx_ir {

std::vector<mlx::core::array> launch_precompiled_cuda_kernel(
    const std::string& kernel_name,
    const std::vector<mlx::core::array>& inputs,
    const std::vector<mlx::core::Shape>& output_shapes,
    const std::vector<mlx::core::Dtype>& output_dtypes,
    const std::vector<mlx::core::fast::ScalarArg>& scalars,
    std::tuple<int, int, int> grid,
    std::tuple<int, int, int> threadgroup,
    mlx::core::Stream stream,
    int shared_memory = 0,
    bool ensure_row_contiguous = false);

void launch_precompiled_cuda_kernel_into(
    const std::string& kernel_name,
    const std::vector<mlx::core::array>& inputs,
    const std::vector<mlx::core::array*>& outputs,
    const std::vector<mlx::core::fast::ScalarArg>& scalars,
    std::tuple<int, int, int> grid,
    std::tuple<int, int, int> threadgroup,
    mlx::core::Stream stream,
    int shared_memory = 0,
    bool allocate_outputs = true);

} // namespace mlx_ir

#endif
