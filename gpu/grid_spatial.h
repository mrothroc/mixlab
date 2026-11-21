#pragma once

#include "ir.h"

namespace mlx_ir {
mlx::core::array spatial_convolution(const IRop& op, const std::vector<mlx::core::array>& inputs);
mlx::core::array spatial_pad(const IRop& op, const mlx::core::array& input);
}
