#pragma once

#include <string>

namespace mlx_ir {

inline constexpr char kMamba3DisableCUDAScan[] =
    "MIXLAB_MAMBA3_DISABLE_CUDA_PRIMITIVE";
inline constexpr char kMamba3AllowMLXScanFallback[] =
    "MIXLAB_ALLOW_MAMBA3_MLX_SCAN_FALLBACK";

// Share the advertised command with the training guard's recovery guidance.
inline std::string mamba3_cuda_scan_fallback_guidance() {
  return std::string("canonical-block CUDA training requires BOTH ") +
      kMamba3DisableCUDAScan + "=1 and " + kMamba3AllowMLXScanFallback +
      "=1 for small debug-only MLX scan fallback runs; keep the fused scan "
      "enabled for production";
}

} // namespace mlx_ir
