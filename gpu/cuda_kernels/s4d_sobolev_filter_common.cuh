#ifndef MIXLAB_S4D_SOBOLEV_FILTER_COMMON_CUH
#define MIXLAB_S4D_SOBOLEV_FILTER_COMMON_CUH

#include <math.h>

struct MixlabComplex64 {
  float real;
  float imag;
};

static_assert(sizeof(MixlabComplex64) == 2 * sizeof(float));

__device__ __forceinline__ float mixlab_s4d_sobolev_factor(
    int frequency,
    int fft_len,
    float beta) {
  if (beta == 0.0f) {
    return 1.0f;
  }
  const float base = 1.0f + static_cast<float>(frequency) /
      static_cast<float>(fft_len);
  return powf(base, beta);
}

#endif
