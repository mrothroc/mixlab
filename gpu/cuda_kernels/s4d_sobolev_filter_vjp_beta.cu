#include "s4d_sobolev_filter_common.cuh"

extern "C" __global__ void s4d_sobolev_filter_vjp_beta(
    const MixlabComplex64* product,
    const MixlabComplex64* cotangent,
    const float* beta,
    float* grad_beta,
    int batch_size,
    int frequency_bins,
    int width,
    int fft_len) {
  extern __shared__ double partials[];
  const int feature = static_cast<int>(blockIdx.x);
  const int thread = static_cast<int>(threadIdx.x);
  double sum = 0.0;
  const int rows = batch_size * frequency_bins;
  for (int row = thread; row < rows; row += static_cast<int>(blockDim.x)) {
    const int frequency = row % frequency_bins;
    const int index = row * width + feature;
    const float base = 1.0f + static_cast<float>(frequency) /
        static_cast<float>(fft_len);
    const float factor = mixlab_s4d_sobolev_factor(
        frequency, fft_len, beta[feature]);
    const double inner =
        static_cast<double>(product[index].real) * cotangent[index].real +
        static_cast<double>(product[index].imag) * cotangent[index].imag;
    sum += inner * static_cast<double>(factor) *
        static_cast<double>(logf(base));
  }
  partials[thread] = sum;
  __syncthreads();
  for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
    if (thread < stride) {
      partials[thread] += partials[thread + stride];
    }
    __syncthreads();
  }
  if (thread == 0) {
    grad_beta[feature] = static_cast<float>(partials[0]);
  }
}
