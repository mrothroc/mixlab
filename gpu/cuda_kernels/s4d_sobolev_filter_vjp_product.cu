#include "s4d_sobolev_filter_common.cuh"

extern "C" __global__ void s4d_sobolev_filter_vjp_product(
    const MixlabComplex64* cotangent,
    const float* beta,
    MixlabComplex64* grad_product,
    int batch_size,
    int frequency_bins,
    int width,
    int fft_len) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int elements = batch_size * frequency_bins * width;
  if (index >= elements) {
    return;
  }
  const int frequency = (index / width) % frequency_bins;
  const int feature = index % width;
  const float factor = mixlab_s4d_sobolev_factor(
      frequency, fft_len, beta[feature]);
  grad_product[index].real = cotangent[index].real * factor;
  grad_product[index].imag = cotangent[index].imag * factor;
}
