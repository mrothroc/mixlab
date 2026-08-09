#include "s4d_sobolev_filter_common.cuh"

extern "C" __global__ void s4d_sobolev_filter_forward(
    const MixlabComplex64* product,
    const float* beta,
    MixlabComplex64* output,
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
  output[index].real = product[index].real * factor;
  output[index].imag = product[index].imag * factor;
}
