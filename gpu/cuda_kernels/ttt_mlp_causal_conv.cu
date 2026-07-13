extern "C" __global__ void ttt_mlp_causal_conv(
    const float* x,
    const float* history,
    const float* weight,
    float* out,
    int batch_size,
    int token_count,
    int width) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int sequence_width = token_count * width;
  if (index >= batch_size * sequence_width) {
    return;
  }
  const int batch = index / sequence_width;
  const int within_batch = index - batch * sequence_width;
  const int token = within_batch / width;
  const int channel = within_batch - token * width;
  if (token >= token_count) {
    return;
  }

  float sum = 0.0f;
  for (int lag = 0; lag < 4; ++lag) {
    const int source_token = token - lag;
    float value;
    if (source_token >= 0) {
      value = x[(batch * token_count + source_token) * width + channel];
    } else {
      value = history[(batch * 3 + (3 + source_token)) * width + channel];
    }
    sum += value * weight[channel * 4 + (3 - lag)];
  }
  out[index] = sum;
}
