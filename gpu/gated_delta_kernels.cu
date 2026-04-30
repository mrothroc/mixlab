extern "C" __global__ void gated_delta_chunk_passthrough(
    const float* raw_attn,
    float* solve_attn,
    int chunk_size) {
  const int matrix_idx = static_cast<int>(blockIdx.y);
  const int linear_idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int matrix_elems = chunk_size * chunk_size;
  if (linear_idx >= matrix_elems) {
    return;
  }
  const int base = matrix_idx * matrix_elems;
  solve_attn[base + linear_idx] = raw_attn[base + linear_idx];
}
