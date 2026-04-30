extern "C" __global__ void gated_delta_chunk_solve(
    const float* raw_attn,
    float* solve_attn,
    int chunk_size) {
  const int matrix_idx = static_cast<int>(blockIdx.y);
  const int col = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (col >= chunk_size) {
    return;
  }

  const int matrix_elems = chunk_size * chunk_size;
  const int base = matrix_idx * matrix_elems;

  for (int row = 0; row < chunk_size; ++row) {
    float acc = 0.0f;
    if (row == col) {
      acc = 1.0f;
    } else if (row > col) {
      const int row_offset = base + row * chunk_size;
      for (int j = col; j < row; ++j) {
        acc += raw_attn[row_offset + j] * solve_attn[base + j * chunk_size + col];
      }
    }
    solve_attn[base + row * chunk_size + col] = acc;
  }
}
