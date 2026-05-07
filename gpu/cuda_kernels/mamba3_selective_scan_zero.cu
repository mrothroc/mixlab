extern "C" __global__ void mamba3_selective_scan_zero(
    float* h_checkpoints,
    float* phi_checkpoints,
    float* grad_x,
    float* grad_dt,
    float* grad_lambda,
    float* grad_theta,
    float* grad_a_log,
    float* grad_b,
    float* grad_c,
    int h_checkpoints_size,
    int phi_checkpoints_size,
    int grad_x_size,
    int grad_dt_size,
    int grad_lambda_size,
    int grad_theta_size,
    int grad_a_log_size,
    int grad_b_size,
    int grad_c_size) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < h_checkpoints_size) {
    h_checkpoints[i] = 0.0f;
  }
  if (i < phi_checkpoints_size) {
    phi_checkpoints[i] = 0.0f;
  }
  if (i < grad_x_size) {
    grad_x[i] = 0.0f;
  }
  if (i < grad_dt_size) {
    grad_dt[i] = 0.0f;
  }
  if (i < grad_lambda_size) {
    grad_lambda[i] = 0.0f;
  }
  if (i < grad_theta_size) {
    grad_theta[i] = 0.0f;
  }
  if (i < grad_a_log_size) {
    grad_a_log[i] = 0.0f;
  }
  if (i < grad_b_size) {
    grad_b[i] = 0.0f;
  }
  if (i < grad_c_size) {
    grad_c[i] = 0.0f;
  }
}
