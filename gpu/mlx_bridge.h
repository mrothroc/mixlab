#ifndef MLX_BRIDGE_H
#define MLX_BRIDGE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Set CUDA graph batch limits before MLX initialization.
// Must be called before mlx_init(). Values <= 0 are ignored.
void mlx_set_cuda_graph_limits(int max_ops, int max_mb);

// Initialize MLX - returns 0 on success, -1 on failure
int mlx_init(void);

// Get device name (e.g. "Apple M1 Max" or "NVIDIA RTX 4090")
const char* mlx_device_name(void);

// Matrix multiply: C = A(m,k) * B(k,n), row-major float32
// Returns 0 on success, -1 on failure
int mlx_sgemm(const float* A, const float* B, float* C, int m, int k, int n);
int mlx_sgemm_transA(const float* A, const float* B, float* C, int m, int k, int n);
int mlx_sgemm_transB(const float* A, const float* B, float* C, int m, int k, int n);

// Handle-based lazy MLX graph API.
int64_t mlx_from_data(const float* data, int rows, int cols);
int64_t mlx_from_data_nocopy(const float* data, int rows, int cols);
int64_t mlx_lazy_matmul(int64_t A, int64_t B);
int64_t mlx_lazy_add(int64_t A, int64_t B);
int64_t mlx_lazy_mul(int64_t A, int64_t B);
int64_t mlx_lazy_sigmoid(int64_t A);
void mlx_eval_handles(int64_t* handles, int count);
void mlx_read_handle(int64_t handle, float* out, int size);
void mlx_free_handle(int64_t handle);
void mlx_free_handles(int64_t* handles, int count);

// Generic IR program and trainer APIs.
int64_t mlx_ir_program_create(int n_weights);
void mlx_ir_program_add_op(
    int64_t prog, int op_type,
    const char** inputs, int n_inputs,
    const char** outputs, int n_outputs,
    const float* float_params, int n_float_params,
    const int* int_params, int n_int_params);
void mlx_ir_program_destroy(int64_t prog);

typedef struct {
    const char* name;
    int dtype; // 0=int32, 1=float32
    const int* shape;
    int ndim;
    const void* data;
    int size_bytes;
} mlx_tensor_input;

typedef struct {
    int kind; // 0=adamw, 1=muon
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    int backend_steps;
    int nesterov;
} mlx_ir_optimizer_group;

typedef struct {
    int group_index;
    int decay;
} mlx_ir_weight_optimizer;

int mlx_ir_eval_program_output_size(
    int64_t program,
    int64_t* weight_handles,
    int n_weights,
    const int* tokens,
    const int* targets,
    int B,
    int T);
int mlx_ir_eval_program_output_size_named(
    int64_t program,
    int64_t* weight_handles,
    int n_weights,
    const mlx_tensor_input* inputs,
    int n_inputs);
int mlx_ir_eval_program_output_size_named_for_output(
    int64_t program,
    int64_t* weight_handles,
    int n_weights,
    const mlx_tensor_input* inputs,
    int n_inputs,
    const char* output_name);
int mlx_ir_eval_program(
    int64_t program,
    int64_t* weight_handles,
    int n_weights,
    const int* tokens,
    const int* targets,
    int B,
    int T,
    float* out,
    int out_size);
int mlx_ir_eval_program_named(
    int64_t program,
    int64_t* weight_handles,
    int n_weights,
    const mlx_tensor_input* inputs,
    int n_inputs,
    float* out,
    int out_size);
int mlx_ir_eval_program_named_for_output(
    int64_t program,
    int64_t* weight_handles,
    int n_weights,
    const mlx_tensor_input* inputs,
    int n_inputs,
    const char* output_name,
    float* out,
    int out_size);

int64_t mlx_ir_create_trainer(
    int64_t program,
    int64_t* weight_handles, int n_weights,
    const int* decay_flags,
    float lr, float beta1, float beta2, float eps, float wd, float max_grad_norm);
int64_t mlx_ir_create_trainer_v2(
    int64_t program,
    int64_t* weight_handles,
    int n_weights,
    const mlx_ir_weight_optimizer* weight_optimizers,
    int n_weight_optimizers,
    const mlx_ir_optimizer_group* optimizer_groups,
    int n_optimizer_groups,
    float max_grad_norm,
    float default_base_lr);
float mlx_ir_trainer_step(int64_t trainer, const int* tokens, const int* targets, int B, int T);
float mlx_ir_trainer_step_named(
    int64_t trainer,
    const mlx_tensor_input* inputs,
    int n_inputs);
void mlx_ir_trainer_submit_step(
    int64_t trainer,
    const mlx_tensor_input* inputs,
    int n_inputs);
float mlx_ir_trainer_collect_loss(int64_t trainer);
void mlx_ir_trainer_flush(int64_t trainer);
float mlx_ir_trainer_evaluate_named(
    int64_t trainer,
    const mlx_tensor_input* inputs,
    int n_inputs);
int mlx_ir_trainer_evaluate_per_token(
    int64_t trainer,
    const mlx_tensor_input* inputs,
    int n_inputs,
    float* out_nlls,
    int max_nlls,
    int* actual_nlls);
float mlx_ir_trainer_evaluate_lora_named(
    int64_t trainer,
    const mlx_tensor_input* inputs,
    int n_inputs,
    int rank,
    int steps,
    float lr);
int mlx_ir_trainer_read_output(
    int64_t trainer,
    const char* output_name,
    float* out,
    int out_size);
float mlx_ir_trainer_evaluate(int64_t trainer, const int* tokens, const int* targets, int B, int T);
int mlx_ir_trainer_num_weights(int64_t trainer);
int mlx_ir_trainer_weight_size(int64_t trainer, int weight_idx);
int mlx_ir_trainer_read_weight(int64_t trainer, int weight_idx, float* out, int size);
int mlx_ir_trainer_set_weight(int64_t trainer, int weight_idx, const float* data, int size);
void mlx_ir_trainer_set_lr(int64_t trainer, float lr);
void mlx_ir_trainer_set_lr_scale(int64_t trainer, float lr_scale);
void mlx_ir_trainer_set_qat(int64_t trainer, const char* mode);
void mlx_ir_trainer_destroy(int64_t trainer);

// Cleanup
void mlx_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif
