//go:build mlx && cgo && (darwin || linux)

package gpu

import "fmt"

// SetCUDAGraphLimits configures MLX's CUDA graph batch size.
// Must be called before Available() or any other GPU function.
// Values <= 0 are ignored.
func SetCUDAGraphLimits(maxOps, maxMB int) {
	mlxSetCUDAGraphLimits(maxOps, maxMB)
}

func Available() bool {
	return mlxInit() == 0
}

func DeviceName() string {
	return mlxDeviceName()
}

func FromData(data []float32, rows, cols int) (int64, error) {
	if rows <= 0 || cols <= 0 {
		return 0, fmt.Errorf("invalid tensor shape rows=%d cols=%d; pass positive dimensions to gpu.FromData", rows, cols)
	}
	if len(data) < rows*cols {
		return 0, fmt.Errorf("tensor data too small: have=%d need=%d float32 values; provide one element per matrix entry", len(data), rows*cols)
	}
	h := mlxFromData(data, rows, cols)
	if h == 0 {
		return 0, fmt.Errorf("mlx_from_data failed; verify the MLX runtime is installed and the process can access the selected device")
	}
	setHandleSize(h, rows*cols)
	return h, nil
}

func FreeHandle(handle int64) {
	if handle == 0 {
		return
	}
	clearHandleSize(handle)
	mlxFreeHandle(handle)
}

func FreeHandles(handles []int64) {
	if len(handles) == 0 {
		return
	}
	for _, handle := range handles {
		clearHandleSize(handle)
	}
	mlxFreeHandles(handles)
}

func NewProgram(nWeights int) (*Program, error) {
	if nWeights <= 0 {
		return nil, fmt.Errorf("invalid nWeights=%d; create GPU programs with at least one trainable weight", nWeights)
	}
	h := mlxCreateProgram(nWeights)
	if h == 0 {
		return nil, fmt.Errorf("mlx_ir_program_create failed; rebuild with `-tags mlx` and verify the MLX native libraries are available")
	}
	return &Program{
		handle:      h,
		nWeights:    nWeights,
		inputDecls:  make(map[string]TensorInput),
		outputDecls: make(map[string]TensorInput),
	}, nil
}

func (p *Program) Destroy() {
	if p == nil || p.handle == 0 {
		return
	}
	mlxDestroyProgram(p.handle)
	p.handle = 0
}

func (p *Program) AddOp(opType int, inputs, outputs []string, floatParams []float32, intParams []int) error {
	if p == nil || p.handle == 0 {
		return fmt.Errorf("invalid GPU program handle; create the program with gpu.NewProgram before declaring ops")
	}
	if len(inputs) > 4 || len(outputs) > 2 || len(floatParams) > 4 || len(intParams) > 8 {
		return fmt.Errorf("IR op params exceed bridge limits; keep inputs<=4 outputs<=2 float_params<=4 int_params<=8")
	}
	if err := mlxAddOp(p.handle, opType, inputs, outputs, floatParams, intParams); err != nil {
		return err
	}
	p.opCount++
	p.opTypes = append(p.opTypes, opType)
	return nil
}

func CreateTrainer(program *Program, weightHandles []int64, spec TrainerOptimizerSpec) (TrainerHandle, error) {
	if program == nil || program.handle == 0 {
		return 0, fmt.Errorf("invalid GPU program; create and populate a gpu.Program before creating a trainer")
	}
	if len(weightHandles) == 0 {
		return 0, fmt.Errorf("no weight handles; upload weights with gpu.FromData before creating a trainer")
	}
	if len(weightHandles) != len(spec.Weights) {
		return 0, fmt.Errorf("weight optimizer mismatch: weights=%d specs=%d", len(weightHandles), len(spec.Weights))
	}
	if program.nWeights != len(weightHandles) {
		return 0, fmt.Errorf("program weight mismatch: program=%d weights=%d", program.nWeights, len(weightHandles))
	}
	if len(spec.Groups) == 0 {
		return 0, fmt.Errorf("no optimizer groups; define at least one optimizer group in TrainerOptimizerSpec")
	}
	return mlxCreateTrainer(program.handle, weightHandles, spec)
}

func createLegacyAdamTrainer(
	program *Program,
	weightHandles []int64,
	decayFlags []bool,
	lr, beta1, beta2, eps, wd, maxGradNorm float32,
) (TrainerHandle, error) {
	if program == nil || program.handle == 0 {
		return 0, fmt.Errorf("invalid program")
	}
	if len(weightHandles) == 0 {
		return 0, fmt.Errorf("no weight handles")
	}
	if len(weightHandles) != len(decayFlags) {
		return 0, fmt.Errorf("decay flag mismatch: weights=%d flags=%d", len(weightHandles), len(decayFlags))
	}
	if program.nWeights != len(weightHandles) {
		return 0, fmt.Errorf("program weight mismatch: program=%d weights=%d", program.nWeights, len(weightHandles))
	}
	return mlxCreateLegacyAdamTrainer(program.handle, weightHandles, decayFlags, lr, beta1, beta2, eps, wd, maxGradNorm)
}

func TrainerStep(t TrainerHandle, inputs []TensorInput) (float32, error) {
	if t == 0 {
		return 0, fmt.Errorf("invalid trainer handle; create the trainer successfully before running a step")
	}
	return mlxTrainerStep(t, inputs)
}

func TrainerSubmitStep(t TrainerHandle, inputs []TensorInput) error {
	if t == 0 {
		return fmt.Errorf("invalid trainer handle; create the trainer successfully before submitting a step")
	}
	return mlxTrainerSubmitStep(t, inputs)
}

func TrainerCollectLoss(t TrainerHandle) (float32, error) {
	if t == 0 {
		return 0, fmt.Errorf("invalid trainer handle; create the trainer successfully before collecting a loss")
	}
	return mlxTrainerCollectLoss(t)
}

func TrainerFlush(t TrainerHandle) error {
	if t == 0 {
		return fmt.Errorf("invalid trainer handle; create the trainer successfully before flushing pending work")
	}
	return mlxTrainerFlush(t)
}

func TrainerEvaluate(t TrainerHandle, inputs []TensorInput) (float32, error) {
	if t == 0 {
		return 0, fmt.Errorf("invalid trainer handle; create the trainer successfully before running evaluation")
	}
	return mlxTrainerEvaluate(t, inputs)
}

func TrainerEvaluateLoRA(t TrainerHandle, inputs []TensorInput, rank, steps int, lr float32) (float32, error) {
	if t == 0 {
		return 0, fmt.Errorf("invalid trainer handle; create the trainer successfully before running LoRA TTT evaluation")
	}
	if rank <= 0 {
		return 0, fmt.Errorf("invalid LoRA rank %d; pass a positive rank", rank)
	}
	if steps < 0 {
		return 0, fmt.Errorf("invalid LoRA step count %d; pass a non-negative step count", steps)
	}
	return mlxTrainerEvaluateLoRA(t, inputs, rank, steps, lr)
}

func TrainerSetLRScale(t TrainerHandle, lrScale float32) {
	if t == 0 {
		return
	}
	mlxTrainerSetLRScale(t, lrScale)
}

func TrainerSetQAT(t TrainerHandle, mode string) error {
	if t == 0 {
		return fmt.Errorf("invalid trainer handle; create the trainer successfully before configuring QAT")
	}
	switch mode {
	case "", "none", "int8", "int6":
	default:
		return fmt.Errorf("invalid QAT mode %q; expected \"none\", \"int8\", or \"int6\"", mode)
	}
	return mlxTrainerSetQAT(t, mode)
}

func TrainerDestroy(t TrainerHandle) {
	if t == 0 {
		return
	}
	mlxTrainerDestroy(t)
}

func TrainerNumWeights(t TrainerHandle) (int, error) {
	if t == 0 {
		return 0, fmt.Errorf("invalid trainer handle; create the trainer before querying weights")
	}
	n := mlxTrainerNumWeights(t)
	if n < 0 {
		return 0, fmt.Errorf("mlx_ir_trainer_num_weights failed")
	}
	return n, nil
}

func TrainerWeightSize(t TrainerHandle, weightIdx int) (int, error) {
	if t == 0 {
		return 0, fmt.Errorf("invalid trainer handle; create the trainer before querying weight sizes")
	}
	if weightIdx < 0 {
		return 0, fmt.Errorf("invalid weight index %d; pass a non-negative index returned by TrainerNumWeights", weightIdx)
	}
	sz := mlxTrainerWeightSize(t, weightIdx)
	if sz < 0 {
		return 0, fmt.Errorf("mlx_ir_trainer_weight_size failed")
	}
	return sz, nil
}

func TrainerReadWeight(t TrainerHandle, weightIdx int, out []float32) error {
	if t == 0 {
		return fmt.Errorf("invalid trainer handle; create the trainer before reading weights")
	}
	if weightIdx < 0 {
		return fmt.Errorf("invalid weight index %d; pass a non-negative index returned by TrainerNumWeights", weightIdx)
	}
	if len(out) == 0 {
		return fmt.Errorf("output buffer is empty; allocate a float32 slice sized by TrainerWeightSize before reading weights")
	}
	if mlxTrainerReadWeight(t, weightIdx, out) != 0 {
		return fmt.Errorf("mlx_ir_trainer_read_weight failed")
	}
	return nil
}

func TrainerReadOutput(t TrainerHandle, name string, shape []int) ([]float32, error) {
	if t == 0 {
		return nil, fmt.Errorf("invalid trainer handle; create the trainer before reading named outputs")
	}
	if name == "" {
		return nil, fmt.Errorf("output name is required; pass a declared program output such as \"loss\"")
	}
	elemCount := 1
	for i, dim := range shape {
		if dim <= 0 {
			return nil, fmt.Errorf("invalid output shape[%d]=%d", i, dim)
		}
		elemCount *= dim
	}
	out := make([]float32, elemCount)
	if err := mlxTrainerReadOutput(t, name, out); err != nil {
		return nil, err
	}
	return out, nil
}
