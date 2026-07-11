//go:build (!mlx && cgo && darwin) || (!mlx && cgo && linux)

// Package gpu provides the MLX Metal backend for mixlab's IR execution.
// This stub is used when the mlx build tag is absent but CGO is available.
// It imports "C" so that Go considers this a CGO package and allows
// bridge.cpp to coexist in the directory. The CGO flags are needed so
// bridge.cpp can find the MLX headers even though the stub doesn't use them.
//
// CGO path override: see mlx.go for instructions on overriding the default
// Homebrew MLX paths via CGO_CFLAGS/CGO_LDFLAGS environment variables.
package gpu

/*
#cgo CFLAGS: -I.
#cgo CXXFLAGS: -std=c++20 -I.
#cgo darwin CFLAGS: -I/opt/homebrew/lib/python3.11/site-packages/mlx/include
#cgo darwin CXXFLAGS: -I/opt/homebrew/lib/python3.11/site-packages/mlx/include -I/opt/homebrew/lib/python3.11/site-packages/mlx/include/metal_cpp
#cgo darwin LDFLAGS: -L/opt/homebrew/lib/python3.11/site-packages/mlx/lib -Wl,-rpath,/opt/homebrew/lib/python3.11/site-packages/mlx/lib -lmlx -framework Metal -framework Foundation -framework Accelerate
*/
import "C"

import "fmt"

var errNotBuilt = fmt.Errorf("MLX backend not built; rebuild with -tags mlx")

func SetCUDAGraphLimits(maxOps, maxMB int) {
	setCUDAGraphLimitEnv(maxOps, maxMB)
}

func Available() bool    { return false }
func DeviceName() string { return "" }

func mlxMemoryStats() MemoryStats { return MemoryStats{} }

func mlxClearMemoryCache() {}

func mlxSetMemoryLimit(bytes uint64) uint64 { return 0 }

func mlxSetMemoryCacheLimit(bytes uint64) uint64 { return 0 }

func FromData(data []float32, rows, cols int) (int64, error) {
	return 0, errNotBuilt
}

func FromDataShape(data []float32, shape []int) (int64, error) {
	return 0, errNotBuilt
}

func FreeHandle(handle int64)     {}
func FreeHandles(handles []int64) {}

func NewProgram(nWeights int) (*Program, error) {
	return nil, errNotBuilt
}

func (p *Program) Destroy() {}

func (p *Program) DeclareInput(name string, dtype int, shape []int) error {
	return errNotBuilt
}

func (p *Program) DeclareOutput(name string, dtype int, shape []int) error {
	return errNotBuilt
}

func (p *Program) AddOp(opType int, inputs, outputs []string, floatParams []float32, intParams []int) error {
	return errNotBuilt
}

func CreateTrainer(program *Program, weightHandles []int64, spec TrainerOptimizerSpec) (TrainerHandle, error) {
	return 0, errNotBuilt
}

func TrainerSetProgram(t TrainerHandle, program *Program) error {
	return errNotBuilt
}

func TrainerStep(t TrainerHandle, inputs []TensorInput) (float32, error) {
	return 0, errNotBuilt
}

func TrainerSubmitStep(t TrainerHandle, inputs []TensorInput) error {
	return errNotBuilt
}

func TrainerCollectLoss(t TrainerHandle) (float32, error) {
	return 0, errNotBuilt
}

func TrainerFlush(t TrainerHandle) error {
	return errNotBuilt
}

func TrainerEvaluate(t TrainerHandle, inputs []TensorInput) (float32, error) {
	return 0, errNotBuilt
}

func TrainerEvaluateWithOutputs(t TrainerHandle, inputs []TensorInput, outputNames []string) (float32, error) {
	return 0, errNotBuilt
}

func TrainerComputeMeanSquareGrads(t TrainerHandle, inputs []TensorInput, outputName string) (float32, error) {
	return 0, errNotBuilt
}

func TrainerEvaluatePerToken(t TrainerHandle, inputs []TensorInput) ([]float32, error) {
	return nil, errNotBuilt
}

func TrainerEvaluateLoRA(t TrainerHandle, inputs []TensorInput, rank, steps int, lr float32) (float32, error) {
	return 0, errNotBuilt
}

func TrainerSetLRScale(t TrainerHandle, lrScale float32) {}

func TrainerSetQAT(t TrainerHandle, mode string) error { return errNotBuilt }

func TrainerDestroy(t TrainerHandle) {}

func TrainerNumWeights(t TrainerHandle) (int, error) {
	return 0, errNotBuilt
}

func TrainerWeightSize(t TrainerHandle, weightIdx int) (int, error) {
	return 0, errNotBuilt
}

func TrainerReadWeight(t TrainerHandle, weightIdx int, out []float32) error {
	return errNotBuilt
}

func TrainerReadGrad(t TrainerHandle, weightIdx int, out []float32) error {
	return errNotBuilt
}

func TrainerSetWeight(t TrainerHandle, weightIdx int, data []float32) error {
	return errNotBuilt
}

func TrainerReadOutput(t TrainerHandle, name string, shape []int) ([]float32, error) {
	return nil, errNotBuilt
}

func TrainerSetStepOutputNames(t TrainerHandle, outputNames []string) error { return errNotBuilt }

func TrainerReadCachedOutput(t TrainerHandle, name string, shape []int) ([]float32, error) {
	return nil, errNotBuilt
}

func EvalProgramOutput(program *Program, weightHandles []int64, inputs []TensorInput, outputName string) ([]float32, error) {
	return nil, errNotBuilt
}

func EvalProgramOutputs(program *Program, weightHandles []int64, inputs []TensorInput, outputNames []string, outputSizes []int) (map[string][]float32, error) {
	return nil, errNotBuilt
}

func TrainerSampleCategoricalOutput(t TrainerHandle, inputs []TensorInput, outputName string, rows, vocab int, temperature float32, seed uint64) ([]int32, error) {
	return nil, errNotBuilt
}

func TrainerSampleCategoricalOutputEager(t TrainerHandle, inputs []TensorInput, outputName string, rows, vocab int, temperature float32, seed uint64) ([]int32, error) {
	return nil, errNotBuilt
}

func TrainerCompileStatsSnapshot(t TrainerHandle) (TrainerCompileStats, error) {
	return TrainerCompileStats{}, errNotBuilt
}

func TrainerOptimizerStatsSnapshot(t TrainerHandle) (TrainerOptimizerStats, error) {
	return TrainerOptimizerStats{}, errNotBuilt
}

func EvalProgramGradientsForOutput(program *Program, weightHandles []int64, inputs []TensorInput, outputName string) (float32, [][]float32, error) {
	return 0, nil, errNotBuilt
}
