//go:build mlx && cgo && linux

// CGO path override: set CGO_CFLAGS, CGO_CXXFLAGS, and CGO_LDFLAGS if MLX is installed outside the default /usr/local prefix.
package gpu

/*
#cgo CFLAGS: -I.
#cgo CXXFLAGS: -std=c++20 -I.
#cgo linux CFLAGS: -I/usr/local/include
#cgo linux CXXFLAGS: -I/usr/local/include
#cgo linux LDFLAGS: -L/usr/local/lib -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/stubs -Wl,-rpath,/usr/local/lib -Wl,-rpath,/usr/local/cuda/lib64 -lmlx -lopenblas -llapack -lcublas -lcublasLt -lcudart -lcudnn -lcufft -lcuda -lnvrtc -lstdc++ -lm

#include <stdlib.h>
#include "mlx_bridge.h"
*/
import "C"

import (
	"fmt"
	"math"
	"unsafe"
)

func mlxInit() int {
	return int(C.mlx_init())
}

func mlxDeviceName() string {
	return C.GoString(C.mlx_device_name())
}

func mlxFromData(data []float32, rows, cols int) int64 {
	h := C.mlx_from_data(
		(*C.float)(unsafe.Pointer(&data[0])),
		C.int(rows),
		C.int(cols),
	)
	return int64(h)
}

func mlxFreeHandle(handle int64) {
	C.mlx_free_handle(C.int64_t(handle))
}

func mlxFreeHandles(handles []int64) {
	cHandles := make([]C.int64_t, len(handles))
	for i := range handles {
		cHandles[i] = C.int64_t(handles[i])
	}
	C.mlx_free_handles((*C.int64_t)(unsafe.Pointer(&cHandles[0])), C.int(len(cHandles)))
}

func mlxCreateProgram(nWeights int) int64 {
	return int64(C.mlx_ir_program_create(C.int(nWeights)))
}

func mlxDestroyProgram(handle int64) {
	C.mlx_ir_program_destroy(C.int64_t(handle))
}

func (p *Program) DeclareInput(name string, dtype int, shape []int) error {
	if p == nil || p.handle == 0 {
		return fmt.Errorf("invalid IR program handle")
	}
	if err := validateTensorDecl(name, dtype, shape); err != nil {
		return err
	}
	p.inputDecls[name] = TensorInput{Name: name, DType: dtype, Shape: cloneShape(shape)}
	return nil
}

func (p *Program) DeclareOutput(name string, dtype int, shape []int) error {
	if p == nil || p.handle == 0 {
		return fmt.Errorf("invalid IR program handle")
	}
	if err := validateTensorDecl(name, dtype, shape); err != nil {
		return err
	}
	if _, exists := p.outputDecls[name]; !exists {
		p.outputOrder = append(p.outputOrder, name)
	}
	p.outputDecls[name] = TensorInput{Name: name, DType: dtype, Shape: cloneShape(shape)}
	return nil
}

func mlxAddOp(handle int64, opType int, inputs, outputs []string, floatParams []float32, intParams []int) error {
	cInputs := make([]*C.char, len(inputs))
	for i, in := range inputs {
		cInputs[i] = C.CString(in)
	}
	defer func() {
		for _, s := range cInputs {
			if s != nil {
				C.free(unsafe.Pointer(s))
			}
		}
	}()

	cOutputs := make([]*C.char, len(outputs))
	for i, out := range outputs {
		cOutputs[i] = C.CString(out)
	}
	defer func() {
		for _, s := range cOutputs {
			if s != nil {
				C.free(unsafe.Pointer(s))
			}
		}
	}()

	var cInputPtr **C.char
	if len(cInputs) > 0 {
		cInputPtr = (**C.char)(unsafe.Pointer(&cInputs[0]))
	}
	var cOutputPtr **C.char
	if len(cOutputs) > 0 {
		cOutputPtr = (**C.char)(unsafe.Pointer(&cOutputs[0]))
	}

	var cFloatPtr *C.float
	if len(floatParams) > 0 {
		cFloatPtr = (*C.float)(unsafe.Pointer(&floatParams[0]))
	}

	cInts := make([]C.int, len(intParams))
	for i, v := range intParams {
		cInts[i] = C.int(v)
	}
	var cIntPtr *C.int
	if len(cInts) > 0 {
		cIntPtr = (*C.int)(unsafe.Pointer(&cInts[0]))
	}

	C.mlx_ir_program_add_op(
		C.int64_t(handle),
		C.int(opType),
		cInputPtr, C.int(len(inputs)),
		cOutputPtr, C.int(len(outputs)),
		cFloatPtr, C.int(len(floatParams)),
		cIntPtr, C.int(len(intParams)),
	)
	return nil
}

func marshalTensorInputs(inputs []TensorInput) ([]C.mlx_tensor_input, func(), error) {
	if len(inputs) == 0 {
		return nil, nil, fmt.Errorf("no tensor inputs")
	}
	cInputs := make([]C.mlx_tensor_input, len(inputs))
	toFree := make([]unsafe.Pointer, 0, len(inputs)*3)
	cleanup := func() {
		for _, p := range toFree {
			C.free(p)
		}
	}
	for i, in := range inputs {
		if in.Name == "" {
			cleanup()
			return nil, nil, fmt.Errorf("tensor input[%d] missing name", i)
		}
		if len(in.Shape) == 0 {
			cleanup()
			return nil, nil, fmt.Errorf("tensor %q has empty shape", in.Name)
		}
		if in.DType != 0 && in.DType != 1 {
			cleanup()
			return nil, nil, fmt.Errorf("tensor %q has unsupported dtype=%d", in.Name, in.DType)
		}
		shape32 := make([]int32, len(in.Shape))
		shapeElemCount := 1
		for j, dim := range in.Shape {
			if dim <= 0 {
				cleanup()
				return nil, nil, fmt.Errorf("tensor %q has invalid shape dim=%d", in.Name, dim)
			}
			shape32[j] = int32(dim)
			shapeElemCount *= dim
		}

		var dataBytes []byte
		switch v := in.Data.(type) {
		case []int32:
			if in.DType != 0 {
				cleanup()
				return nil, nil, fmt.Errorf("tensor %q dtype/data mismatch", in.Name)
			}
			if len(v) != shapeElemCount {
				cleanup()
				return nil, nil, fmt.Errorf("tensor %q data length=%d shape_elems=%d", in.Name, len(v), shapeElemCount)
			}
			dataBytes = cBytesFromInt32(v)
		case []float32:
			if in.DType != 1 {
				cleanup()
				return nil, nil, fmt.Errorf("tensor %q dtype/data mismatch", in.Name)
			}
			if len(v) != shapeElemCount {
				cleanup()
				return nil, nil, fmt.Errorf("tensor %q data length=%d shape_elems=%d", in.Name, len(v), shapeElemCount)
			}
			dataBytes = cBytesFromFloat32(v)
		default:
			cleanup()
			return nil, nil, fmt.Errorf("tensor %q unsupported Data type %T", in.Name, in.Data)
		}
		if len(dataBytes) == 0 {
			cleanup()
			return nil, nil, fmt.Errorf("tensor %q has empty data", in.Name)
		}

		cName := C.CString(in.Name)
		cShape := C.CBytes(cBytesFromInt32(shape32))
		cData := C.CBytes(dataBytes)
		toFree = append(toFree, unsafe.Pointer(cName), cShape, cData)

		cInputs[i] = C.mlx_tensor_input{
			name:       cName,
			dtype:      C.int(in.DType),
			shape:      (*C.int)(cShape),
			ndim:       C.int(len(shape32)),
			data:       cData,
			size_bytes: C.int(len(dataBytes)),
		}
	}
	return cInputs, cleanup, nil
}

func mlxCreateTrainer(programHandle int64, weightHandles []int64, spec TrainerOptimizerSpec) (TrainerHandle, error) {
	cWeights := make([]C.int64_t, len(weightHandles))
	for i := range weightHandles {
		cWeights[i] = C.int64_t(weightHandles[i])
	}
	cWeightSpecs := make([]C.mlx_ir_weight_optimizer, len(spec.Weights))
	for i, weightSpec := range spec.Weights {
		if weightSpec.GroupIndex < 0 || weightSpec.GroupIndex >= len(spec.Groups) {
			return 0, fmt.Errorf("weight %d has invalid group index %d", i, weightSpec.GroupIndex)
		}
		cWeightSpecs[i] = C.mlx_ir_weight_optimizer{
			group_index: C.int(weightSpec.GroupIndex),
			decay:       0,
		}
		if weightSpec.Decay {
			cWeightSpecs[i].decay = 1
		}
	}
	cGroups := make([]C.mlx_ir_optimizer_group, len(spec.Groups))
	for i, group := range spec.Groups {
		if group.Kind != OptimizerAdamW && group.Kind != OptimizerMuon {
			return 0, fmt.Errorf("optimizer group %d has unsupported kind %d", i, group.Kind)
		}
		cGroups[i] = C.mlx_ir_optimizer_group{
			kind:          C.int(group.Kind),
			lr:            C.float(group.LR),
			beta1:         C.float(group.Beta1),
			beta2:         C.float(group.Beta2),
			eps:           C.float(group.Epsilon),
			weight_decay:  C.float(group.WeightDecay),
			backend_steps: C.int(group.BackendSteps),
			nesterov:      0,
		}
		if group.Nesterov {
			cGroups[i].nesterov = 1
		}
	}
	h := C.mlx_ir_create_trainer_v2(
		C.int64_t(programHandle),
		(*C.int64_t)(unsafe.Pointer(&cWeights[0])), C.int(len(cWeights)),
		(*C.mlx_ir_weight_optimizer)(unsafe.Pointer(&cWeightSpecs[0])), C.int(len(cWeightSpecs)),
		(*C.mlx_ir_optimizer_group)(unsafe.Pointer(&cGroups[0])), C.int(len(cGroups)),
		C.float(spec.MaxGradNorm), C.float(spec.DefaultBaseLR),
	)
	if h == 0 {
		return 0, fmt.Errorf("mlx_ir_create_trainer_v2 failed")
	}
	return TrainerHandle(h), nil
}

func mlxCreateLegacyAdamTrainer(
	programHandle int64,
	weightHandles []int64,
	decayFlags []bool,
	lr, beta1, beta2, eps, wd, maxGradNorm float32,
) (TrainerHandle, error) {
	cWeights := make([]C.int64_t, len(weightHandles))
	cDecay := make([]C.int, len(decayFlags))
	for i := range weightHandles {
		cWeights[i] = C.int64_t(weightHandles[i])
		if decayFlags[i] {
			cDecay[i] = 1
		}
	}
	h := C.mlx_ir_create_trainer(
		C.int64_t(programHandle),
		(*C.int64_t)(unsafe.Pointer(&cWeights[0])), C.int(len(cWeights)),
		(*C.int)(unsafe.Pointer(&cDecay[0])),
		C.float(lr), C.float(beta1), C.float(beta2), C.float(eps), C.float(wd), C.float(maxGradNorm),
	)
	if h == 0 {
		return 0, fmt.Errorf("mlx_ir_create_trainer failed")
	}
	return TrainerHandle(h), nil
}

func mlxTrainerStep(t TrainerHandle, inputs []TensorInput) (float32, error) {
	cInputs, cleanup, err := marshalTensorInputs(inputs)
	if err != nil {
		return 0, err
	}
	defer cleanup()
	loss := float32(C.mlx_ir_trainer_step_named(
		C.int64_t(t),
		(*C.mlx_tensor_input)(unsafe.Pointer(&cInputs[0])),
		C.int(len(cInputs)),
	))
	if math.IsNaN(float64(loss)) {
		return 0, fmt.Errorf("mlx_ir_trainer_step_named failed")
	}
	return loss, nil
}

func mlxTrainerEvaluate(t TrainerHandle, inputs []TensorInput) (float32, error) {
	cInputs, cleanup, err := marshalTensorInputs(inputs)
	if err != nil {
		return 0, err
	}
	defer cleanup()
	loss := float32(C.mlx_ir_trainer_evaluate_named(
		C.int64_t(t),
		(*C.mlx_tensor_input)(unsafe.Pointer(&cInputs[0])),
		C.int(len(cInputs)),
	))
	if math.IsNaN(float64(loss)) {
		return 0, fmt.Errorf("mlx_ir_trainer_evaluate_named failed")
	}
	return loss, nil
}

func mlxTrainerSetLRScale(t TrainerHandle, lrScale float32) {
	C.mlx_ir_trainer_set_lr_scale(C.int64_t(t), C.float(lrScale))
}

func mlxTrainerDestroy(t TrainerHandle) {
	C.mlx_ir_trainer_destroy(C.int64_t(t))
}

func mlxTrainerNumWeights(t TrainerHandle) int {
	return int(C.mlx_ir_trainer_num_weights(C.int64_t(t)))
}

func mlxTrainerWeightSize(t TrainerHandle, weightIdx int) int {
	return int(C.mlx_ir_trainer_weight_size(C.int64_t(t), C.int(weightIdx)))
}

func mlxTrainerReadWeight(t TrainerHandle, weightIdx int, out []float32) int {
	return int(C.mlx_ir_trainer_read_weight(
		C.int64_t(t),
		C.int(weightIdx),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(len(out)),
	))
}

func mlxTrainerReadOutput(t TrainerHandle, name string, out []float32) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	rc := int(C.mlx_ir_trainer_read_output(
		C.int64_t(t),
		cName,
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(len(out)),
	))
	if rc != 0 {
		return fmt.Errorf("mlx_ir_trainer_read_output failed for %q", name)
	}
	return nil
}
