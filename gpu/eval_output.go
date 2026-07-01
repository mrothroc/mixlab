//go:build mlx && cgo && (darwin || linux)

package gpu

/*
#include <stdlib.h>
#include "mlx_bridge.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

func evalProgramOutput(program *Program, weightHandles []int64, inputs []TensorInput, outputName string) ([]float32, error) {
	cInputs, cleanup, err := marshalTensorInputs(inputs)
	if err != nil {
		return nil, err
	}
	defer cleanup()

	cWeights := make([]C.int64_t, len(weightHandles))
	for i, handle := range weightHandles {
		cWeights[i] = C.int64_t(handle)
	}
	cName := C.CString(outputName)
	defer C.free(unsafe.Pointer(cName))

	size := int(C.mlx_ir_eval_program_output_size_named_for_output(
		C.int64_t(program.handle),
		(*C.int64_t)(unsafe.Pointer(&cWeights[0])),
		C.int(len(cWeights)),
		(*C.mlx_tensor_input)(unsafe.Pointer(&cInputs[0])),
		C.int(len(cInputs)),
		cName,
	))
	if size <= 0 {
		return nil, fmt.Errorf("mlx_ir_eval_program_output_size_named_for_output failed for %q", outputName)
	}

	out := make([]float32, size)
	rc := int(C.mlx_ir_eval_program_named_for_output(
		C.int64_t(program.handle),
		(*C.int64_t)(unsafe.Pointer(&cWeights[0])),
		C.int(len(cWeights)),
		(*C.mlx_tensor_input)(unsafe.Pointer(&cInputs[0])),
		C.int(len(cInputs)),
		cName,
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(len(out)),
	))
	if rc != 0 {
		return nil, fmt.Errorf("mlx_ir_eval_program_named_for_output failed for %q", outputName)
	}
	return out, nil
}

func evalProgramOutputs(program *Program, weightHandles []int64, inputs []TensorInput, outputNames []string, outputSizes []int) (map[string][]float32, error) {
	cInputs, cleanup, err := marshalTensorInputs(inputs)
	if err != nil {
		return nil, err
	}
	defer cleanup()

	cWeights := make([]C.int64_t, len(weightHandles))
	for i, handle := range weightHandles {
		cWeights[i] = C.int64_t(handle)
	}
	nameArray := C.malloc(C.size_t(len(outputNames)) * C.size_t(unsafe.Sizeof(uintptr(0))))
	outArray := C.malloc(C.size_t(len(outputNames)) * C.size_t(unsafe.Sizeof(uintptr(0))))
	sizeArray := C.malloc(C.size_t(len(outputNames)) * C.size_t(unsafe.Sizeof(C.int(0))))
	if nameArray == nil || outArray == nil || sizeArray == nil {
		if nameArray != nil {
			C.free(nameArray)
		}
		if outArray != nil {
			C.free(outArray)
		}
		if sizeArray != nil {
			C.free(sizeArray)
		}
		return nil, fmt.Errorf("allocating output arrays for mlx_ir_eval_program_named_outputs failed")
	}
	defer C.free(nameArray)
	defer C.free(outArray)
	defer C.free(sizeArray)

	cNames := unsafe.Slice((**C.char)(nameArray), len(outputNames))
	cOuts := unsafe.Slice((**C.float)(outArray), len(outputNames))
	cSizes := unsafe.Slice((*C.int)(sizeArray), len(outputNames))
	defer func() {
		for _, name := range cNames {
			if name != nil {
				C.free(unsafe.Pointer(name))
			}
		}
		for _, out := range cOuts {
			if out != nil {
				C.free(unsafe.Pointer(out))
			}
		}
	}()
	for i, name := range outputNames {
		cNames[i] = C.CString(name)
		cOuts[i] = (*C.float)(C.malloc(C.size_t(outputSizes[i]) * C.size_t(unsafe.Sizeof(C.float(0)))))
		if cOuts[i] == nil {
			return nil, fmt.Errorf("allocating output buffer for %q failed", name)
		}
		cSizes[i] = C.int(outputSizes[i])
	}
	rc := int(C.mlx_ir_eval_program_named_outputs(
		C.int64_t(program.handle),
		(*C.int64_t)(unsafe.Pointer(&cWeights[0])),
		C.int(len(cWeights)),
		(*C.mlx_tensor_input)(unsafe.Pointer(&cInputs[0])),
		C.int(len(cInputs)),
		(**C.char)(nameArray),
		C.int(len(cNames)),
		(**C.float)(outArray),
		(*C.int)(sizeArray),
	))
	if rc != 0 {
		return nil, fmt.Errorf("mlx_ir_eval_program_named_outputs failed for %v", outputNames)
	}
	out := make(map[string][]float32, len(outputNames))
	for i, name := range outputNames {
		values := make([]float32, outputSizes[i])
		cValues := unsafe.Slice(cOuts[i], outputSizes[i])
		for j, v := range cValues {
			values[j] = float32(v)
		}
		out[name] = values
	}
	return out, nil
}
