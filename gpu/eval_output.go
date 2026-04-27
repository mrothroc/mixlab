//go:build mlx && cgo && darwin

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
