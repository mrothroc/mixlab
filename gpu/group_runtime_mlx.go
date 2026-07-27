//go:build mlx && cgo && (darwin || linux)

package gpu

/*
#cgo CFLAGS: -I.
#cgo CXXFLAGS: -std=c++20 -I.
#cgo darwin CFLAGS: -I/opt/homebrew/opt/mlx/include
#cgo darwin CXXFLAGS: -I/opt/homebrew/opt/mlx/include -I/opt/homebrew/opt/mlx/include/metal_cpp
#cgo darwin LDFLAGS: -L/opt/homebrew/opt/mlx/lib -Wl,-rpath,/opt/homebrew/opt/mlx/lib -lmlx -framework Metal -framework Foundation -framework Accelerate
#cgo linux CFLAGS: -I/usr/local/include -I/usr/local/cuda/include
#cgo linux CXXFLAGS: -I/usr/local/include -I/usr/local/cuda/include
#cgo linux LDFLAGS: -L/usr/local/lib -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/stubs -Wl,-rpath,/usr/local/lib -Wl,-rpath,/usr/local/cuda/lib64 -lmlx -lopenblas -llapack -lcublas -lcublasLt -lcudart -lcudnn -lcufft -lcuda -lnvrtc -lnccl -lstdc++ -lm

#include <stdlib.h>
#include "mlx_bridge.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

func mlxGroupRuntimeCreate(backend string, strict bool) int64 {
	cBackend := C.CString(backend)
	defer C.free(unsafe.Pointer(cBackend))
	cStrict := C.int(0)
	if strict {
		cStrict = 1
	}
	return int64(C.mlx_group_runtime_create(cBackend, cStrict))
}

func mlxGroupRuntimeRank(handle int64) int {
	return int(C.mlx_group_runtime_rank(C.int64_t(handle)))
}

func mlxGroupRuntimeWorldSize(handle int64) int {
	return int(C.mlx_group_runtime_world_size(C.int64_t(handle)))
}

func mlxGroupRuntimeValidateIdentity(
	handle int64,
	generation uint64,
	membershipDigest [8]uint32,
	expectedMemberDigests []uint32,
	localMemberDigest [8]uint32,
) int {
	return int(C.mlx_group_runtime_validate_identity(
		C.int64_t(handle),
		C.uint64_t(generation),
		(*C.uint32_t)(unsafe.Pointer(&membershipDigest[0])),
		(*C.uint32_t)(unsafe.Pointer(&expectedMemberDigests[0])),
		C.int(len(expectedMemberDigests)/8),
		(*C.uint32_t)(unsafe.Pointer(&localMemberDigest[0])),
	))
}

func mlxGroupRuntimeDestroy(handle int64) {
	C.mlx_group_runtime_destroy(C.int64_t(handle))
}

func (r *GroupRuntime) attachTrainer(trainer TrainerHandle) error {
	if r == nil {
		return fmt.Errorf("distributed group runtime is nil")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.handle == 0 {
		return fmt.Errorf("distributed group runtime is closed")
	}
	return mlxTrainerSetGroupRuntime(trainer, r.handle)
}

func mlxTrainerSetGroupRuntime(trainer TrainerHandle, groupRuntime int64) error {
	if C.mlx_ir_trainer_set_group_runtime(
		C.int64_t(trainer),
		C.int64_t(groupRuntime),
	) != 0 {
		return fmt.Errorf("mlx_ir_trainer_set_group_runtime failed")
	}
	return nil
}

func mlxTrainerSetNextLossNormalizer(trainer TrainerHandle, lossNormalizer float32) error {
	if C.mlx_ir_trainer_set_next_loss_normalizer(
		C.int64_t(trainer),
		C.float(lossNormalizer),
	) != 0 {
		return fmt.Errorf("mlx_ir_trainer_set_next_loss_normalizer failed")
	}
	return nil
}

func mlxTrainerLastStageTrace(trainer TrainerHandle) ([]string, error) {
	required := int(C.mlx_ir_trainer_last_stage_trace(C.int64_t(trainer), nil, 0))
	if required <= 0 {
		return nil, fmt.Errorf("mlx_ir_trainer_last_stage_trace failed")
	}
	buf := make([]byte, required)
	if C.mlx_ir_trainer_last_stage_trace(
		C.int64_t(trainer),
		(*C.char)(unsafe.Pointer(&buf[0])),
		C.int(len(buf)),
	) != C.int(required) {
		return nil, fmt.Errorf("mlx_ir_trainer_last_stage_trace read failed")
	}
	return splitStageTrace(string(buf[:len(buf)-1])), nil
}
