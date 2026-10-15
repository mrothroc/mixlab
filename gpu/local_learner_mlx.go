//go:build mlx && cgo && (darwin || linux)

package gpu

/*
#include "mlx_bridge.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// TrainerSetLocalAccumulation uses the shared staged optimizer with no MLX group.
func TrainerSetLocalAccumulation(t TrainerHandle, steps int) error {
	if steps < 1 || steps > 1<<20 || C.mlx_ir_trainer_set_local_accumulation(C.int64_t(t), C.int(steps)) != 0 {
		return fmt.Errorf("configure local accumulation failed")
	}
	return nil
}

// TrainerInstallWeights atomically replaces all model tensors, preserving moments.
func TrainerInstallWeights(t TrainerHandle, handles []int64) error {
	if len(handles) == 0 {
		return fmt.Errorf("empty global model")
	}
	if C.mlx_ir_trainer_install_weights(C.int64_t(t), (*C.int64_t)(unsafe.Pointer(&handles[0])), C.int(len(handles))) != 0 {
		return fmt.Errorf("atomic global model install failed")
	}
	return nil
}
