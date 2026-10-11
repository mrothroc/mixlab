//go:build mlx && cgo && (darwin || linux)

package gpu

/*
#include "mlx_bridge.h"
*/
import "C"

import (
	"fmt"
	"os"
	"runtime"
)

func PrecisionBackend() string {
	if runtime.GOOS == "linux" {
		return "cuda"
	}
	return "metal"
}

// ConfigurePrecision must precede GPU work. The native bridge serializes it
// against initialization and freezes the setting for the process lifetime.
func ConfigurePrecision(dtype string, requested *bool) (PrecisionPolicy, error) {
	p, err := ResolvePrecisionPolicy(dtype, PrecisionBackend(), requested, os.LookupEnv)
	if err != nil {
		return p, err
	}
	if p.Backend == "cuda" || p.Backend == "metal" {
		v := 0
		if p.TF32Allowed {
			v = 1
		}
		if C.mlx_configure_tf32(C.int(v)) != 0 {
			return p, fmt.Errorf("GPU precision policy differs from the initialized MLX policy; start a fresh process to change TF32")
		}
	}
	return p, nil
}
