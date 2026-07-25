//go:build mlx && cgo && (darwin || linux)

package train

import (
	"fmt"
	"os"

	ir "github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/gpu"
)

func startMamba3MetalPrewarm(prog *ir.Program) (bool, error) {
	if !irProgramHasCanonicalMamba3(prog) ||
		os.Getenv("MIXLAB_MAMBA3_DISABLE_METAL_PREWARM") == "1" {
		return false, nil
	}
	if err := gpu.StartMamba3MetalPrewarm(); err != nil {
		return false, fmt.Errorf("start Mamba3 Metal prewarm: %w", err)
	}
	return true, nil
}

func irProgramHasCanonicalMamba3(prog *ir.Program) bool {
	if prog == nil {
		return false
	}
	for _, op := range prog.Ops {
		if op.Code == ir.OpMamba3SelectiveScan ||
			op.Code == ir.OpMamba3CanonicalBlock {
			return true
		}
	}
	return false
}
