//go:build mlx && cgo && (darwin || linux)

package train

import (
	"fmt"

	ir "github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/gpu"
)

// lowerIRToGPU converts a mixlab ir.Program to a gpu.Program.
func lowerIRToGPU(prog *ir.Program) (*gpu.Program, error) {
	gpuProg, err := gpu.NewProgram(prog.NumWeights)
	if err != nil {
		return nil, err
	}

	// Declare inputs
	for _, inp := range prog.Inputs {
		dtype := gpu.TensorInt32
		if inp.DType == ir.TensorFloat32 {
			dtype = gpu.TensorFloat32
		}
		if err := gpuProg.DeclareInput(inp.Name, dtype, inp.Shape); err != nil {
			gpuProg.Destroy()
			return nil, fmt.Errorf("declare input %q: %w", inp.Name, err)
		}
	}

	// Emit ops
	for _, op := range prog.Ops {
		if err := gpuProg.AddOp(op.Code, op.Inputs, op.Outputs, op.FloatParams, op.IntParams); err != nil {
			gpuProg.Destroy()
			return nil, fmt.Errorf("add op code=%d: %w", op.Code, err)
		}
	}

	// Declare outputs
	for _, out := range prog.Outputs {
		dtype := gpu.TensorInt32
		if out.DType == ir.TensorFloat32 {
			dtype = gpu.TensorFloat32
		}
		if err := gpuProg.DeclareOutput(out.Name, dtype, out.Shape); err != nil {
			gpuProg.Destroy()
			return nil, fmt.Errorf("declare output %q: %w", out.Name, err)
		}
	}

	return gpuProg, nil
}

func preferredEvalLossOutputName(prog *ir.Program) string {
	if prog != nil {
		for _, out := range prog.Outputs {
			if out.Name == "generation_eval_loss" {
				return "generation_eval_loss"
			}
		}
		for _, out := range prog.Outputs {
			if out.Name == "eval_loss" {
				return "eval_loss"
			}
		}
	}
	return "loss"
}

func programDeclaresInput(prog *ir.Program, name string) bool {
	if prog == nil {
		return false
	}
	for _, in := range prog.Inputs {
		if in.Name == name {
			return true
		}
	}
	return false
}
