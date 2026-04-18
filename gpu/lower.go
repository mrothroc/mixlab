// Package gpu provides the MLX Metal backend for mixlab's IR execution.
// This file lowers an ir.Program (pure Go) to a gpu.Program (C++ handle).
package gpu

import (
	"fmt"

	ir "github.com/mrothroc/mixlab/arch"
)

// irToGPUOp maps ir.Op* codes to gpu.Op* codes.
// They are intentionally aligned, but this mapping makes the contract explicit.
var irToGPUOp = map[int]int{
	ir.OpEmbed:        OpEmbed,
	ir.OpMatMul:       OpMatMul,
	ir.OpAdd:          OpAdd,
	ir.OpMul:          OpMul,
	ir.OpScalarMul:    OpScalarMul,
	ir.OpSigmoid:      OpSigmoid,
	ir.OpSiLU:         OpSiLU,
	ir.OpSoftmax:      OpSoftmax,
	ir.OpReshape:      OpReshape,
	ir.OpTranspose:    OpTranspose,
	ir.OpSlice:        OpSlice,
	ir.OpConcat:       OpConcat,
	ir.OpCausalMask:   OpCausalMask,
	ir.OpCrossEntropy: OpCrossEntropy,
	ir.OpDropout:      OpDropout,
	ir.OpSquare:       OpSquare,
	ir.OpSub:          OpSub,
	ir.OpDiv:          OpDiv,
	ir.OpArange:       OpArange,
	ir.OpMeanAxis:     OpMeanAxis,
	ir.OpFull:         OpFull,
	ir.OpRMSNorm:      OpRMSNorm,
	ir.OpRoPE:         OpRoPE,
	ir.OpExp:          OpExp,
	ir.OpOuter:        OpOuter,
	ir.OpGELU:         OpGELU,
	ir.OpReLU:         OpReLU,
	ir.OpTanh:         OpTanh,
	ir.OpScan:         OpScan,
}

// irToGPUDType maps ir tensor dtype codes to gpu dtype codes.
var irToGPUDType = map[int]int{
	ir.TensorInt32:   TensorInt32,
	ir.TensorFloat32: TensorFloat32,
}

// LowerIRProgram takes a pure-Go ir.Program and produces a gpu.Program
// backed by the C++ MLX IR interpreter. This is the bridge between
// the ir package (no CGO) and the gpu package (all CGO).
func LowerIRProgram(irProg *ir.Program) (*Program, error) {
	if irProg == nil {
		return nil, fmt.Errorf("nil IR program")
	}
	if irProg.NumWeights <= 0 {
		return nil, fmt.Errorf("IR program has no weights")
	}

	gpuProg, err := NewProgram(irProg.NumWeights)
	if err != nil {
		return nil, fmt.Errorf("creating GPU program: %w", err)
	}

	// Declare inputs.
	for _, decl := range irProg.Inputs {
		gpuDType, ok := irToGPUDType[decl.DType]
		if !ok {
			gpuProg.Destroy()
			return nil, fmt.Errorf("unsupported input dtype %d for %q", decl.DType, decl.Name)
		}
		if err := gpuProg.DeclareInput(decl.Name, gpuDType, decl.Shape); err != nil {
			gpuProg.Destroy()
			return nil, fmt.Errorf("declaring input %q: %w", decl.Name, err)
		}
	}

	// Declare outputs.
	for _, decl := range irProg.Outputs {
		gpuDType, ok := irToGPUDType[decl.DType]
		if !ok {
			gpuProg.Destroy()
			return nil, fmt.Errorf("unsupported output dtype %d for %q", decl.DType, decl.Name)
		}
		if err := gpuProg.DeclareOutput(decl.Name, gpuDType, decl.Shape); err != nil {
			gpuProg.Destroy()
			return nil, fmt.Errorf("declaring output %q: %w", decl.Name, err)
		}
	}

	// Emit ops.
	for i, op := range irProg.Ops {
		gpuOp, ok := irToGPUOp[op.Code]
		if !ok {
			gpuProg.Destroy()
			return nil, fmt.Errorf("unsupported IR op code %d at index %d", op.Code, i)
		}
		if err := gpuProg.AddOp(gpuOp, op.Inputs, op.Outputs, op.FloatParams, op.IntParams); err != nil {
			gpuProg.Destroy()
			return nil, fmt.Errorf("adding op %d (code=%d): %w", i, op.Code, err)
		}
	}

	return gpuProg, nil
}
