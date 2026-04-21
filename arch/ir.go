// Package ir provides an intermediate representation for model forward passes.
// The IR program is a sequence of typed operations that can be lowered to
// MLX (or other backends) for execution.
package arch

import "fmt"

// Op codes for IR operations. These MUST match the C++ enum in h.
// Do NOT invent new opcodes — express new operations using existing ones.
const (
	OpEmbed            = 1  // OP_EMBED
	OpMatMul           = 2  // OP_MATMUL
	OpAdd              = 3  // OP_ADD
	OpMul              = 4  // OP_MUL
	OpScalarMul        = 5  // OP_SCALAR_MUL
	OpSigmoid          = 6  // OP_SIGMOID
	OpSiLU             = 7  // OP_SILU
	OpSoftmax          = 8  // OP_SOFTMAX
	OpReshape          = 9  // OP_RESHAPE
	OpTranspose        = 10 // OP_TRANSPOSE
	OpSlice            = 11 // OP_SLICE
	OpConcat           = 12 // OP_CONCAT
	OpCausalMask       = 13 // OP_CAUSAL_MASK
	OpCrossEntropy     = 14 // OP_CROSS_ENTROPY
	OpDropout          = 15 // OP_DROPOUT
	OpSquare           = 20 // OP_SQUARE
	OpSub              = 21 // OP_SUB
	OpDiv              = 22 // OP_DIV
	OpArange           = 28 // OP_ARANGE
	OpMeanAxis         = 29 // OP_MEAN_AXIS
	OpFull             = 30 // OP_FULL
	OpRMSNorm          = 33 // OP_RMSNORM
	OpRoPE             = 34 // OP_ROPE
	OpExp              = 39 // OP_EXP
	OpOuter            = 40 // OP_OUTER
	OpGELU             = 42 // OP_GELU
	OpReLU             = 43 // OP_RELU
	OpTanh             = 44 // OP_TANH
	OpScan             = 49 // OP_SCAN
	OpGatherPositions  = 51 // OP_GATHER_POSITIONS
	OpScatterPositions = 52 // OP_SCATTER_POSITIONS
	OpRoPEIndexed      = 53 // OP_ROPE_INDEXED

	TensorInt32   = 0
	TensorFloat32 = 1
)

// Op represents a single IR operation.
type Op struct {
	Code        int
	Inputs      []string
	Outputs     []string
	FloatParams []float32
	IntParams   []int
}

// TensorDecl describes an input or output tensor declaration.
type TensorDecl struct {
	Name  string
	DType int
	Shape []int
}

// Program holds a complete IR forward-pass graph.
type Program struct {
	NumWeights int
	Inputs     []TensorDecl
	Outputs    []TensorDecl
	Ops        []Op
}

// NewProgram creates an empty IR program expecting nWeights trainable weight tensors.
func NewProgram(nWeights int) *Program {
	return &Program{NumWeights: nWeights}
}

// DeclareInput registers a named input tensor.
func (p *Program) DeclareInput(name string, dtype int, shape []int) {
	p.Inputs = append(p.Inputs, TensorDecl{Name: name, DType: dtype, Shape: shape})
}

// DeclareOutput registers a named output tensor.
func (p *Program) DeclareOutput(name string, dtype int, shape []int) {
	p.Outputs = append(p.Outputs, TensorDecl{Name: name, DType: dtype, Shape: shape})
}

// AddOp appends a raw operation to the program.
func (p *Program) AddOp(code int, inputs, outputs []string, floatParams []float32, intParams []int) {
	p.Ops = append(p.Ops, Op{
		Code:        code,
		Inputs:      inputs,
		Outputs:     outputs,
		FloatParams: floatParams,
		IntParams:   intParams,
	})
}

// Embed emits an embedding lookup: output = table[indices].
func (p *Program) Embed(table, indices, output string) {
	p.AddOp(OpEmbed, []string{table, indices}, []string{output}, nil, nil)
}

// MatMul emits a matrix multiply: output = a @ b.
func (p *Program) MatMul(a, b, output string) {
	p.AddOp(OpMatMul, []string{a, b}, []string{output}, nil, nil)
}

// Add emits element-wise addition: output = a + b.
func (p *Program) Add(a, b, output string) {
	p.AddOp(OpAdd, []string{a, b}, []string{output}, nil, nil)
}

// Sub emits element-wise subtraction: output = a - b.
func (p *Program) Sub(a, b, output string) {
	p.AddOp(OpSub, []string{a, b}, []string{output}, nil, nil)
}

// GELU emits a GELU activation.
func (p *Program) GELU(a, output string) {
	p.AddOp(OpGELU, []string{a}, []string{output}, nil, nil)
}

// Tanh emits a tanh activation.
func (p *Program) Tanh(a, output string) {
	p.AddOp(OpTanh, []string{a}, []string{output}, nil, nil)
}

// Mul emits element-wise multiplication: output = a * b.
func (p *Program) Mul(a, b, output string) {
	p.AddOp(OpMul, []string{a, b}, []string{output}, nil, nil)
}

// ScalarMul emits scalar multiplication: output = a * s.
func (p *Program) ScalarMul(a string, s float32, output string) {
	p.AddOp(OpScalarMul, []string{a}, []string{output}, []float32{s}, nil)
}

// Dropout emits inverted dropout with probability rate.
func (p *Program) Dropout(a string, rate float32, output string) {
	p.AddOp(OpDropout, []string{a}, []string{output}, []float32{rate}, nil)
}

// Sigmoid emits a sigmoid activation.
func (p *Program) Sigmoid(a, output string) {
	p.AddOp(OpSigmoid, []string{a}, []string{output}, nil, nil)
}

// SiLU emits a SiLU (swish) activation.
func (p *Program) SiLU(a, output string) {
	p.AddOp(OpSiLU, []string{a}, []string{output}, nil, nil)
}

// Softmax emits a softmax along the given axis.
func (p *Program) Softmax(a string, axis int, output string) {
	p.AddOp(OpSoftmax, []string{a}, []string{output}, nil, []int{axis})
}

// Reshape emits a reshape to the given shape.
func (p *Program) Reshape(a string, shape []int, output string) {
	p.AddOp(OpReshape, []string{a}, []string{output}, nil, shape)
}

// Transpose emits a transpose along the given axes.
func (p *Program) Transpose(a string, axes []int, output string) {
	p.AddOp(OpTranspose, []string{a}, []string{output}, nil, axes)
}

// CausalMask applies a causal attention mask.
func (p *Program) CausalMask(scores string, T int, output string) {
	p.AddOp(OpCausalMask, []string{scores}, []string{output}, nil, []int{T})
}

// CrossEntropy emits a cross-entropy loss computation.
func (p *Program) CrossEntropy(logits, targets, output string) {
	p.AddOp(OpCrossEntropy, []string{logits, targets}, []string{output}, nil, nil)
}

// RMSNorm emits RMS normalization with a learned scale parameter.
func (p *Program) RMSNorm(x, scale, output string, eps float32) {
	p.AddOp(OpRMSNorm, []string{x, scale}, []string{output}, []float32{eps}, nil)
}

// RoPE emits rotary position embeddings.
func (p *Program) RoPE(q, k, qOut, kOut string, T, headDim int, base float32) {
	p.AddOp(OpRoPE, []string{q, k}, []string{qOut, kOut}, []float32{base}, []int{T, headDim})
}

// RoPEIndexed emits rotary position embeddings using explicit position indices.
func (p *Program) RoPEIndexed(q, k, positions, qOut, kOut string, K, headDim int, base float32) {
	p.AddOp(OpRoPEIndexed, []string{q, k, positions}, []string{qOut, kOut}, []float32{base}, []int{K, headDim})
}

// Broadcast tiles a 1-D or 2-D tensor along a new leading batch dimension.
// Implemented via reshape + full + mul using backend broadcasting.
// This helper is intentionally limited to 1-D inputs, producing [repeats, D].
func (p *Program) Broadcast(a string, repeats int, output string) {
	row := output + "_row"
	ones := output + "_ones"
	p.Reshape(a, []int{1, -1}, row)
	p.Full([]int{repeats, 1}, 1.0, ones)
	p.Mul(row, ones, output)
}

// Slice emits a slice along axis: output = a[start:end:step] on the given axis.
// IntParams layout: [start, end, step, axis].
func (p *Program) Slice(a string, start, end, step, axis int, output string) {
	p.AddOp(OpSlice, []string{a}, []string{output}, nil, []int{start, end, step, axis})
}

// Concat emits concatenation of two tensors along axis.
func (p *Program) Concat(a, b string, axis int, output string) {
	p.AddOp(OpConcat, []string{a, b}, []string{output}, nil, []int{axis})
}

// Scan emits a gated recurrence (OpScan) over a sequence.
// IntParams layout: [B, T, D].
func (p *Program) Scan(x, decay, output string, B, T, D int) {
	p.AddOp(OpScan, []string{x, decay}, []string{output}, nil, []int{B, T, D})
}

// GatherPositions selects K entries from the position axis of a [B,T,D] tensor.
// IntParams layout: [B, K, D].
func (p *Program) GatherPositions(input, positions, output string, B, K, D int) {
	p.AddOp(OpGatherPositions, []string{input, positions}, []string{output}, nil, []int{B, K, D})
}

// ScatterPositions overwrites K entries on the position axis of a [B,T,D] tensor.
// IntParams layout: [B, T, K, D].
func (p *Program) ScatterPositions(input, updates, positions, output string, B, T, K, D int) {
	p.AddOp(OpScatterPositions, []string{input, updates, positions}, []string{output}, nil, []int{B, T, K, D})
}

// Exp emits element-wise exponential: output = exp(a).
func (p *Program) Exp(a, output string) {
	p.AddOp(OpExp, []string{a}, []string{output}, nil, nil)
}

// ReLU emits element-wise rectified linear unit: output = max(a, 0).
func (p *Program) ReLU(a, output string) {
	p.AddOp(OpReLU, []string{a}, []string{output}, nil, nil)
}

// Square emits element-wise square: output = a * a.
func (p *Program) Square(a, output string) {
	p.AddOp(OpSquare, []string{a}, []string{output}, nil, nil)
}

// Div emits element-wise division: output = a / b.
func (p *Program) Div(a, b, output string) {
	p.AddOp(OpDiv, []string{a, b}, []string{output}, nil, nil)
}

// DivSafe emits element-wise division with epsilon: output = a / (b + eps).
// FloatParams layout: [eps].
func (p *Program) DivSafe(a, b string, eps float32, output string) {
	p.AddOp(OpDiv, []string{a, b}, []string{output}, []float32{eps}, nil)
}

// Arange emits a 1-D integer range [start, end).
func (p *Program) Arange(start, end int, output string) {
	p.AddOp(OpArange, nil, []string{output}, nil, []int{start, end})
}

// MeanAxis emits a mean reduction along the given axis.
func (p *Program) MeanAxis(a string, axis int, output string) {
	p.AddOp(OpMeanAxis, []string{a}, []string{output}, nil, []int{axis})
}

// Full emits a full tensor with the given shape and scalar value.
func (p *Program) Full(shape []int, value float32, output string) {
	p.AddOp(OpFull, nil, []string{output}, []float32{value}, shape)
}

// Outer emits an outer product between flattened inputs.
func (p *Program) Outer(a, b, output string) {
	p.AddOp(OpOuter, []string{a, b}, []string{output}, nil, nil)
}

// NegExp emits element-wise negative exponential: output = exp(-a).
// Implemented as ScalarMul(-1) then Exp.
func (p *Program) NegExp(a, neg, output string) {
	p.ScalarMul(a, -1.0, neg)
	p.Exp(neg, output)
}

// weightName returns the canonical IR name for weight index i.
func weightName(i int) string {
	return fmt.Sprintf("w%d", i)
}

// tmpName returns a unique temporary variable name.
func tmpName(base string, idx int) string {
	return fmt.Sprintf("%s_%d", base, idx)
}
