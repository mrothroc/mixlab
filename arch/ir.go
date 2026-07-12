// Package ir provides an intermediate representation for model forward passes.
// The IR program is a sequence of typed operations that can be lowered to
// MLX (or other backends) for execution.
package arch

import "fmt"

// Op codes for IR operations. These MUST match the C++ enum in h.
const (
	OpEmbed                = 1  // OP_EMBED
	OpMatMul               = 2  // OP_MATMUL
	OpAdd                  = 3  // OP_ADD
	OpMul                  = 4  // OP_MUL
	OpScalarMul            = 5  // OP_SCALAR_MUL
	OpSigmoid              = 6  // OP_SIGMOID
	OpSiLU                 = 7  // OP_SILU
	OpSoftmax              = 8  // OP_SOFTMAX
	OpReshape              = 9  // OP_RESHAPE
	OpTranspose            = 10 // OP_TRANSPOSE
	OpSlice                = 11 // OP_SLICE
	OpConcat               = 12 // OP_CONCAT
	OpCausalMask           = 13 // OP_CAUSAL_MASK
	OpCrossEntropy         = 14 // OP_CROSS_ENTROPY
	OpDropout              = 15 // OP_DROPOUT
	OpSquare               = 20 // OP_SQUARE
	OpSub                  = 21 // OP_SUB
	OpDiv                  = 22 // OP_DIV
	OpWhere                = 25 // OP_WHERE
	OpLessThan             = 26 // OP_LESS_THAN
	OpGreaterEq            = 27 // OP_GREATER_EQ
	OpArange               = 28 // OP_ARANGE
	OpMeanAxis             = 29 // OP_MEAN_AXIS
	OpFull                 = 30 // OP_FULL
	OpRMSNorm              = 33 // OP_RMSNORM
	OpRoPE                 = 34 // OP_ROPE
	OpSqrt                 = 35 // OP_SQRT
	OpRSqrt                = 36 // OP_RSQRT
	OpExp                  = 39 // OP_EXP
	OpOuter                = 40 // OP_OUTER
	OpGELU                 = 42 // OP_GELU
	OpReLU                 = 43 // OP_RELU
	OpTanh                 = 44 // OP_TANH
	OpScan                 = 49 // OP_SCAN
	OpGatherPositions      = 51 // OP_GATHER_POSITIONS
	OpScatterPositions     = 52 // OP_SCATTER_POSITIONS
	OpRoPEIndexed          = 53 // OP_ROPE_INDEXED
	OpLeakyReLU            = 54 // OP_LEAKY_RELU
	OpXSAProject           = 55 // OP_XSA_PROJECT
	OpCrossEntropyPerToken = 56 // OP_CROSS_ENTROPY_PER_TOKEN
	OpMatrixScan           = 57 // OP_MATRIX_SCAN
	OpScanTV               = 58 // OP_SCAN_TV
	OpSoftplus             = 59 // OP_SOFTPLUS
	OpGatedDeltaScan       = 60 // OP_GATED_DELTA_SCAN
	OpStopGradient         = 61 // OP_STOP_GRADIENT
	OpDepthwiseConv1D      = 62 // OP_DEPTHWISE_CONV1D
	OpMamba3SelectiveScan  = 63 // OP_MAMBA3_SELECTIVE_SCAN
	OpMamba3CanonicalBlock = 64 // OP_MAMBA3_CANONICAL_BLOCK
	OpRandomNormal         = 65 // OP_RANDOM_NORMAL
	OpFirstByteMaskedCE    = 66 // OP_FIRST_BYTE_MASKED_CROSS_ENTROPY
	OpMaskedCrossEntropy   = 67 // OP_MASKED_CROSS_ENTROPY
	OpMaskedCEPerToken     = 68 // OP_MASKED_CROSS_ENTROPY_PER_TOKEN
	OpDistillationKL       = 69 // OP_DISTILLATION_KL
	OpHGRN2Scan            = 70 // OP_HGRN2_SCAN
	OpMLSTMScan            = 71 // OP_MLSTM_SCAN
	OpDebertaRelativeBias  = 72 // OP_DEBERTA_RELATIVE_BIAS
	OpCharFeatureBag       = 73 // OP_CHAR_FEATURE_BAG
	OpMoEFeedForward       = 74 // OP_MOE_FEED_FORWARD
	OpMaskedSmoothL1       = 75 // OP_MASKED_SMOOTH_L1
	OpZLoss                = 76 // OP_Z_LOSS
	OpLog                  = 77 // OP_LOG
	OpReciprocal           = 78 // OP_RECIPROCAL
	OpPow                  = 79 // OP_POW
	OpAbs                  = 80 // OP_ABS
	OpClamp                = 81 // OP_CLAMP
	OpMinimum              = 82 // OP_MINIMUM
	OpMaximum              = 83 // OP_MAXIMUM
	OpGreaterThan          = 84 // OP_GREATER_THAN
	OpLessEq               = 85 // OP_LESS_EQ
	OpEqual                = 86 // OP_EQUAL
	OpLayerNorm            = 87 // OP_LAYERNORM
	OpSelectiveCausalMask  = 88 // OP_SELECTIVE_CAUSAL_MASK
	OpSegmentAttentionMask = 89 // OP_SEGMENT_ATTENTION_MASK
	OpBlockDiffusionMask   = 90 // OP_BLOCK_DIFFUSION_MASK
	OpGELUExact            = 91 // OP_GELU_EXACT
	OpMaskedBCEWithLogits  = 92 // OP_MASKED_BCE_WITH_LOGITS
	OpMaskedBinaryAccuracy = 93 // OP_MASKED_BINARY_ACCURACY
	OpEnergyPairwiseLoss   = 94 // OP_ENERGY_PAIRWISE_LOSS
	OpEnergySpanPool       = 95 // OP_ENERGY_SPAN_POOL
	OpEnergySpanPairwise   = 96 // OP_ENERGY_SPAN_PAIRWISE_LOSS
	OpSpanPLLPool          = 97 // OP_SPAN_PLL_POOL
	OpSpanPLLPairwise      = 98 // OP_SPAN_PLL_PAIRWISE_LOSS
	OpMaskedDistillationKL = 99 // OP_MASKED_DISTILLATION_KL
)

const OpMaskedSymmetricKL = 100 // OP_MASKED_SYMMETRIC_KL

const OpMaskedMarginPLL = 101 // OP_MASKED_MARGIN_PLL

const OpMaskedZLoss = 102 // OP_MASKED_Z_LOSS

const (
	SegmentMaskModeNone            = 0
	SegmentMaskModeCausal          = 1
	SegmentMaskModeSelectiveCausal = 2

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

// GELUExact emits the exact erf-based GELU activation.
func (p *Program) GELUExact(a, output string) {
	p.AddOp(OpGELUExact, []string{a}, []string{output}, nil, nil)
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
// windowSize <= 0 preserves the full causal lower triangle.
func (p *Program) CausalMask(scores string, T, windowSize int, output string) {
	p.AddOp(OpCausalMask, []string{scores}, []string{output}, nil, []int{T, windowSize})
}

// SelectiveCausalMask applies a causal mask only for batch rows where
// causalRows[b] > 0. Other rows keep dense bidirectional scores.
func (p *Program) SelectiveCausalMask(scores, causalRows string, T, windowSize int, output string) {
	p.AddOp(OpSelectiveCausalMask, []string{scores, causalRows}, []string{output}, nil, []int{T, windowSize})
}

// SegmentAttentionMask applies a block-diagonal segment mask from segmentIDs.
// mode selects whether to also apply no causal mask, all-row causal masking, or
// per-batch-row selective causal masking using causalRows.
func (p *Program) SegmentAttentionMask(scores, segmentIDs, causalRows string, T, windowSize, mode int, output string) {
	inputs := []string{scores, segmentIDs}
	if causalRows != "" {
		inputs = append(inputs, causalRows)
	}
	p.AddOp(OpSegmentAttentionMask, inputs, []string{output}, nil, []int{T, windowSize, mode})
}

// BlockDiffusionMask applies prefix-plus-block attention masking from
// per-example active block boundaries.
func (p *Program) BlockDiffusionMask(scores, blockStart, blockEnd string, T int, output string) {
	p.AddOp(OpBlockDiffusionMask, []string{scores, blockStart, blockEnd}, []string{output}, nil, []int{T})
}

// CrossEntropy emits a cross-entropy loss computation.
func (p *Program) CrossEntropy(logits, targets, output string) {
	p.AddOp(OpCrossEntropy, []string{logits, targets}, []string{output}, nil, nil)
}

// FirstByteMaskedCrossEntropy emits mean cross-entropy using a masked softmax
// for rows whose target token starts a UTF-8 codepoint. firstByteValid is a
// length-vocab int32 vector where nonzero entries are valid first-byte tokens.
func (p *Program) FirstByteMaskedCrossEntropy(logits, targets, firstByteValid, output string) {
	p.AddOp(OpFirstByteMaskedCE, []string{logits, targets, firstByteValid}, []string{output}, nil, nil)
}

// MaskedCrossEntropy emits mean cross-entropy over rows where lossMask > 0.
func (p *Program) MaskedCrossEntropy(logits, targets, lossMask, output string) {
	p.AddOp(OpMaskedCrossEntropy, []string{logits, targets, lossMask}, []string{output}, nil, nil)
}

// CrossEntropyPerToken emits per-token negative log-likelihoods.
func (p *Program) CrossEntropyPerToken(logits, targets, output string) {
	p.AddOp(OpCrossEntropyPerToken, []string{logits, targets}, []string{output}, nil, nil)
}

// MaskedCrossEntropyPerToken emits per-token NLLs zeroed where lossMask <= 0.
func (p *Program) MaskedCrossEntropyPerToken(logits, targets, lossMask, output string) {
	p.AddOp(OpMaskedCEPerToken, []string{logits, targets, lossMask}, []string{output}, nil, nil)
}

// DistillationKL emits mean KL(P_teacher || P_student) from teacher
// probabilities and student logits.
func (p *Program) DistillationKL(logits, teacherProbs, output string) {
	p.AddOp(OpDistillationKL, []string{logits, teacherProbs}, []string{output}, nil, nil)
}

// MaskedDistillationKL emits mean KL(P_teacher || P_student) over rows where
// lossMask > 0. Student logits are divided by temperature before softmax.
func (p *Program) MaskedDistillationKL(logits, teacherProbs, lossMask string, temperature float32, output string) {
	p.AddOp(OpMaskedDistillationKL, []string{logits, teacherProbs, lossMask}, []string{output}, []float32{temperature}, nil)
}

// MaskedSymmetricKL emits 0.5 * (KL(P_a||P_b) + KL(P_b||P_a)) for
// contiguous A/B sequence-row pairs. lossMask selects the one compared
// position in each view; inactive pairs have all-zero masks.
func (p *Program) MaskedSymmetricKL(logits, lossMask string, seqLen int, output string) {
	p.AddOp(OpMaskedSymmetricKL, []string{logits, lossMask}, []string{output}, nil, []int{seqLen})
}

// MaskedSmoothL1 emits mean Huber/SmoothL1 loss over rows where mask > 0.
func (p *Program) MaskedSmoothL1(pred, target, mask string, beta float32, output string) {
	p.AddOp(OpMaskedSmoothL1, []string{pred, target, mask}, []string{output}, []float32{beta}, nil)
}

// MaskedBCEWithLogits emits mean binary cross-entropy over rows where mask > 0.
func (p *Program) MaskedBCEWithLogits(logits, targets, mask, output string) {
	p.AddOp(OpMaskedBCEWithLogits, []string{logits, targets, mask}, []string{output}, nil, nil)
}

// MaskedBinaryAccuracy emits mean binary accuracy over rows where mask > 0.
func (p *Program) MaskedBinaryAccuracy(logits, targets, mask, output string) {
	p.AddOp(OpMaskedBinaryAccuracy, []string{logits, targets, mask}, []string{output}, nil, nil)
}

// EnergyPairwiseLoss emits pairwise ranking loss/diagnostics over contiguous
// clean/corrupt row pairs. lossKind is EnergyPairLossLogistic or
// EnergyPairLossHinge; rowMask marks active rows and both rows in a pair must
// be active for that pair to contribute.
func (p *Program) EnergyPairwiseLoss(logits, rowMask string, lossKind int, margin float32, loss, accuracy, cleanMean, corruptMean string) {
	p.AddOp(OpEnergyPairwiseLoss, []string{logits, rowMask}, []string{loss, accuracy, cleanMean, corruptMean}, []float32{margin}, []int{lossKind})
}

// EnergySpanPool mean-pools per-token scalar energies over positive mask rows.
// The logits input may be [B*T] or [B*T,1]; mask is [B*T]. Output is [B,1].
func (p *Program) EnergySpanPool(logits, spanMask string, seqLen int, output string) {
	p.AddOp(OpEnergySpanPool, []string{logits, spanMask}, []string{output}, nil, []int{seqLen})
}

// EnergySpanPairwiseLoss pools per-token scalar energies by span and applies
// clean/corrupt pairwise logistic or hinge ranking loss.
func (p *Program) EnergySpanPairwiseLoss(logits, spanMask string, seqLen, lossKind int, margin float32, loss, accuracy, cleanMean, corruptMean string) {
	p.AddOp(OpEnergySpanPairwise, []string{logits, spanMask}, []string{loss, accuracy, cleanMean, corruptMean}, []float32{margin}, []int{seqLen, lossKind})
}

// SpanPLLPool sums log_softmax(logits)[target] over positive span-mask
// positions for each row. Higher pooled scores are better.
func (p *Program) SpanPLLPool(logits, targets, spanMask string, seqLen int, output string) {
	p.AddOp(OpSpanPLLPool, []string{logits, targets, spanMask}, []string{output}, nil, []int{seqLen})
}

// SpanPLLPairwiseLoss pools span PLL scores and applies clean/corrupt pairwise
// logistic or hinge ranking loss where higher clean scores are preferred.
func (p *Program) SpanPLLPairwiseLoss(logits, targets, spanMask string, seqLen, lossKind int, margin float32, loss, accuracy, cleanMean, corruptMean string) {
	p.AddOp(OpSpanPLLPairwise, []string{logits, targets, spanMask}, []string{loss, accuracy, cleanMean, corruptMean}, []float32{margin}, []int{seqLen, lossKind})
}

// MaskedMarginPLL emits a directional paired span-PLL margin loss over
// contiguous preferred/contrast row pairs. Each row's positive mask selects
// the unchanged target span. The result is rank_loss + anchor_weight *
// negative(preferred_score), where rank_loss is softplus(margin - delta).
func (p *Program) MaskedMarginPLL(logits, targets, spanMask string, seqLen int, margin, anchorWeight float32, loss, rankLoss, anchorLoss, deltaMean string) {
	p.AddOp(OpMaskedMarginPLL, []string{logits, targets, spanMask}, []string{loss, rankLoss, anchorLoss, deltaMean}, []float32{margin, anchorWeight}, []int{seqLen})
}

// ZLoss emits mean(square(logsumexp(logits))) over token rows.
func (p *Program) ZLoss(logits, output string) {
	p.AddOp(OpZLoss, []string{logits}, []string{output}, nil, nil)
}

// MaskedZLoss emits mean(square(logsumexp(logits))) over positive mask rows.
func (p *Program) MaskedZLoss(logits, lossMask, output string) {
	p.AddOp(OpMaskedZLoss, []string{logits, lossMask}, []string{output}, nil, nil)
}

// RMSNorm emits RMS normalization with a learned scale parameter.
func (p *Program) RMSNorm(x, scale, output string, eps float32) {
	p.AddOp(OpRMSNorm, []string{x, scale}, []string{output}, []float32{eps}, nil)
}

// LayerNorm emits layer normalization over the last dimension with learned
// affine scale and bias parameters.
func (p *Program) LayerNorm(x, scale, bias, output string, eps float32) {
	p.AddOp(OpLayerNorm, []string{x, scale, bias}, []string{output}, []float32{eps}, nil)
}

// LayerNormNoAffine emits layer normalization over the last dimension without
// learned affine parameters.
func (p *Program) LayerNormNoAffine(x, output string, eps float32) {
	p.AddOp(OpLayerNorm, []string{x}, []string{output}, []float32{eps}, nil)
}

// RoPE emits rotary position embeddings.
func (p *Program) RoPE(q, k, qOut, kOut string, T, headDim, ropeDims int, base float32) {
	p.AddOp(OpRoPE, []string{q, k}, []string{qOut, kOut}, []float32{base}, []int{T, headDim, ropeDims})
}

// RoPEWithConvention emits rotary position embeddings using the selected
// convention. Existing RoPE calls use Mixlab's adjacent-pair convention.
func (p *Program) RoPEWithConvention(q, k, qOut, kOut string, T, headDim, ropeDims int, base float32, convention string) {
	p.AddOp(OpRoPE, []string{q, k}, []string{qOut, kOut}, []float32{base}, []int{T, headDim, ropeDims, 0, 1, ropeConventionCode(convention)})
}

// RoPEIndexed emits rotary position embeddings using explicit position indices.
func (p *Program) RoPEIndexed(q, k, positions, qOut, kOut string, K, headDim, ropeDims int, base float32) {
	p.AddOp(OpRoPEIndexed, []string{q, k, positions}, []string{qOut, kOut}, []float32{base}, []int{K, headDim, ropeDims})
}

// RoPEIndexedWithConvention emits indexed rotary embeddings with the selected
// convention. Existing RoPEIndexed calls use adjacent-pair rotation.
func (p *Program) RoPEIndexedWithConvention(q, k, positions, qOut, kOut string, K, headDim, ropeDims int, base float32, convention string) {
	p.AddOp(OpRoPEIndexed, []string{q, k, positions}, []string{qOut, kOut}, []float32{base}, []int{K, headDim, ropeDims, ropeConventionCode(convention)})
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

// DepthwiseConv1D emits a causal depthwise 1-D convolution over a flattened
// [B*T, D] sequence. Weight shape is [D, K]. IntParams layout: [B, T, D, K].
func (p *Program) DepthwiseConv1D(x, weight, output string, B, T, D, K int) {
	p.AddOp(OpDepthwiseConv1D, []string{x, weight}, []string{output}, nil, []int{B, T, D, K})
}

// Mamba3SelectiveScan emits the canonical Mamba-3 recurrent core from Lahoti
// et al. 2026, Sections 3.1-3.3 / Propositions 1, 2, and 4.
//
// Inputs:
//
//	x      [B*T, D]     SSM input branch after optional causal depthwise conv
//	dt     [B*T, D]     raw delta logits including dt_bias; kernel applies softplus
//	lambda [B*T, D]     raw trapezoid gate logits; kernel applies sigmoid
//	theta  [B*T, D*N/2] raw complex-state angular velocities for N/2 pairs
//	a_log  [D, N]       log state decay magnitudes; A = -exp(a_log)
//	B_proj [B*T, G*N]   MIMO/grouped B, interpreted as [B,T,G,N]
//	C_proj [B*T, G*N]   MIMO/grouped C, interpreted as [B,T,G,N]
//
// Output is [B*T, D]. IntParams layout: [B, T, D, N, G] or
// [B, T, D, N, G, scan_chunk_size]. A positive scan_chunk_size uses an exact
// chunked affine scan; 0/omitted keeps the original full-sequence scan.
func (p *Program) Mamba3SelectiveScan(x, dt, lambda, theta, aLog, bProj, cProj, output string, B, T, D, N, G int) {
	p.Mamba3SelectiveScanChunked(x, dt, lambda, theta, aLog, bProj, cProj, output, B, T, D, N, G, 0)
}

// Mamba3SelectiveScanChunked emits the same canonical recurrence as
// Mamba3SelectiveScan, evaluated in chunks when scanChunkSize > 0.
func (p *Program) Mamba3SelectiveScanChunked(x, dt, lambda, theta, aLog, bProj, cProj, output string, B, T, D, N, G, scanChunkSize int) {
	params := []int{B, T, D, N, G}
	if scanChunkSize > 0 {
		params = append(params, scanChunkSize)
	}
	p.AddOp(OpMamba3SelectiveScan, []string{x, dt, lambda, theta, aLog, bProj, cProj}, []string{output}, nil, params)
}

// Mamba3CanonicalBlock emits the full canonical Mamba-3 block as one IR op.
// Input layout:
//
//	x, pre_norm, W_X, [conv_w], W_dt_low, W_dt_high, W_lambda_low,
//	W_lambda_high, W_theta_low, W_theta_high, W_B, W_C, B_norm_scale,
//	C_norm_scale, B_bias, C_bias, A_log, dt_bias, post_norm_scale, W_Z, W_O.
//
// IntParams layout: [B, T, use_conv, scan_chunk_size].
func (p *Program) Mamba3CanonicalBlock(inputs []string, output string, B, T int, useConv bool, scanChunkSize int) {
	useConvInt := 0
	if useConv {
		useConvInt = 1
	}
	p.AddOp(OpMamba3CanonicalBlock, inputs, []string{output}, []float32{1e-5}, []int{B, T, useConvInt, scanChunkSize})
}

// MatrixScan emits a matrix-state gated recurrence (OpMatrixScan) over a sequence.
// IntParams layout: [B, T, Da, Db].
func (p *Program) MatrixScan(update, gate, out string, B, T, Da, Db int) {
	p.AddOp(OpMatrixScan, []string{update, gate}, []string{out}, nil, []int{B, T, Da, Db})
}

// ScanTV emits a time-varying gated recurrence over a sequence.
// IntParams layout: [B, T, D].
func (p *Program) ScanTV(x, gate, out string, B, T, D int) {
	p.AddOp(OpScanTV, []string{x, gate}, []string{out}, nil, []int{B, T, D})
}

// GatedDeltaScan emits the full gated delta-rule recurrence over a sequence.
// Inputs:
//
//	q: [B,T,H,Dk] already L2-normalized and scaled
//	k: [B,T,H,Dk] already L2-normalized
//	v: [B,T,H,Dv]
//	beta: [B,T,H]
//	gate: [B,T,H] decay in (0, 1]
//
// Output:
//
//	out: [B*T*H,Dv]
//
// IntParams layout: [B, T, H, Dk, Dv, chunkSize].
// chunkSize <= 0 keeps the legacy naive recurrence for parity/debugging.
func (p *Program) GatedDeltaScan(q, k, v, beta, gate, out string, B, T, H, Dk, Dv, chunkSize int) {
	p.AddOp(OpGatedDeltaScan, []string{q, k, v, beta, gate}, []string{out}, nil, []int{B, T, H, Dk, Dv, chunkSize})
}

// HGRN2Scan emits the HGRN2 matrix-state recurrence over a sequence.
// Inputs:
//
//	q:    [B,T,H,Ds]
//	k:    [B,T,H,Ds]
//	v:    [B,T,H,Dv]
//	gate: [B,T,H,Ds] forget gate in [0, 1]
//
// Output:
//
//	out: [B*T*H,Dv]
//
// IntParams layout: [B, T, H, Ds, Dv].
func (p *Program) HGRN2Scan(q, k, v, gate, out string, B, T, H, Ds, Dv int) {
	p.AddOp(OpHGRN2Scan, []string{q, k, v, gate}, []string{out}, nil, []int{B, T, H, Ds, Dv})
}

// MLSTMScan emits the stabilized mLSTM matrix-memory recurrence.
// Inputs:
//
//	q:          [B,T,H,Dk]
//	k:          [B,T,H,Dk]
//	v:          [B,T,H,Dv]
//	inputGate:  [B,T,H] input-gate preactivation
//	forgetGate: [B,T,H] forget-gate preactivation
//
// Output:
//
//	out: [B*T*H,Dv]
//
// IntParams layout: [B, T, H, Dk, Dv].
func (p *Program) MLSTMScan(q, k, v, inputGate, forgetGate, out string, B, T, H, Dk, Dv int) {
	p.AddOp(OpMLSTMScan, []string{q, k, v, inputGate, forgetGate}, []string{out}, nil, []int{B, T, H, Dk, Dv})
}

// DebertaRelativeBias emits DeBERTa-style C2P+P2C disentangled relative
// attention bias. Inputs are q/k [B,H,T,D] and projected position tensors
// [H,2*window-1,D]. Relative positions use the GPT-BERT/DeBERTa log-bucketed
// q-k index matrix for both C2P and P2C terms. Output is [B,H,T,T].
func (p *Program) DebertaRelativeBias(q, k, posKey, posQuery, out string, B, T, H, D, window int) {
	p.AddOp(OpDebertaRelativeBias, []string{q, k, posKey, posQuery}, []string{out}, nil, []int{B, T, H, D, window})
}

// CharFeatureBag sums fixed sparse character/byte feature embeddings for each
// token. table is [char_vocab_size,D], ids is [B,T,K], output is [B*T,D].
// Padding id 0 contributes exactly zero.
func (p *Program) CharFeatureBag(table, ids, out string, B, T, K, D int) {
	p.AddOp(OpCharFeatureBag, []string{table, ids}, []string{out}, nil, []int{B, T, K, D})
}

// MoEFeedForward emits a routed feed-forward Mixture-of-Experts block. Inputs
// are x [B*T,D], router_w [D,E], followed by per-expert FFN weights in expert
// index order. Outputs are delta [B*T,D], unweighted load-balance loss [1],
// and router entropy [1]. IntParams layout: [B,T,D,E,topK,expertType,ffn,activation].
func (p *Program) MoEFeedForward(inputs []string, delta, auxLoss, entropy string, B, T, D, experts, topK, expertType, ffn, activation int, leakySlope float32) {
	p.AddOp(OpMoEFeedForward, inputs, []string{delta, auxLoss, entropy}, []float32{leakySlope}, []int{B, T, D, experts, topK, expertType, ffn, activation})
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

// Log emits element-wise natural logarithm.
func (p *Program) Log(a, output string) {
	p.AddOp(OpLog, []string{a}, []string{output}, nil, nil)
}

// Sqrt emits element-wise square root.
func (p *Program) Sqrt(a, output string) {
	p.AddOp(OpSqrt, []string{a}, []string{output}, nil, nil)
}

// RSqrt emits element-wise reciprocal square root.
func (p *Program) RSqrt(a, output string) {
	p.AddOp(OpRSqrt, []string{a}, []string{output}, nil, nil)
}

// Reciprocal emits element-wise reciprocal.
func (p *Program) Reciprocal(a, output string) {
	p.AddOp(OpReciprocal, []string{a}, []string{output}, nil, nil)
}

// Pow emits element-wise power with tensor exponents.
func (p *Program) Pow(a, exponent, output string) {
	p.AddOp(OpPow, []string{a, exponent}, []string{output}, nil, nil)
}

// PowScalar emits element-wise power with a scalar exponent.
func (p *Program) PowScalar(a string, exponent float32, output string) {
	p.AddOp(OpPow, []string{a}, []string{output}, []float32{exponent}, nil)
}

// Abs emits element-wise absolute value.
func (p *Program) Abs(a, output string) {
	p.AddOp(OpAbs, []string{a}, []string{output}, nil, nil)
}

// Clamp emits element-wise clamp to [minValue, maxValue].
func (p *Program) Clamp(a string, minValue, maxValue float32, output string) {
	p.AddOp(OpClamp, []string{a}, []string{output}, []float32{minValue, maxValue}, nil)
}

// Minimum emits element-wise minimum.
func (p *Program) Minimum(a, b, output string) {
	p.AddOp(OpMinimum, []string{a, b}, []string{output}, nil, nil)
}

// Maximum emits element-wise maximum.
func (p *Program) Maximum(a, b, output string) {
	p.AddOp(OpMaximum, []string{a, b}, []string{output}, nil, nil)
}

// Where emits element-wise select: condition ? ifTrue : ifFalse.
func (p *Program) Where(condition, ifTrue, ifFalse, output string) {
	p.AddOp(OpWhere, []string{condition, ifTrue, ifFalse}, []string{output}, nil, nil)
}

// LessThan emits element-wise less-than comparison.
func (p *Program) LessThan(a, b, output string) {
	p.AddOp(OpLessThan, []string{a, b}, []string{output}, nil, nil)
}

// LessThanScalar emits element-wise less-than comparison against a scalar.
func (p *Program) LessThanScalar(a string, scalar float32, output string) {
	p.AddOp(OpLessThan, []string{a}, []string{output}, []float32{scalar}, nil)
}

// GreaterThan emits element-wise greater-than comparison.
func (p *Program) GreaterThan(a, b, output string) {
	p.AddOp(OpGreaterThan, []string{a, b}, []string{output}, nil, nil)
}

// GreaterThanScalar emits element-wise greater-than comparison against a scalar.
func (p *Program) GreaterThanScalar(a string, scalar float32, output string) {
	p.AddOp(OpGreaterThan, []string{a}, []string{output}, []float32{scalar}, nil)
}

// GreaterEq emits element-wise greater-than-or-equal comparison.
func (p *Program) GreaterEq(a, b, output string) {
	p.AddOp(OpGreaterEq, []string{a, b}, []string{output}, nil, nil)
}

// GreaterEqScalar emits element-wise greater-than-or-equal comparison against a scalar.
func (p *Program) GreaterEqScalar(a string, scalar float32, output string) {
	p.AddOp(OpGreaterEq, []string{a}, []string{output}, []float32{scalar}, nil)
}

// LessEq emits element-wise less-than-or-equal comparison.
func (p *Program) LessEq(a, b, output string) {
	p.AddOp(OpLessEq, []string{a, b}, []string{output}, nil, nil)
}

// LessEqScalar emits element-wise less-than-or-equal comparison against a scalar.
func (p *Program) LessEqScalar(a string, scalar float32, output string) {
	p.AddOp(OpLessEq, []string{a}, []string{output}, []float32{scalar}, nil)
}

// Equal emits element-wise equality comparison.
func (p *Program) Equal(a, b, output string) {
	p.AddOp(OpEqual, []string{a, b}, []string{output}, nil, nil)
}

// EqualScalar emits element-wise equality comparison against a scalar.
func (p *Program) EqualScalar(a string, scalar float32, output string) {
	p.AddOp(OpEqual, []string{a}, []string{output}, []float32{scalar}, nil)
}

// Softplus emits element-wise softplus: output = log(1 + exp(a)).
func (p *Program) Softplus(a, output string) {
	p.AddOp(OpSoftplus, []string{a}, []string{output}, nil, nil)
}

// ReLU emits element-wise rectified linear unit: output = max(a, 0).
func (p *Program) ReLU(a, output string) {
	p.AddOp(OpReLU, []string{a}, []string{output}, nil, nil)
}

// LeakyReLU emits element-wise leaky rectified linear unit.
func (p *Program) LeakyReLU(a, output string, negativeSlope float32) {
	p.AddOp(OpLeakyReLU, []string{a}, []string{output}, []float32{negativeSlope}, nil)
}

// XSAProject projects y orthogonal to v along the last dimension.
func (p *Program) XSAProject(y, v, output string) {
	p.AddOp(OpXSAProject, []string{y, v}, []string{output}, nil, nil)
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

// RandomNormal emits a fresh i.i.d. Gaussian tensor with the given shape,
// mean, and standard deviation. The tensor is sampled per forward pass using
// the MLX backend's random source, so successive forward calls see different
// values (this is the property V7n-B "random gaussian solo" relies on). The
// output is wrapped in stop_gradient, so autograd treats it as a constant.
// Caller is responsible for any reparameterization-trick gradient routing.
func (p *Program) RandomNormal(shape []int, mean, stddev float32, output string) {
	p.AddOp(OpRandomNormal, nil, []string{output}, []float32{mean, stddev}, shape)
}

// StopGradient emits an identity op whose reverse-mode gradient is zero.
func (p *Program) StopGradient(input, output string) {
	p.AddOp(OpStopGradient, []string{input}, []string{output}, nil, nil)
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
