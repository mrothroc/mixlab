package gpu

import (
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

func TestIRToGPUOpCodeAlignment(t *testing.T) {
	// Verify that every ir.Op* code maps to the same gpu.Op* code.
	// The codes are intentionally aligned, but this test makes that explicit.
	cases := []struct {
		name  string
		irOp  int
		gpuOp int
	}{
		{"Embed", ir.OpEmbed, OpEmbed},
		{"MatMul", ir.OpMatMul, OpMatMul},
		{"Add", ir.OpAdd, OpAdd},
		{"Mul", ir.OpMul, OpMul},
		{"ScalarMul", ir.OpScalarMul, OpScalarMul},
		{"Sigmoid", ir.OpSigmoid, OpSigmoid},
		{"SiLU", ir.OpSiLU, OpSiLU},
		{"Softmax", ir.OpSoftmax, OpSoftmax},
		{"Reshape", ir.OpReshape, OpReshape},
		{"Transpose", ir.OpTranspose, OpTranspose},
		{"Slice", ir.OpSlice, OpSlice},
		{"Concat", ir.OpConcat, OpConcat},
		{"CausalMask", ir.OpCausalMask, OpCausalMask},
		{"CrossEntropy", ir.OpCrossEntropy, OpCrossEntropy},
		{"CrossEntropyPerToken", ir.OpCrossEntropyPerToken, OpCrossEntropyPerToken},
		{"MaskedCrossEntropy", ir.OpMaskedCrossEntropy, OpMaskedCrossEntropy},
		{"MaskedCrossEntropyPerToken", ir.OpMaskedCEPerToken, OpMaskedCEPerToken},
		{"Dropout", ir.OpDropout, OpDropout},
		{"Square", ir.OpSquare, OpSquare},
		{"Sub", ir.OpSub, OpSub},
		{"Div", ir.OpDiv, OpDiv},
		{"Where", ir.OpWhere, OpWhere},
		{"LessThan", ir.OpLessThan, OpLessThan},
		{"GreaterEq", ir.OpGreaterEq, OpGreaterEq},
		{"Arange", ir.OpArange, OpArange},
		{"MeanAxis", ir.OpMeanAxis, OpMeanAxis},
		{"Full", ir.OpFull, OpFull},
		{"RMSNorm", ir.OpRMSNorm, OpRMSNorm},
		{"RoPE", ir.OpRoPE, OpRoPE},
		{"Sqrt", ir.OpSqrt, OpSqrt},
		{"RSqrt", ir.OpRSqrt, OpRSqrt},
		{"Exp", ir.OpExp, OpExp},
		{"Outer", ir.OpOuter, OpOuter},
		{"GELU", ir.OpGELU, OpGELU},
		{"ReLU", ir.OpReLU, OpReLU},
		{"Tanh", ir.OpTanh, OpTanh},
		{"Scan", ir.OpScan, OpScan},
		{"GatherPositions", ir.OpGatherPositions, OpGatherPositions},
		{"ScatterPositions", ir.OpScatterPositions, OpScatterPositions},
		{"RoPEIndexed", ir.OpRoPEIndexed, OpRoPEIndexed},
		{"LeakyReLU", ir.OpLeakyReLU, OpLeakyReLU},
		{"XSAProject", ir.OpXSAProject, OpXSAProject},
		{"MatrixScan", ir.OpMatrixScan, OpMatrixScan},
		{"ScanTV", ir.OpScanTV, OpScanTV},
		{"Softplus", ir.OpSoftplus, OpSoftplus},
		{"GatedDeltaScan", ir.OpGatedDeltaScan, OpGatedDeltaScan},
		{"HGRN2Scan", ir.OpHGRN2Scan, OpHGRN2Scan},
		{"MLSTMScan", ir.OpMLSTMScan, OpMLSTMScan},
		{"DebertaRelativeBias", ir.OpDebertaRelativeBias, OpDebertaRelativeBias},
		{"CharFeatureBag", ir.OpCharFeatureBag, OpCharFeatureBag},
		{"MoEFeedForward", ir.OpMoEFeedForward, OpMoEFeedForward},
		{"MaskedSmoothL1", ir.OpMaskedSmoothL1, OpMaskedSmoothL1},
		{"ZLoss", ir.OpZLoss, OpZLoss},
		{"Log", ir.OpLog, OpLog},
		{"Reciprocal", ir.OpReciprocal, OpReciprocal},
		{"Pow", ir.OpPow, OpPow},
		{"Abs", ir.OpAbs, OpAbs},
		{"Clamp", ir.OpClamp, OpClamp},
		{"Minimum", ir.OpMinimum, OpMinimum},
		{"Maximum", ir.OpMaximum, OpMaximum},
		{"GreaterThan", ir.OpGreaterThan, OpGreaterThan},
		{"LessEq", ir.OpLessEq, OpLessEq},
		{"Equal", ir.OpEqual, OpEqual},
		{"LayerNorm", ir.OpLayerNorm, OpLayerNorm},
		{"SelectiveCausalMask", ir.OpSelectiveCausalMask, OpSelectiveCausalMask},
		{"SegmentAttentionMask", ir.OpSegmentAttentionMask, OpSegmentAttentionMask},
		{"BlockDiffusionMask", ir.OpBlockDiffusionMask, OpBlockDiffusionMask},
		{"GELUExact", ir.OpGELUExact, OpGELUExact},
		{"MaskedBCEWithLogits", ir.OpMaskedBCEWithLogits, OpMaskedBCEWithLogits},
		{"MaskedBinaryAccuracy", ir.OpMaskedBinaryAccuracy, OpMaskedBinaryAccuracy},
		{"EnergyPairwiseLoss", ir.OpEnergyPairwiseLoss, OpEnergyPairwiseLoss},
		{"EnergySpanPool", ir.OpEnergySpanPool, OpEnergySpanPool},
		{"EnergySpanPairwise", ir.OpEnergySpanPairwise, OpEnergySpanPairwise},
		{"SpanPLLPool", ir.OpSpanPLLPool, OpSpanPLLPool},
		{"SpanPLLPairwise", ir.OpSpanPLLPairwise, OpSpanPLLPairwise},
		{"MaskedDistillationKL", ir.OpMaskedDistillationKL, OpMaskedDistillationKL},
		{"MaskedSymmetricKL", ir.OpMaskedSymmetricKL, OpMaskedSymmetricKL},
		{"MaskedMarginPLL", ir.OpMaskedMarginPLL, OpMaskedMarginPLL},
		{"StopGradient", ir.OpStopGradient, OpStopGradient},
		{"FirstByteMaskedCrossEntropy", ir.OpFirstByteMaskedCE, OpFirstByteMaskedCE},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if tc.irOp != tc.gpuOp {
				t.Errorf("ir.Op%s=%d != gpu.Op%s=%d", tc.name, tc.irOp, tc.name, tc.gpuOp)
			}
			mapped, ok := irToGPUOp[tc.irOp]
			if !ok {
				t.Fatalf("ir op code %d not in irToGPUOp map", tc.irOp)
			}
			if mapped != tc.gpuOp {
				t.Errorf("irToGPUOp[%d]=%d, want %d", tc.irOp, mapped, tc.gpuOp)
			}
		})
	}
}

func TestLowerIRProgram_NilProgram(t *testing.T) {
	_, err := LowerIRProgram(nil)
	if err == nil {
		t.Fatal("expected error for nil program")
	}
}

func TestLowerIRProgram_NoWeights(t *testing.T) {
	prog := &ir.Program{NumWeights: 0}
	_, err := LowerIRProgram(prog)
	if err == nil {
		t.Fatal("expected error for zero weights")
	}
}

func TestLowerIRProgram_UnsupportedOp(t *testing.T) {
	prog := ir.NewProgram(1)
	prog.DeclareInput("tokens", ir.TensorInt32, []int{1, 4})
	prog.DeclareOutput("loss", ir.TensorFloat32, []int{1})
	// Use an op code that is not in the mapping.
	prog.AddOp(9999, []string{"tokens"}, []string{"loss"}, nil, nil)

	_, err := LowerIRProgram(prog)
	if err == nil {
		t.Fatal("expected error for unsupported op code")
	}
}

func TestLowerIRProgram_BasicProgram(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}
	// Build a minimal IR program: embed + matmul + cross_entropy.
	prog := ir.NewProgram(2)
	prog.DeclareInput("tokens", ir.TensorInt32, []int{1, 4})
	prog.DeclareInput("targets", ir.TensorInt32, []int{1, 4})
	prog.DeclareOutput("loss", ir.TensorFloat32, []int{1})

	prog.Embed("w0", "tokens", "emb")
	prog.MatMul("emb", "w1", "logits")
	prog.CrossEntropy("logits", "targets", "loss")

	gpuProg, err := LowerIRProgram(prog)
	if err != nil {
		t.Fatalf("LowerIRProgram: %v", err)
	}
	defer gpuProg.Destroy()

	if gpuProg.operationCount() != 3 {
		t.Errorf("expected 3 ops, got %d", gpuProg.operationCount())
	}

	opTypes := gpuProg.operationTypes()
	if len(opTypes) != 3 {
		t.Fatalf("expected 3 op types, got %d", len(opTypes))
	}
	if opTypes[0] != OpEmbed {
		t.Errorf("op[0] = %d, want %d (Embed)", opTypes[0], OpEmbed)
	}
	if opTypes[1] != OpMatMul {
		t.Errorf("op[1] = %d, want %d (MatMul)", opTypes[1], OpMatMul)
	}
	if opTypes[2] != OpCrossEntropy {
		t.Errorf("op[2] = %d, want %d (CrossEntropy)", opTypes[2], OpCrossEntropy)
	}
}

func TestIRToGPUDTypeAlignment(t *testing.T) {
	if irToGPUDType[ir.TensorInt32] != TensorInt32 {
		t.Errorf("TensorInt32 mismatch")
	}
	if irToGPUDType[ir.TensorFloat32] != TensorFloat32 {
		t.Errorf("TensorFloat32 mismatch")
	}
}
