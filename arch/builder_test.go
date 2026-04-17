package arch

import (
	"testing"
)

func requireTailOpCodes(t *testing.T, prog *Program, want ...int) {
	t.Helper()
	if len(prog.Ops) < len(want) {
		t.Fatalf("program has %d ops, want at least %d", len(prog.Ops), len(want))
	}
	start := len(prog.Ops) - len(want)
	for i, code := range want {
		if got := prog.Ops[start+i].Code; got != code {
			t.Fatalf("tail op[%d]=%d want %d", start+i, got, code)
		}
	}
}

// --- CountWeights ---

func TestCountWeights_Plain3L(t *testing.T) {
	blocks := []BlockSpec{
		{Type: "plain", Heads: 4},
		{Type: "swiglu"},
		{Type: "plain", Heads: 4},
		{Type: "swiglu"},
		{Type: "plain", Heads: 4},
		{Type: "swiglu"},
	}
	n, err := CountWeights(DefaultFFNMultiplier, false, false, false, false, blocks)
	if err != nil {
		t.Fatalf("CountWeights: %v", err)
	}
	if n != 36 {
		t.Fatalf("expected 36 weights, got %d", n)
	}
}

func TestCountWeights_UnsupportedType(t *testing.T) {
	blocks := []BlockSpec{{Type: "garch"}}
	_, err := CountWeights(DefaultFFNMultiplier, false, false, false, false, blocks)
	if err == nil {
		t.Fatal("expected error for unsupported type")
	}
}

func TestCountWeights_RWKV(t *testing.T) {
	blocks := []BlockSpec{
		{Type: "rwkv"},
		{Type: "rwkv"},
		{Type: "swiglu"},
	}
	n, err := CountWeights(DefaultFFNMultiplier, false, false, false, false, blocks)
	if err != nil {
		t.Fatalf("CountWeights: %v", err)
	}
	if n != 27 {
		t.Fatalf("expected 27 weights, got %d", n)
	}
}

func TestCountWeights_RetNet(t *testing.T) {
	blocks := []BlockSpec{
		{Type: "retnet", Heads: 4},
		{Type: "retnet", Heads: 4},
		{Type: "swiglu"},
	}
	n, err := CountWeights(DefaultFFNMultiplier, false, false, false, false, blocks)
	if err != nil {
		t.Fatalf("CountWeights: %v", err)
	}
	if n != 23 {
		t.Fatalf("expected 23 weights, got %d", n)
	}
}

func TestCountWeights_WithMamba(t *testing.T) {
	blocks := []BlockSpec{
		{Type: "mamba"},
		{Type: "plain", Heads: 4},
		{Type: "swiglu"},
	}
	n, err := CountWeights(DefaultFFNMultiplier, false, false, false, false, blocks)
	if err != nil {
		t.Fatalf("CountWeights: %v", err)
	}
	if n != 18 {
		t.Fatalf("expected 18 weights, got %d", n)
	}
}

func TestCountWeightsWithBigram_IncreasesAsExpected(t *testing.T) {
	blocks := []BlockSpec{{Type: "plain", Heads: 4}}
	n, err := CountWeightsWithBigram(128, DefaultFFNMultiplier, false, false, false, false, 257, 32, blocks)
	if err != nil {
		t.Fatalf("CountWeightsWithBigram: %v", err)
	}
	if n != 13 {
		t.Fatalf("expected 13 weights, got %d", n)
	}

	nNoProj, err := CountWeightsWithBigram(128, DefaultFFNMultiplier, false, false, false, false, 257, 128, blocks)
	if err != nil {
		t.Fatalf("CountWeightsWithBigram no-proj: %v", err)
	}
	if nNoProj != 12 {
		t.Fatalf("expected 12 weights without projection, got %d", nNoProj)
	}
}

func TestCountWeights_TiedEmbeddingsDropsHeadWeight(t *testing.T) {
	blocks := []BlockSpec{{Type: "plain", Heads: 4}}
	untied, err := CountWeights(DefaultFFNMultiplier, false, false, false, false, blocks)
	if err != nil {
		t.Fatalf("CountWeights untied: %v", err)
	}
	tied, err := CountWeights(DefaultFFNMultiplier, true, false, false, false, blocks)
	if err != nil {
		t.Fatalf("CountWeights tied: %v", err)
	}
	if tied != untied-1 {
		t.Fatalf("expected tied weights=%d, got %d", untied-1, tied)
	}
}

func TestCountWeights_UNetAddsSkipWeights(t *testing.T) {
	blocks := []BlockSpec{
		{Type: "plain", Heads: 4},
		{Type: "plain", Heads: 4},
		{Type: "plain", Heads: 4},
		{Type: "plain", Heads: 4},
		{Type: "plain", Heads: 4},
	}
	n, err := CountWeights(DefaultFFNMultiplier, false, false, false, true, blocks)
	if err != nil {
		t.Fatalf("CountWeights unet: %v", err)
	}
	if n != 40 {
		t.Fatalf("expected 40 weights, got %d", n)
	}
}

// --- BuildIRProgram: sequential model ---

func TestBuildIRProgram_Plain3L(t *testing.T) {
	blocks := []BlockSpec{
		{Type: "plain", Heads: 4},
		{Type: "swiglu"},
		{Type: "plain", Heads: 4},
		{Type: "swiglu"},
		{Type: "plain", Heads: 4},
		{Type: "swiglu"},
	}
	prog, err := BuildIRProgram(128, 1024, 128, 1, DefaultFFNMultiplier, false, false, false, false, 0, blocks)
	if err != nil {
		t.Fatalf("BuildIRProgram: %v", err)
	}

	if prog.NumWeights != 36 {
		t.Fatalf("expected 36 weights, got %d", prog.NumWeights)
	}
	if len(prog.Inputs) != 2 {
		t.Fatalf("expected 2 inputs, got %d", len(prog.Inputs))
	}
	if prog.Inputs[0].Name != "tokens" {
		t.Fatalf("expected first input 'tokens', got %q", prog.Inputs[0].Name)
	}
	if prog.Inputs[1].Name != "targets" {
		t.Fatalf("expected second input 'targets', got %q", prog.Inputs[1].Name)
	}
	if len(prog.Outputs) != 3 || prog.Outputs[0].Name != "loss" || prog.Outputs[1].Name != "x_hidden" || prog.Outputs[2].Name != "logits" {
		t.Fatalf("expected outputs [loss x_hidden logits], got %+v", prog.Outputs)
	}
	if len(prog.Ops) < 10 {
		t.Fatalf("program has too few ops: %d", len(prog.Ops))
	}
	if prog.Ops[0].Code != OpEmbed {
		t.Fatalf("first op should be Embed, got %d", prog.Ops[0].Code)
	}
	last := prog.Ops[len(prog.Ops)-1]
	if last.Code != OpCrossEntropy {
		t.Fatalf("last op should be CrossEntropy, got %d", last.Code)
	}
}

func TestBuildIRProgramWithBigram_AddsInputAndOps(t *testing.T) {
	blocks := []BlockSpec{{Type: "plain", Heads: 4}}
	prog, err := BuildIRProgramWithBigram(64, 256, 16, 2, DefaultFFNMultiplier, false, false, false, false, 257, 32, 0, blocks)
	if err != nil {
		t.Fatalf("BuildIRProgramWithBigram: %v", err)
	}
	if prog.NumWeights != 13 {
		t.Fatalf("expected 13 weights, got %d", prog.NumWeights)
	}
	if len(prog.Inputs) != 3 {
		t.Fatalf("expected 3 inputs, got %d", len(prog.Inputs))
	}
	if prog.Inputs[2].Name != "bigram_ids" {
		t.Fatalf("expected third input bigram_ids, got %q", prog.Inputs[2].Name)
	}
	if prog.Inputs[2].Shape[0] != 2 || prog.Inputs[2].Shape[1] != 16 {
		t.Fatalf("unexpected bigram input shape %v", prog.Inputs[2].Shape)
	}

	hasBigramEmbed := false
	hasBigramProj := false
	hasBigramScale := false
	hasBigramAdd := false
	for _, op := range prog.Ops {
		if op.Code == OpEmbed && len(op.Inputs) == 2 && op.Inputs[1] == "bigram_ids" {
			hasBigramEmbed = true
		}
		if op.Code == OpMatMul && len(op.Inputs) == 2 && op.Inputs[0] == "bigram_flat" && op.Inputs[1] == "w4" {
			hasBigramProj = true
		}
		if op.Code == OpMul && len(op.Inputs) == 2 && op.Inputs[0] == "bigram_proj" && op.Inputs[1] == "w5" {
			hasBigramScale = true
		}
		if op.Code == OpAdd && len(op.Inputs) == 2 && op.Inputs[0] == "x_tok" && op.Inputs[1] == "bigram_scaled" {
			hasBigramAdd = true
		}
	}
	if !hasBigramEmbed {
		t.Fatal("missing bigram embedding op")
	}
	if !hasBigramProj {
		t.Fatal("missing bigram projection op")
	}
	if !hasBigramScale {
		t.Fatal("missing learned bigram scale multiply op")
	}
	if !hasBigramAdd {
		t.Fatal("missing bigram residual add op")
	}
}

func TestBuildIRProgramWithBigram_DisabledLeavesInputsUnchanged(t *testing.T) {
	blocks := []BlockSpec{{Type: "plain", Heads: 4}}
	prog, err := BuildIRProgramWithBigram(64, 256, 16, 1, DefaultFFNMultiplier, false, false, false, false, 0, 0, 0, blocks)
	if err != nil {
		t.Fatalf("BuildIRProgramWithBigram: %v", err)
	}
	if len(prog.Inputs) != 2 {
		t.Fatalf("expected 2 inputs when disabled, got %d", len(prog.Inputs))
	}
	for _, in := range prog.Inputs {
		if in.Name == "bigram_ids" {
			t.Fatal("unexpected bigram_ids input when disabled")
		}
	}
}

func TestBuildIRProgram_TiedEmbeddingsUsesTransposeHead(t *testing.T) {
	blocks := []BlockSpec{{Type: "plain", Heads: 4}}
	prog, err := BuildIRProgram(64, 256, 16, 1, DefaultFFNMultiplier, true, false, false, false, 0, blocks)
	if err != nil {
		t.Fatalf("BuildIRProgram: %v", err)
	}
	if prog.NumWeights != 9 {
		t.Fatalf("expected 9 weights with tied embeddings, got %d", prog.NumWeights)
	}

	hasTranspose := false
	hasTiedMatMul := false
	for _, op := range prog.Ops {
		if op.Code == OpTranspose && len(op.Inputs) == 1 && op.Inputs[0] == "w0" && len(op.Outputs) == 1 && op.Outputs[0] == "tied_head" {
			hasTranspose = true
		}
		if op.Code == OpMatMul && len(op.Inputs) == 2 && op.Inputs[0] == "x_final_norm" && op.Inputs[1] == "tied_head" {
			hasTiedMatMul = true
		}
		if op.Code == OpMatMul && len(op.Inputs) == 2 && op.Inputs[1] == "w1" {
			t.Fatal("unexpected separate head matmul in tied-embedding program")
		}
	}
	if !hasTranspose {
		t.Fatal("missing transpose of embedding table for tied head")
	}
	if !hasTiedMatMul {
		t.Fatal("missing tied head matmul")
	}
}

func TestBuildIRProgram_ResidMixAddsPreludeAndX0(t *testing.T) {
	blocks := []BlockSpec{{Type: "plain", Heads: 4}}
	prog, err := BuildIRProgram(64, 256, 16, 1, DefaultFFNMultiplier, false, false, true, false, 0, blocks)
	if err != nil {
		t.Fatalf("BuildIRProgram resid_mix: %v", err)
	}
	if prog.NumWeights != 11 {
		t.Fatalf("expected 11 weights with resid_mix, got %d", prog.NumWeights)
	}

	hasX0Copy := false
	sliceCount := 0
	for _, op := range prog.Ops {
		if op.Code == OpScalarMul && len(op.Inputs) == 1 && op.Inputs[0] == "x" && len(op.Outputs) == 1 && op.Outputs[0] == "x0" {
			hasX0Copy = true
		}
		if op.Code == OpSlice && len(op.Inputs) == 1 && op.Inputs[0] == "w3" {
			sliceCount++
		}
	}
	if !hasX0Copy {
		t.Fatal("missing x0 copy before first block")
	}
	if sliceCount != 2 {
		t.Fatalf("expected 2 slices from resid_mix weight, got %d", sliceCount)
	}
}

func TestBuildIRProgram_UNetAddsSkipConnections(t *testing.T) {
	blocks := []BlockSpec{
		{Type: "plain", Heads: 4},
		{Type: "plain", Heads: 4},
	}
	prog, err := BuildIRProgram(64, 256, 16, 1, DefaultFFNMultiplier, false, false, false, true, 0, blocks)
	if err != nil {
		t.Fatalf("BuildIRProgram unet: %v", err)
	}
	if prog.NumWeights != 18 {
		t.Fatalf("expected 18 weights with unet, got %d", prog.NumWeights)
	}
	hasEncCopy := false
	hasSkipMul := false
	for _, op := range prog.Ops {
		if op.Code == OpScalarMul && len(op.Inputs) == 1 && op.Inputs[0] == "x" && len(op.Outputs) == 1 && op.Outputs[0] == "enc_0" {
			hasEncCopy = true
		}
		if op.Code == OpMul && len(op.Inputs) == 2 && op.Inputs[0] == "enc_0" && op.Inputs[1] == "w10" {
			hasSkipMul = true
		}
	}
	if !hasEncCopy {
		t.Fatal("missing encoder activation copy")
	}
	if !hasSkipMul {
		t.Fatal("missing skip-weight multiply before decoder block")
	}
}

// --- Error cases ---

func TestBuildIRProgram_InvalidShape(t *testing.T) {
	blocks := []BlockSpec{{Type: "plain", Heads: 4}}
	_, err := BuildIRProgram(128, 1024, 128, 0, DefaultFFNMultiplier, false, false, false, false, 0, blocks)
	if err == nil {
		t.Fatal("expected error for B=0")
	}
}

func TestBuildIRProgram_InvalidModelDim(t *testing.T) {
	blocks := []BlockSpec{{Type: "plain", Heads: 4}}
	_, err := BuildIRProgram(0, 1024, 128, 1, DefaultFFNMultiplier, false, false, false, false, 0, blocks)
	if err == nil {
		t.Fatal("expected error for D=0")
	}
}

func TestBuildIRProgram_InvalidVocabSize(t *testing.T) {
	blocks := []BlockSpec{{Type: "plain", Heads: 4}}
	_, err := BuildIRProgram(128, 0, 128, 1, DefaultFFNMultiplier, false, false, false, false, 0, blocks)
	if err == nil {
		t.Fatal("expected error for V=0")
	}
}

func TestBuildIRProgram_UnsupportedBlockType(t *testing.T) {
	blocks := []BlockSpec{{Type: "unknown_type"}}
	_, err := BuildIRProgram(128, 1024, 128, 1, DefaultFFNMultiplier, false, false, false, false, 0, blocks)
	if err == nil {
		t.Fatal("expected error for unsupported block type")
	}
}

// --- Stream emission ---

func TestEmitStreamIR(t *testing.T) {
	specs := []BlockSpec{
		{Type: "plain", Heads: 4},
		{Type: "swiglu"},
		{Type: "plain", Heads: 4},
	}
	p := NewProgram(18)
	opIdx := 0
	wi, err := emitStreamIR(p, specs, "x", "x", 0, 128, 64, 1, 1024, &opIdx, DefaultFFNMultiplier, false, false)
	if err != nil {
		t.Fatalf("emitStreamIR: %v", err)
	}
	if wi != 18 {
		t.Fatalf("expected wi=18, got %d", wi)
	}
	if opIdx != 3 {
		t.Fatalf("expected opIdx=3, got %d", opIdx)
	}
}

// --- Batch size > 1 ---

func TestBuildIRProgram_MultiBatch(t *testing.T) {
	blocks := []BlockSpec{
		{Type: "plain", Heads: 4},
		{Type: "swiglu"},
	}
	prog, err := BuildIRProgram(128, 1024, 64, 4, DefaultFFNMultiplier, false, false, false, false, 0, blocks)
	if err != nil {
		t.Fatalf("BuildIRProgram with B=4: %v", err)
	}
	if prog.Inputs[1].Shape[0] != 256 {
		t.Fatalf("expected targets shape [256], got %v", prog.Inputs[1].Shape)
	}
}

// --- Op count verification for specific configs ---

func TestPlain3L_OpCounts(t *testing.T) {
	blocks := []BlockSpec{
		{Type: "plain", Heads: 4},
		{Type: "swiglu"},
		{Type: "plain", Heads: 4},
		{Type: "swiglu"},
		{Type: "plain", Heads: 4},
		{Type: "swiglu"},
	}
	prog, err := BuildIRProgram(128, 1024, 128, 1, DefaultFFNMultiplier, false, false, false, false, 0, blocks)
	if err != nil {
		t.Fatalf("BuildIRProgram: %v", err)
	}

	if n := countOps(prog, OpRoPE); n != 3 {
		t.Errorf("expected 3 RoPE ops, got %d", n)
	}
	if n := countOps(prog, OpCausalMask); n != 3 {
		t.Errorf("expected 3 CausalMask ops, got %d", n)
	}
	if n := countOps(prog, OpSigmoid); n != 3 {
		t.Errorf("expected 3 Sigmoid ops, got %d", n)
	}
	if n := countOps(prog, OpSiLU); n != 3 {
		t.Errorf("expected 3 SiLU ops, got %d", n)
	}
	if n := countOps(prog, OpEmbed); n != 1 {
		t.Errorf("expected 1 Embed op, got %d", n)
	}
	if n := countOps(prog, OpCrossEntropy); n != 1 {
		t.Errorf("expected 1 CrossEntropy op, got %d", n)
	}
	if n := countOps(prog, OpRMSNorm); n != 7 {
		t.Errorf("expected 7 RMSNorm ops, got %d", n)
	}
}

func TestBuildIRProgram_Perceiver(t *testing.T) {
	blocks := []BlockSpec{
		{Type: "perceiver", Heads: 4, NumLatents: 32},
		{Type: "perceiver", Heads: 4, NumLatents: 32},
	}
	prog, err := BuildIRProgram(128, 1024, 128, 1, DefaultFFNMultiplier, false, false, false, false, 0, blocks)
	if err != nil {
		t.Fatalf("BuildIRProgram: %v", err)
	}
	if prog.NumWeights != 33 {
		t.Fatalf("expected 33 weights, got %d", prog.NumWeights)
	}
	if n := countOps(prog, OpSoftmax); n != 6 {
		t.Errorf("expected 6 Softmax ops, got %d", n)
	}
	if n := countOps(prog, OpScalarMul); n < 2 {
		t.Errorf("expected at least 2 ScalarMul ops, got %d", n)
	}
}

func TestBuildIRProgram_Bottleneck(t *testing.T) {
	blocks := []BlockSpec{
		{Type: "bottleneck", Heads: 4, NumLatents: 4},
		{Type: "swiglu"},
	}
	prog, err := BuildIRProgram(128, 1024, 128, 1, DefaultFFNMultiplier, false, false, false, false, 0, blocks)
	if err != nil {
		t.Fatalf("BuildIRProgram: %v", err)
	}
	if prog.NumWeights != 22 {
		t.Fatalf("expected 22 weights, got %d", prog.NumWeights)
	}
}

func TestBuildIRProgram_BottleneckDefaultLatents(t *testing.T) {
	blocks := []BlockSpec{
		{Type: "bottleneck", Heads: 4},
	}
	prog, err := BuildIRProgram(128, 1024, 128, 1, DefaultFFNMultiplier, false, false, false, false, 0, blocks)
	if err != nil {
		t.Fatalf("BuildIRProgram: %v", err)
	}
	if prog.NumWeights != 18 {
		t.Fatalf("expected 18 weights, got %d", prog.NumWeights)
	}
}

func TestBuildIRProgram_RWKV(t *testing.T) {
	blocks := []BlockSpec{
		{Type: "rwkv"},
		{Type: "rwkv"},
	}
	prog, err := BuildIRProgram(64, 1024, 128, 1, DefaultFFNMultiplier, false, false, false, false, 0, blocks)
	if err != nil {
		t.Fatalf("BuildIRProgram: %v", err)
	}

	if prog.NumWeights != 23 {
		t.Fatalf("expected 23 weights, got %d", prog.NumWeights)
	}
	if prog.Ops[0].Code != OpEmbed {
		t.Fatalf("first op should be Embed, got %d", prog.Ops[0].Code)
	}
	last := prog.Ops[len(prog.Ops)-1]
	if last.Code != OpCrossEntropy {
		t.Fatalf("last op should be CrossEntropy, got %d", last.Code)
	}
	if n := countOps(prog, OpConcat); n != 4 {
		t.Errorf("expected 4 Concat ops, got %d", n)
	}
	if n := countOps(prog, OpScan); n != 4 {
		t.Errorf("expected 4 Scan ops, got %d", n)
	}
}

func TestBuildIRProgram_RetNet(t *testing.T) {
	blocks := []BlockSpec{
		{Type: "retnet", Heads: 4},
		{Type: "swiglu"},
		{Type: "retnet", Heads: 4},
		{Type: "swiglu"},
	}
	prog, err := BuildIRProgram(128, 1024, 128, 1, DefaultFFNMultiplier, false, false, false, false, 0, blocks)
	if err != nil {
		t.Fatalf("BuildIRProgram: %v", err)
	}

	if prog.NumWeights != 27 {
		t.Fatalf("expected 27 weights, got %d", prog.NumWeights)
	}
	if len(prog.Inputs) != 2 {
		t.Fatalf("expected 2 inputs, got %d", len(prog.Inputs))
	}
	if len(prog.Outputs) != 3 || prog.Outputs[0].Name != "loss" || prog.Outputs[1].Name != "x_hidden" || prog.Outputs[2].Name != "logits" {
		t.Fatalf("expected outputs [loss x_hidden logits], got %+v", prog.Outputs)
	}
	if n := countOps(prog, OpCausalMask); n != 2 {
		t.Errorf("expected 2 CausalMask ops, got %d", n)
	}
	if n := countOps(prog, OpRMSNorm); n != 5 {
		t.Errorf("expected 5 RMSNorm ops, got %d", n)
	}
	if prog.Ops[0].Code != OpEmbed {
		t.Fatalf("first op should be Embed, got %d", prog.Ops[0].Code)
	}
	last := prog.Ops[len(prog.Ops)-1]
	if last.Code != OpCrossEntropy {
		t.Fatalf("last op should be CrossEntropy, got %d", last.Code)
	}
}

func TestBuildIRProgram_MambaSequential(t *testing.T) {
	blocks := []BlockSpec{
		{Type: "mamba"},
		{Type: "mamba"},
		{Type: "swiglu"},
	}
	prog, err := BuildIRProgram(128, 1024, 64, 1, DefaultFFNMultiplier, false, false, false, false, 0, blocks)
	if err != nil {
		t.Fatalf("BuildIRProgram: %v", err)
	}
	if prog.NumWeights != 15 {
		t.Fatalf("expected 15 weights, got %d", prog.NumWeights)
	}
	if n := countOps(prog, OpScan); n != 2 {
		t.Errorf("expected 2 OpScan ops, got %d", n)
	}
	if prog.Ops[0].Code != OpEmbed {
		t.Errorf("first op should be Embed, got %d", prog.Ops[0].Code)
	}
	if last := prog.Ops[len(prog.Ops)-1]; last.Code != OpCrossEntropy {
		t.Errorf("last op should be CrossEntropy, got %d", last.Code)
	}
}

func TestBuildIRProgram_MambaWithCustomInnerDim(t *testing.T) {
	blocks := []BlockSpec{
		{Type: "mamba", InnerDim: 64},
	}
	prog, err := BuildIRProgram(128, 1024, 64, 1, DefaultFFNMultiplier, false, false, false, false, 0, blocks)
	if err != nil {
		t.Fatalf("BuildIRProgram: %v", err)
	}
	if prog.NumWeights != 7 {
		t.Fatalf("expected 7 weights, got %d", prog.NumWeights)
	}
	for _, op := range prog.Ops {
		if op.Code == OpScan && op.IntParams[2] != 64 {
			t.Errorf("OpScan inner = %d, want 64", op.IntParams[2])
		}
	}
}

func TestBuildIRProgram_LogitSoftcapEnabled_AppendsTailOps(t *testing.T) {
	blocks := []BlockSpec{{Type: "plain", Heads: 4}}
	prog, err := BuildIRProgram(128, 1024, 128, 1, DefaultFFNMultiplier, false, false, false, false, 12.5, blocks)
	if err != nil {
		t.Fatalf("BuildIRProgram: %v", err)
	}
	if prog.NumWeights != 10 {
		t.Fatalf("expected 10 weights, got %d", prog.NumWeights)
	}
	if n := countOps(prog, OpTanh); n != 1 {
		t.Fatalf("expected 1 Tanh op, got %d", n)
	}
	requireTailOpCodes(t, prog, OpRMSNorm, OpReshape, OpMatMul, OpScalarMul, OpTanh, OpScalarMul, OpScalarMul, OpCrossEntropy)
}

func TestBuildIRProgram_LogitSoftcapDisabled_HasNoTanh(t *testing.T) {
	blocks := []BlockSpec{{Type: "plain", Heads: 4}}
	prog, err := BuildIRProgram(128, 1024, 128, 1, DefaultFFNMultiplier, false, false, false, false, 0, blocks)
	if err != nil {
		t.Fatalf("BuildIRProgram: %v", err)
	}
	if prog.NumWeights != 10 {
		t.Fatalf("expected 10 weights, got %d", prog.NumWeights)
	}
	if n := countOps(prog, OpTanh); n != 0 {
		t.Fatalf("expected 0 Tanh ops, got %d", n)
	}
	requireTailOpCodes(t, prog, OpRMSNorm, OpReshape, OpMatMul, OpCrossEntropy)
}
