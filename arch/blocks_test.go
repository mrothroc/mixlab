package arch

import (
	"reflect"
	"testing"
)

// --- Block weight counting ---

func TestBlockWeightCount(t *testing.T) {
	tests := []struct {
		spec BlockSpec
		want int
	}{
		{BlockSpec{Type: "plain", Heads: 4}, 7},
		{BlockSpec{Type: "plain", Heads: 4, QKGain: 5.25}, 8},
		{BlockSpec{Type: "swiglu"}, 4},
		{BlockSpec{Type: "mlp"}, 3},
		{BlockSpec{Type: "mamba"}, 4},
		{BlockSpec{Type: "token_blend"}, 1},
	}
	for _, tt := range tests {
		n, err := BlockWeightCount(tt.spec, false, false)
		if err != nil {
			t.Fatalf("BlockWeightCount(%q): %v", tt.spec.Type, err)
		}
		if n != tt.want {
			t.Fatalf("BlockWeightCount(%q) = %d, want %d", tt.spec.Type, n, tt.want)
		}
	}
}

func TestBlockWeightCount_WithBlockScalesAndResidMix(t *testing.T) {
	plain, err := BlockWeightCount(BlockSpec{Type: "plain", Heads: 4}, true, true)
	if err != nil {
		t.Fatalf("BlockWeightCount(plain scaled+mixed): %v", err)
	}
	if plain != 10 {
		t.Fatalf("plain count = %d, want 10", plain)
	}
	swiglu, err := BlockWeightCount(BlockSpec{Type: "swiglu"}, true, false)
	if err != nil {
		t.Fatalf("BlockWeightCount(swiglu scaled): %v", err)
	}
	if swiglu != 5 {
		t.Fatalf("swiglu count = %d, want 5", swiglu)
	}
}

func TestBlockWeightCountUnsupported(t *testing.T) {
	_, err := BlockWeightCount(BlockSpec{Type: "unknown_type"}, false, false)
	if err == nil {
		t.Fatal("expected error for unsupported block type")
	}
}

// --- Attention block emission ---

func TestEmitPlainAttentionIR(t *testing.T) {
	p := NewProgram(7)
	wi, err := emitPlainAttentionIR(p, "x", 0, 4, 0, 128, 64, 2, 0, DefaultFFNMultiplier, false)
	if err != nil {
		t.Fatalf("emitPlainAttentionIR: %v", err)
	}
	if wi != 7 {
		t.Fatalf("expected wi=7, got %d", wi)
	}
	hasRMSNorm := false
	hasRoPE := false
	hasCausalMask := false
	hasSiLU := false
	for _, op := range p.Ops {
		switch op.Code {
		case OpRMSNorm:
			hasRMSNorm = true
		case OpRoPE:
			hasRoPE = true
		case OpCausalMask:
			hasCausalMask = true
		case OpSiLU:
			hasSiLU = true
		}
	}
	if !hasRMSNorm {
		t.Error("missing RMSNorm op")
	}
	if !hasRoPE {
		t.Error("missing RoPE op")
	}
	if !hasCausalMask {
		t.Error("missing CausalMask op")
	}
	if !hasSiLU {
		t.Error("missing SiLU op (feed-forward tail)")
	}
}

func TestEmitPlainAttentionIR_WithBlockScales(t *testing.T) {
	p := NewProgram(9)
	wi, err := emitPlainAttentionIR(p, "x", 0, 4, 0, 128, 64, 2, 0, DefaultFFNMultiplier, true)
	if err != nil {
		t.Fatalf("emitPlainAttentionIR scaled: %v", err)
	}
	if wi != 9 {
		t.Fatalf("expected wi=9, got %d", wi)
	}
	if n := countOps(p, OpMul); n != 2 {
		t.Fatalf("expected 2 Mul ops for attn/mlp scales, got %d", n)
	}
}

func TestEmitPlainAttentionIR_RopeDims(t *testing.T) {
	p := NewProgram(7)
	if _, err := emitBlockIR(p, BlockSpec{Type: "plain", Heads: 4, RopeDims: 16}, "x", 0, 128, 64, 2, 1024, 0, nil, DefaultFFNMultiplier, false); err != nil {
		t.Fatalf("emitBlockIR: %v", err)
	}
	for _, op := range p.Ops {
		if op.Code == OpRoPE {
			if len(op.IntParams) != 3 || op.IntParams[0] != 64 || op.IntParams[1] != 32 || op.IntParams[2] != 16 {
				t.Fatalf("RoPE int params = %v, want [64 32 16]", op.IntParams)
			}
			return
		}
	}
	t.Fatal("missing RoPE op")
}

func TestEmitPlainAttentionIR_SkipAttentionPreservesWeights(t *testing.T) {
	p := NewProgram(7)
	wi, err := emitPlainAttentionIRWithOptions(p, "x", 0, 4, 0, 128, 64, 2, 0, DefaultFFNMultiplier, false, 0, true, 0, 0)
	if err != nil {
		t.Fatalf("emitPlainAttentionIRWithOptions skip attention: %v", err)
	}
	if wi != 7 {
		t.Fatalf("expected wi=7, got %d", wi)
	}
	if n := countOps(p, OpCausalMask); n != 0 {
		t.Fatalf("expected no CausalMask ops, got %d", n)
	}
	if n := countOps(p, OpSoftmax); n != 0 {
		t.Fatalf("expected no Softmax ops, got %d", n)
	}
	if n := countOps(p, OpRoPE); n != 0 {
		t.Fatalf("expected no RoPE ops, got %d", n)
	}
	if n := countOps(p, OpMatMul); n != 2 {
		t.Fatalf("expected 2 FFN MatMul ops, got %d", n)
	}
	if n := countOps(p, OpSiLU); n != 1 {
		t.Fatalf("expected 1 SiLU op, got %d", n)
	}
}

func TestEmitPlainAttentionIR_QKGainBeforeMaskAndSoftmax(t *testing.T) {
	p := NewProgram(8)
	wi, err := emitBlockIR(p, BlockSpec{Type: "plain", Heads: 4, QKGain: 5.25}, "x", 0, 128, 64, 2, 1024, 0, nil, DefaultFFNMultiplier, false)
	if err != nil {
		t.Fatalf("emitBlockIR qk_gain: %v", err)
	}
	if wi != 8 {
		t.Fatalf("expected wi=8, got %d", wi)
	}

	gainReshapeIdx := -1
	gainMulIdx := -1
	maskIdx := -1
	softmaxIdx := -1
	for i, op := range p.Ops {
		switch op.Code {
		case OpReshape:
			if len(op.Inputs) == 1 && op.Inputs[0] == "w4" && reflect.DeepEqual(op.IntParams, []int{1, 4, 1, 1}) {
				gainReshapeIdx = i
			}
		case OpMul:
			if len(op.Inputs) == 2 && op.Inputs[0] == "x_attn_0_scores_scaled" && op.Inputs[1] == "x_attn_0_scores_qk_gain" {
				gainMulIdx = i
			}
		case OpCausalMask:
			if maskIdx == -1 {
				maskIdx = i
			}
		case OpSoftmax:
			if softmaxIdx == -1 {
				softmaxIdx = i
			}
		}
	}
	if gainReshapeIdx == -1 {
		t.Fatal("missing qk_gain reshape from w4 to [1,4,1,1]")
	}
	if gainMulIdx == -1 {
		t.Fatal("missing qk_gain score multiply")
	}
	if gainReshapeIdx >= gainMulIdx || gainMulIdx >= maskIdx || maskIdx >= softmaxIdx {
		t.Fatalf("unexpected op order reshape=%d mul=%d mask=%d softmax=%d", gainReshapeIdx, gainMulIdx, maskIdx, softmaxIdx)
	}
}

func TestEmitMLPIR_LeakyReLUSquared(t *testing.T) {
	p := NewProgram(3)
	wi, err := emitBlockIR(p, BlockSpec{Type: "mlp", Activation: "leaky_relu_sq", LeakySlope: 0.25}, "x", 0, 64, 32, 1, 256, 0, nil, DefaultFFNMultiplier, false)
	if err != nil {
		t.Fatalf("emitBlockIR mlp: %v", err)
	}
	if wi != 3 {
		t.Fatalf("expected wi=3, got %d", wi)
	}

	leakyIdx := -1
	squareIdx := -1
	for i, op := range p.Ops {
		switch op.Code {
		case OpLeakyReLU:
			leakyIdx = i
			if len(op.FloatParams) != 1 || op.FloatParams[0] != 0.25 {
				t.Fatalf("LeakyReLU float params = %v, want [0.25]", op.FloatParams)
			}
		case OpSquare:
			squareIdx = i
		}
	}
	if leakyIdx == -1 {
		t.Fatal("missing LeakyReLU op")
	}
	if squareIdx == -1 {
		t.Fatal("missing Square op")
	}
	if leakyIdx > squareIdx {
		t.Fatalf("LeakyReLU op index %d should precede Square op index %d", leakyIdx, squareIdx)
	}
}

func TestEmitPlainAttentionIR_InvalidDims(t *testing.T) {
	p := NewProgram(7)
	_, err := emitPlainAttentionIR(p, "x", 0, 3, 0, 128, 64, 2, 0, DefaultFFNMultiplier, false)
	if err == nil {
		t.Fatal("expected error for D%H != 0")
	}
}

func TestEmitPlainAttentionIR_ZeroHeads(t *testing.T) {
	p := NewProgram(7)
	_, err := emitPlainAttentionIR(p, "x", 0, 0, 0, 128, 64, 2, 0, DefaultFFNMultiplier, false)
	if err == nil {
		t.Fatal("expected error for zero heads")
	}
}

func TestEmitPlainAttentionIR_GQAWeightCountLessThanMHA(t *testing.T) {
	mha, err := blockWeightShapes(BlockSpec{Type: "plain", Heads: 8}, 128, 64, 2, 1024, DefaultFFNMultiplier, false, false)
	if err != nil {
		t.Fatalf("blockWeightShapes(MHA): %v", err)
	}
	gqa, err := blockWeightShapes(BlockSpec{Type: "plain", Heads: 8, KVHeads: 4}, 128, 64, 2, 1024, DefaultFFNMultiplier, false, false)
	if err != nil {
		t.Fatalf("blockWeightShapes(GQA): %v", err)
	}

	countElems := func(metas []WeightMeta) int {
		total := 0
		for _, meta := range metas {
			size := 1
			for _, dim := range meta.Shape {
				size *= dim
			}
			total += size
		}
		return total
	}

	if got, want := gqa[2].Shape, []int{128, 64}; !reflect.DeepEqual(got, want) {
		t.Fatalf("wk shape = %v, want %v", got, want)
	}
	if got, want := gqa[3].Shape, []int{128, 64}; !reflect.DeepEqual(got, want) {
		t.Fatalf("wv shape = %v, want %v", got, want)
	}
	if countElems(gqa) >= countElems(mha) {
		t.Fatalf("expected GQA params < MHA params, got gqa=%d mha=%d", countElems(gqa), countElems(mha))
	}
}

func TestEmitPlainAttentionIR_GQABuilds(t *testing.T) {
	p := NewProgram(7)
	wi, err := emitPlainAttentionIR(p, "x", 0, 8, 4, 128, 64, 2, 0, DefaultFFNMultiplier, false)
	if err != nil {
		t.Fatalf("emitPlainAttentionIR(GQA): %v", err)
	}
	if wi != 7 {
		t.Fatalf("expected wi=7, got %d", wi)
	}
	if n := countOps(p, OpFull); n != 2 {
		t.Fatalf("expected 2 Full ops for K/V repeat, got %d", n)
	}
	if n := countOps(p, OpMul); n != 2 {
		t.Fatalf("expected 2 Mul ops for K/V repeat, got %d", n)
	}
}

func TestEmitPlainAttentionIR_KVHeadsEqualHeadsMatchesMHA(t *testing.T) {
	mha := NewProgram(7)
	if _, err := emitPlainAttentionIR(mha, "x", 0, 8, 0, 128, 64, 2, 0, DefaultFFNMultiplier, false); err != nil {
		t.Fatalf("emitPlainAttentionIR(MHA): %v", err)
	}
	same := NewProgram(7)
	if _, err := emitPlainAttentionIR(same, "x", 0, 8, 8, 128, 64, 2, 0, DefaultFFNMultiplier, false); err != nil {
		t.Fatalf("emitPlainAttentionIR(kv_heads=heads): %v", err)
	}
	if !reflect.DeepEqual(same.Ops, mha.Ops) {
		t.Fatal("expected kv_heads=heads to match standard MHA IR")
	}
}

func TestEmitPlainAttentionIR_InvalidKVHeads(t *testing.T) {
	p := NewProgram(7)
	_, err := emitPlainAttentionIR(p, "x", 0, 8, 3, 128, 64, 2, 0, DefaultFFNMultiplier, false)
	if err == nil {
		t.Fatal("expected error for invalid kv_heads")
	}
}

// --- SwiGLU block emission ---

func TestEmitSwiGLUIR(t *testing.T) {
	p := NewProgram(4)
	wi, err := emitSwiGLUIR(p, "x", 0, 0, DefaultFFNMultiplier, false)
	if err != nil {
		t.Fatalf("emitSwiGLUIR: %v", err)
	}
	if wi != 4 {
		t.Fatalf("expected wi=4, got %d", wi)
	}
	hasSigmoid := false
	hasMul := false
	for _, op := range p.Ops {
		if op.Code == OpSigmoid {
			hasSigmoid = true
		}
		if op.Code == OpMul {
			hasMul = true
		}
	}
	if !hasSigmoid {
		t.Error("missing Sigmoid op in SwiGLU")
	}
	if !hasMul {
		t.Error("missing Mul op in SwiGLU (gate * up)")
	}
}

func TestEmitSwiGLUIR_WithBlockScales(t *testing.T) {
	p := NewProgram(5)
	wi, err := emitSwiGLUIR(p, "x", 0, 0, DefaultFFNMultiplier, true)
	if err != nil {
		t.Fatalf("emitSwiGLUIR scaled: %v", err)
	}
	if wi != 5 {
		t.Fatalf("expected wi=5, got %d", wi)
	}
	if n := countOps(p, OpMul); n != 2 {
		t.Fatalf("expected 2 Mul ops with mlp scale, got %d", n)
	}
}

// --- Block dispatch ---

func TestEmitBlockIR_Dispatch(t *testing.T) {
	tests := []struct {
		spec   BlockSpec
		wantWi int
	}{
		{BlockSpec{Type: "plain", Heads: 4}, 7},
		{BlockSpec{Type: "swiglu"}, 4},
		{BlockSpec{Type: "mlp"}, 3},
	}
	for _, tt := range tests {
		p := NewProgram(tt.wantWi)
		wi, err := emitBlockIR(p, tt.spec, "x", 0, 128, 64, 1, 1024, 0, nil, DefaultFFNMultiplier, false)
		if err != nil {
			t.Fatalf("emitBlockIR(%q): %v", tt.spec.Type, err)
		}
		if wi != tt.wantWi {
			t.Fatalf("emitBlockIR(%q) wi=%d, want %d", tt.spec.Type, wi, tt.wantWi)
		}
	}
}

func TestEmitBlockIR_DefaultHeads(t *testing.T) {
	p := NewProgram(7)
	wi, err := emitBlockIR(p, BlockSpec{Type: "plain", Heads: 0}, "x", 0, 128, 64, 1, 1024, 0, nil, DefaultFFNMultiplier, false)
	if err != nil {
		t.Fatalf("emitBlockIR: %v", err)
	}
	if wi != 7 {
		t.Fatalf("expected wi=7, got %d", wi)
	}
}

// --- blockTypeKey ---

func TestBlockTypeKey(t *testing.T) {
	tests := []struct {
		spec BlockSpec
		want string
	}{
		{BlockSpec{Type: "plain"}, "plain"},
		{BlockSpec{Type: "SWIGLU"}, "swiglu"},
		{BlockSpec{Type: " Plain "}, "plain"},
	}
	for _, tt := range tests {
		got := blockTypeKey(tt.spec)
		if got != tt.want {
			t.Errorf("blockTypeKey(%q) = %q, want %q", tt.spec.Type, got, tt.want)
		}
	}
}

func TestEmitTokenBlendIR(t *testing.T) {
	p := NewProgram(1)
	wi, err := emitTokenBlendIR(p, "x", 0, 64, 32, 2, 0)
	if err != nil {
		t.Fatalf("emitTokenBlendIR: %v", err)
	}
	if wi != 1 {
		t.Fatalf("expected wi=1, got %d", wi)
	}

	hasSigmoid := false
	hasConcat := false
	hasFull := false
	for _, op := range p.Ops {
		switch op.Code {
		case OpSigmoid:
			hasSigmoid = true
		case OpConcat:
			hasConcat = true
		case OpFull:
			hasFull = true
		}
	}
	if !hasSigmoid {
		t.Error("missing Sigmoid op in token_blend")
	}
	if !hasConcat {
		t.Error("missing Concat op for token shift")
	}
	if !hasFull {
		t.Error("missing Full op for token shift/ones broadcast")
	}
}

func TestEmitTokenBlendIR_InvalidDims(t *testing.T) {
	p := NewProgram(1)
	if _, err := emitTokenBlendIR(p, "x", 0, 0, 32, 1, 0); err == nil {
		t.Fatal("expected error for D=0")
	}
	if _, err := emitTokenBlendIR(p, "x", 0, 64, 0, 1, 0); err == nil {
		t.Fatal("expected error for T=0")
	}
	if _, err := emitTokenBlendIR(p, "x", 0, 64, 32, 0, 0); err == nil {
		t.Fatal("expected error for B=0")
	}
}

func TestEmitBlockIR_TokenBlend(t *testing.T) {
	p := NewProgram(1)
	wi, err := emitBlockIR(p, BlockSpec{Type: "token_blend"}, "x", 0, 64, 32, 1, 1024, 0, nil, DefaultFFNMultiplier, false)
	if err != nil {
		t.Fatalf("emitBlockIR(token_blend): %v", err)
	}
	if wi != 1 {
		t.Fatalf("emitBlockIR(token_blend) wi=%d, want 1", wi)
	}
	if n := countOps(p, OpSigmoid); n != 1 {
		t.Fatalf("expected 1 Sigmoid op, got %d", n)
	}
}
