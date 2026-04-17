package arch

import "testing"

func TestBlockWeightCount_Perceiver(t *testing.T) {
	for _, typ := range []string{"perceiver", "bottleneck"} {
		n, err := BlockWeightCount(BlockSpec{Type: typ, Heads: 4, NumLatents: 16}, false, false)
		if err != nil {
			t.Fatalf("BlockWeightCount(%q): %v", typ, err)
		}
		if n != 15 {
			t.Fatalf("BlockWeightCount(%q) = %d, want 15", typ, n)
		}
	}
}

func TestBlockWeightCount_RWKV(t *testing.T) {
	n, err := BlockWeightCount(BlockSpec{Type: "rwkv"}, false, false)
	if err != nil {
		t.Fatalf("BlockWeightCount(rwkv): %v", err)
	}
	if n != 10 {
		t.Fatalf("BlockWeightCount(rwkv) = %d, want 10", n)
	}
}

func TestBlockWeightCount_RetNet(t *testing.T) {
	n, err := BlockWeightCount(BlockSpec{Type: "retnet", Heads: 4}, false, false)
	if err != nil {
		t.Fatalf("BlockWeightCount(retnet): %v", err)
	}
	if n != 8 {
		t.Fatalf("expected 8 weights for retnet, got %d", n)
	}
}

func TestBlockWeightCount_CrossAttention(t *testing.T) {
	n, err := BlockWeightCount(BlockSpec{Type: "cross_attention", Heads: 4, SourceStream: "low"}, false, false)
	if err != nil {
		t.Fatalf("BlockWeightCount(cross_attention): %v", err)
	}
	if n != 7 {
		t.Fatalf("expected 7, got %d", n)
	}
}

// --- Perceiver block emission ---

func TestEmitPerceiverIR(t *testing.T) {
	p := NewProgram(15)
	wi, err := emitPerceiverIR(p, "x", 0, 4, 8, 128, 64, 2, 0)
	if err != nil {
		t.Fatalf("emitPerceiverIR: %v", err)
	}
	if wi != 15 {
		t.Fatalf("expected wi=15, got %d", wi)
	}

	hasLatentRepeat := false
	hasRoPE := false
	hasCausalMask := false
	hasSiLU := false
	hasSoftmax := false
	for _, op := range p.Ops {
		switch op.Code {
		case OpConcat, OpScalarMul:
			hasLatentRepeat = true
		case OpRoPE:
			hasRoPE = true
		case OpCausalMask:
			hasCausalMask = true
		case OpSiLU:
			hasSiLU = true
		case OpSoftmax:
			hasSoftmax = true
		}
	}
	if !hasLatentRepeat {
		t.Error("missing latent repeat sequence")
	}
	if !hasRoPE {
		t.Error("missing RoPE op (self-attention on latents)")
	}
	if !hasCausalMask {
		t.Error("missing CausalMask op (self-attention on latents)")
	}
	if !hasSiLU {
		t.Error("missing SiLU op (feed-forward)")
	}
	if !hasSoftmax {
		t.Error("missing Softmax op (attention)")
	}
}

func TestEmitPerceiverIR_InvalidDims(t *testing.T) {
	p := NewProgram(15)
	_, err := emitPerceiverIR(p, "x", 0, 3, 8, 128, 64, 2, 0)
	if err == nil {
		t.Fatal("expected error for D%H != 0")
	}
}

func TestEmitPerceiverIR_ZeroLatents(t *testing.T) {
	p := NewProgram(15)
	_, err := emitPerceiverIR(p, "x", 0, 4, 0, 128, 64, 2, 0)
	if err == nil {
		t.Fatal("expected error for zero latents")
	}
}

func TestEmitPerceiverIR_SoftmaxCount(t *testing.T) {
	p := NewProgram(15)
	_, err := emitPerceiverIR(p, "x", 0, 4, 8, 128, 64, 2, 0)
	if err != nil {
		t.Fatalf("emitPerceiverIR: %v", err)
	}
	if n := countOps(p, OpSoftmax); n != 3 {
		t.Errorf("expected 3 Softmax ops (cross + self + broadcast), got %d", n)
	}
}

func TestEmitBlockIR_PerceiverDispatch(t *testing.T) {
	tests := []struct {
		spec   BlockSpec
		wantWi int
	}{
		{BlockSpec{Type: "perceiver", Heads: 4, NumLatents: 16}, 15},
		{BlockSpec{Type: "bottleneck", Heads: 4, NumLatents: 4}, 15},
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

// --- RWKV block ---

func TestEmitRWKVIR(t *testing.T) {
	p := NewProgram(10)
	wi, err := emitRWKVIR(p, "x", 0, 64, 32, 2, 0)
	if err != nil {
		t.Fatalf("emitRWKVIR: %v", err)
	}
	if wi != 10 {
		t.Fatalf("expected wi=10, got %d", wi)
	}

	hasSigmoid := false
	hasExp := false
	hasScan := false
	hasDiv := false
	hasReLU := false
	hasSquare := false
	hasShiftConcat := false
	hasBroadcastPieces := false
	for _, op := range p.Ops {
		switch op.Code {
		case OpSigmoid:
			hasSigmoid = true
		case OpExp:
			hasExp = true
		case OpScan:
			hasScan = true
		case OpDiv:
			hasDiv = true
		case OpReLU:
			hasReLU = true
		case OpSquare:
			hasSquare = true
		case OpConcat:
			hasShiftConcat = true
		case OpFull:
			hasBroadcastPieces = true
		}
	}
	if !hasSigmoid {
		t.Error("missing Sigmoid op (receptance gate)")
	}
	if !hasExp {
		t.Error("missing Exp op (exp(k) or decay)")
	}
	if !hasScan {
		t.Error("missing Scan op (time-mixing recurrence)")
	}
	if !hasDiv {
		t.Error("missing Div op (WKV normalization)")
	}
	if !hasReLU {
		t.Error("missing ReLU op (channel-mix key)")
	}
	if !hasSquare {
		t.Error("missing Square op (channel-mix relu_square)")
	}
	if !hasShiftConcat {
		t.Error("missing Concat op (token shift history)")
	}
	if !hasBroadcastPieces {
		t.Error("missing Full op (broadcast/token shift helper)")
	}
}

func TestEmitRWKVIR_OpCounts(t *testing.T) {
	p := NewProgram(10)
	_, err := emitRWKVIR(p, "x", 0, 64, 32, 2, 0)
	if err != nil {
		t.Fatalf("emitRWKVIR: %v", err)
	}

	if n := countOps(p, OpConcat); n != 2 {
		t.Errorf("expected 2 Concat ops, got %d", n)
	}
	if n := countOps(p, OpScan); n != 2 {
		t.Errorf("expected 2 Scan ops, got %d", n)
	}
	if n := countOps(p, OpSigmoid); n != 4 {
		t.Errorf("expected 4 Sigmoid ops, got %d", n)
	}
	if n := countOps(p, OpReLU); n != 1 {
		t.Errorf("expected 1 ReLU op, got %d", n)
	}
	if n := countOps(p, OpSquare); n != 1 {
		t.Errorf("expected 1 Square op, got %d", n)
	}
}

func TestEmitRWKVIR_InvalidDim(t *testing.T) {
	p := NewProgram(10)
	_, err := emitRWKVIR(p, "x", 0, 0, 32, 2, 0)
	if err == nil {
		t.Fatal("expected error for D=0")
	}
}

func TestEmitBlockIR_RWKV(t *testing.T) {
	p := NewProgram(10)
	wi, err := emitBlockIR(p, BlockSpec{Type: "rwkv"}, "x", 0, 64, 32, 1, 1024, 0, nil, DefaultFFNMultiplier, false)
	if err != nil {
		t.Fatalf("emitBlockIR(rwkv): %v", err)
	}
	if wi != 10 {
		t.Fatalf("emitBlockIR(rwkv) wi=%d, want 10", wi)
	}
}

// --- RetNet block tests ---

func TestEmitRetNetIR(t *testing.T) {
	p := NewProgram(8)
	wi, err := emitRetNetIR(p, "x", 0, 4, 128, 64, 2, 0)
	if err != nil {
		t.Fatalf("emitRetNetIR: %v", err)
	}
	if wi != 8 {
		t.Fatalf("expected wi=8, got %d", wi)
	}
	hasRMSNorm := false
	hasRetentionMask := false
	hasSiLU := false
	hasMatMul := false
	for _, op := range p.Ops {
		switch op.Code {
		case OpRMSNorm:
			hasRMSNorm = true
		case OpArange, OpOuter, OpMeanAxis, OpExp:
			hasRetentionMask = true
		case OpSiLU:
			hasSiLU = true
		case OpMatMul:
			hasMatMul = true
		}
	}
	if !hasRMSNorm {
		t.Error("missing RMSNorm op")
	}
	if !hasRetentionMask {
		t.Error("missing retention decomposition ops")
	}
	if !hasSiLU {
		t.Error("missing SiLU op (feed-forward tail)")
	}
	if !hasMatMul {
		t.Error("missing MatMul op")
	}
}

func TestEmitRetNetIR_NoCausalMask(t *testing.T) {
	p := NewProgram(8)
	_, err := emitRetNetIR(p, "x", 0, 4, 128, 64, 2, 0)
	if err != nil {
		t.Fatalf("emitRetNetIR: %v", err)
	}
	hasCausalMask := false
	for _, op := range p.Ops {
		if op.Code == OpCausalMask {
			hasCausalMask = true
		}
		if op.Code == OpSoftmax {
			t.Error("RetNet should not use Softmax")
		}
	}
	if !hasCausalMask {
		t.Error("RetNet should use CausalMask to zero future retention weights")
	}
}

func TestEmitRetNetIR_InvalidDims(t *testing.T) {
	p := NewProgram(8)
	_, err := emitRetNetIR(p, "x", 0, 3, 128, 64, 2, 0)
	if err == nil {
		t.Fatal("expected error for D%H != 0")
	}
}

func TestEmitRetNetIR_ZeroHeads(t *testing.T) {
	p := NewProgram(8)
	_, err := emitRetNetIR(p, "x", 0, 0, 128, 64, 2, 0)
	if err == nil {
		t.Fatal("expected error for zero heads")
	}
}

func TestEmitBlockIR_RetNet(t *testing.T) {
	p := NewProgram(8)
	wi, err := emitBlockIR(p, BlockSpec{Type: "retnet", Heads: 4}, "x", 0, 128, 64, 1, 1024, 0, nil, DefaultFFNMultiplier, false)
	if err != nil {
		t.Fatalf("emitBlockIR(retnet): %v", err)
	}
	if wi != 8 {
		t.Fatalf("expected wi=8, got %d", wi)
	}
}

func TestEmitBlockIR_RetNetDefaultHeads(t *testing.T) {
	p := NewProgram(8)
	wi, err := emitBlockIR(p, BlockSpec{Type: "retnet", Heads: 0}, "x", 0, 128, 64, 1, 1024, 0, nil, DefaultFFNMultiplier, false)
	if err != nil {
		t.Fatalf("emitBlockIR(retnet, heads=0): %v", err)
	}
	if wi != 8 {
		t.Fatalf("expected wi=8, got %d", wi)
	}
}

// --- Mamba block emission ---

func TestEmitMambaIR(t *testing.T) {
	p := NewProgram(4)
	wi, err := emitMambaIR(p, "x", 0, 128, 64, 1, 0)
	if err != nil {
		t.Fatalf("emitMambaIR: %v", err)
	}
	if wi != 4 {
		t.Fatalf("expected wi=4, got %d", wi)
	}
	hasScan := false
	hasSiLU := false
	hasMul := false
	hasSlice := false
	for _, op := range p.Ops {
		switch op.Code {
		case OpScan:
			hasScan = true
			if len(op.IntParams) != 3 {
				t.Errorf("OpScan expected 3 int params, got %d", len(op.IntParams))
			}
			if op.IntParams[0] != 1 || op.IntParams[1] != 64 || op.IntParams[2] != 128 {
				t.Errorf("OpScan params = %v, want [1, 64, 128]", op.IntParams)
			}
		case OpSiLU:
			hasSiLU = true
		case OpMul:
			hasMul = true
		case OpSlice:
			hasSlice = true
		}
	}
	if !hasScan {
		t.Error("missing OpScan op in mamba block")
	}
	if !hasSiLU {
		t.Error("missing SiLU op in mamba block (gating)")
	}
	if !hasMul {
		t.Error("missing Mul op in mamba block (h_scan * gate)")
	}
	if !hasSlice {
		t.Error("missing Slice op in mamba block (proj split)")
	}
}

func TestEmitMambaIR_CustomInnerDim(t *testing.T) {
	p := NewProgram(4)
	wi, err := emitMambaIR(p, "x", 0, 64, 32, 2, 0)
	if err != nil {
		t.Fatalf("emitMambaIR: %v", err)
	}
	if wi != 4 {
		t.Fatalf("expected wi=4, got %d", wi)
	}
	for _, op := range p.Ops {
		if op.Code == OpScan && op.IntParams[2] != 64 {
			t.Errorf("OpScan inner dim = %d, want 64", op.IntParams[2])
		}
	}
}

func TestEmitMambaIR_InvalidInnerDim(t *testing.T) {
	p := NewProgram(4)
	_, err := emitMambaIR(p, "x", 0, 0, 64, 1, 0)
	if err == nil {
		t.Fatal("expected error for inner_dim=0")
	}
}

func TestEmitBlockIR_MambaDispatch(t *testing.T) {
	p := NewProgram(4)
	wi, err := emitBlockIR(p, BlockSpec{Type: "mamba"}, "x", 0, 128, 64, 1, 1024, 0, nil, DefaultFFNMultiplier, false)
	if err != nil {
		t.Fatalf("emitBlockIR(mamba): %v", err)
	}
	if wi != 4 {
		t.Fatalf("expected wi=4, got %d", wi)
	}
}

func TestEmitBlockIR_MambaDefaultInner(t *testing.T) {
	p := NewProgram(4)
	wi, err := emitBlockIR(p, BlockSpec{Type: "mamba", InnerDim: 0}, "x", 0, 128, 64, 1, 1024, 0, nil, DefaultFFNMultiplier, false)
	if err != nil {
		t.Fatalf("emitBlockIR(mamba): %v", err)
	}
	if wi != 4 {
		t.Fatalf("expected wi=4, got %d", wi)
	}
	for _, op := range p.Ops {
		if op.Code == OpScan && op.IntParams[2] != 128 {
			t.Errorf("OpScan inner dim = %d, want 128 (default to D)", op.IntParams[2])
		}
	}
}

// --- emitCrossAttentionIR ---

func TestEmitCrossAttentionIR(t *testing.T) {
	p := NewProgram(7)
	wi, err := emitCrossAttentionIR(p, "q_stream", "kv_stream", 0, 4, 128, 32, 64, 2, 0)
	if err != nil {
		t.Fatalf("emitCrossAttentionIR: %v", err)
	}
	if wi != 7 {
		t.Fatalf("expected wi=7, got %d", wi)
	}

	hasRMSNorm := false
	hasSoftmax := false
	hasSiLU := false
	hasMatMul := false
	for _, op := range p.Ops {
		switch op.Code {
		case OpRMSNorm:
			hasRMSNorm = true
		case OpSoftmax:
			hasSoftmax = true
		case OpSiLU:
			hasSiLU = true
		case OpMatMul:
			hasMatMul = true
		}
	}
	if !hasRMSNorm {
		t.Error("missing RMSNorm op")
	}
	if !hasSoftmax {
		t.Error("missing Softmax op")
	}
	if !hasSiLU {
		t.Error("missing SiLU op (feed-forward tail)")
	}
	if !hasMatMul {
		t.Error("missing MatMul op")
	}

	for _, op := range p.Ops {
		if op.Code == OpCausalMask {
			t.Error("cross-attention should not have CausalMask")
		}
		if op.Code == OpRoPE {
			t.Error("cross-attention should not have RoPE")
		}
	}
}

func TestEmitCrossAttentionIR_InvalidDims(t *testing.T) {
	p := NewProgram(7)
	_, err := emitCrossAttentionIR(p, "q", "kv", 0, 3, 128, 32, 64, 2, 0)
	if err == nil {
		t.Fatal("expected error for D%H != 0")
	}
}

func TestEmitCrossAttentionIR_ZeroHeads(t *testing.T) {
	p := NewProgram(7)
	_, err := emitCrossAttentionIR(p, "q", "kv", 0, 0, 128, 32, 64, 2, 0)
	if err == nil {
		t.Fatal("expected error for zero heads")
	}
}

func TestEmitCrossAttentionIR_KVStreamInput(t *testing.T) {
	p := NewProgram(7)
	_, err := emitCrossAttentionIR(p, "q_stream", "kv_stream", 0, 4, 128, 32, 64, 2, 0)
	if err != nil {
		t.Fatalf("emitCrossAttentionIR: %v", err)
	}
	matMulOps := make([]Op, 0)
	for _, op := range p.Ops {
		if op.Code == OpMatMul {
			matMulOps = append(matMulOps, op)
		}
	}
	if len(matMulOps) < 3 {
		t.Fatalf("expected at least 3 MatMul ops, got %d", len(matMulOps))
	}
	if matMulOps[1].Inputs[0] != "kv_stream" {
		t.Errorf("K projection input should be kv_stream, got %q", matMulOps[1].Inputs[0])
	}
	if matMulOps[2].Inputs[0] != "kv_stream" {
		t.Errorf("V projection input should be kv_stream, got %q", matMulOps[2].Inputs[0])
	}
}

func TestEmitBlockIR_CrossAttentionDispatch(t *testing.T) {
	spec := BlockSpec{
		Type:         "cross_attention",
		Heads:        4,
		SourceStream: "low",
	}
	streamSeqLens := map[string]int{"low": 64}
	p := NewProgram(7)
	wi, err := emitBlockIR(p, spec, "high", 0, 128, 32, 1, 1024, 0, streamSeqLens, DefaultFFNMultiplier, false)
	if err != nil {
		t.Fatalf("emitBlockIR(cross_attention): %v", err)
	}
	if wi != 7 {
		t.Fatalf("expected wi=7, got %d", wi)
	}
}

func TestEmitBlockIR_CrossAttentionMissingSourceStream(t *testing.T) {
	spec := BlockSpec{
		Type:  "cross_attention",
		Heads: 4,
	}
	p := NewProgram(7)
	_, err := emitBlockIR(p, spec, "high", 0, 128, 32, 1, 1024, 0, nil, DefaultFFNMultiplier, false)
	if err == nil {
		t.Fatal("expected error for missing source_stream")
	}
}

func TestEmitBlockIR_CrossAttentionUnknownSourceStream(t *testing.T) {
	spec := BlockSpec{
		Type:         "cross_attention",
		Heads:        4,
		SourceStream: "nonexistent",
	}
	streamSeqLens := map[string]int{"low": 64}
	p := NewProgram(7)
	_, err := emitBlockIR(p, spec, "high", 0, 128, 32, 1, 1024, 0, streamSeqLens, DefaultFFNMultiplier, false)
	if err == nil {
		t.Fatal("expected error for unknown source_stream")
	}
}

func TestEmitBlockIR_CrossAttentionDefaultHeads(t *testing.T) {
	spec := BlockSpec{
		Type:         "cross_attention",
		Heads:        0,
		SourceStream: "low",
	}
	streamSeqLens := map[string]int{"low": 64}
	p := NewProgram(7)
	wi, err := emitBlockIR(p, spec, "high", 0, 128, 32, 1, 1024, 0, streamSeqLens, DefaultFFNMultiplier, false)
	if err != nil {
		t.Fatalf("emitBlockIR(cross_attention default heads): %v", err)
	}
	if wi != 7 {
		t.Fatalf("expected wi=7, got %d", wi)
	}
}

// --- emitMamba3IR ---

func TestEmitMamba3IR(t *testing.T) {
	p := NewProgram(6)
	wi, err := emitMamba3IR(p, "x", 0, 128, 64, 2, 0)
	if err != nil {
		t.Fatalf("emitMamba3IR: %v", err)
	}
	if wi != 6 {
		t.Fatalf("expected wi=6, got %d", wi)
	}

	hasRMSNorm := false
	hasSiLU := false
	hasSigmoid := false
	hasScan := false
	hasMul := false
	hasAdd := false
	for _, op := range p.Ops {
		switch op.Code {
		case OpRMSNorm:
			hasRMSNorm = true
		case OpSiLU:
			hasSiLU = true
		case OpSigmoid:
			hasSigmoid = true
		case OpScan:
			hasScan = true
		case OpMul:
			hasMul = true
		case OpAdd:
			hasAdd = true
		}
	}
	if !hasRMSNorm {
		t.Error("missing RMSNorm op")
	}
	if !hasSiLU {
		t.Error("missing SiLU op (gate branch)")
	}
	if !hasSigmoid {
		t.Error("missing Sigmoid op (delta_t gating)")
	}
	if !hasScan {
		t.Error("missing Scan op (gated recurrence)")
	}
	if !hasMul {
		t.Error("missing Mul op")
	}
	if !hasAdd {
		t.Error("missing Add op (residual)")
	}
}

func TestEmitMamba3IR_InvalidInnerDim(t *testing.T) {
	p := NewProgram(6)
	_, err := emitMamba3IR(p, "x", 0, 0, 64, 2, 0)
	if err == nil {
		t.Fatal("expected error for inner_dim=0")
	}
}

func TestEmitMamba3IR_WeightConsumption(t *testing.T) {
	p := NewProgram(6)
	wi, err := emitMamba3IR(p, "x", 10, 128, 64, 2, 0)
	if err != nil {
		t.Fatalf("emitMamba3IR: %v", err)
	}
	if wi != 16 {
		t.Fatalf("expected wi=16, got %d", wi)
	}
	if p.Ops[0].Code != OpRMSNorm {
		t.Fatalf("first op should be RMSNorm, got %d", p.Ops[0].Code)
	}
	if p.Ops[0].Inputs[1] != "w10" {
		t.Errorf("RMSNorm scale should be w10, got %q", p.Ops[0].Inputs[1])
	}
}

func TestEmitMamba3IR_OpCounts(t *testing.T) {
	p := NewProgram(6)
	_, err := emitMamba3IR(p, "x", 0, 128, 64, 2, 0)
	if err != nil {
		t.Fatalf("emitMamba3IR: %v", err)
	}
	if len(p.Ops) != 11 {
		t.Fatalf("expected 11 ops, got %d", len(p.Ops))
	}
}
