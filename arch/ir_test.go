package arch

import (
	"testing"
)

// --- Op and Program basics ---

func TestNewProgram(t *testing.T) {
	p := NewProgram(5)
	if p.NumWeights != 5 {
		t.Fatalf("expected 5 weights, got %d", p.NumWeights)
	}
	if len(p.Ops) != 0 {
		t.Fatalf("expected 0 ops, got %d", len(p.Ops))
	}
}

func TestDeclareInputOutput(t *testing.T) {
	p := NewProgram(1)
	p.DeclareInput("tokens", TensorInt32, []int{2, 128})
	p.DeclareOutput("loss", TensorFloat32, []int{1})

	if len(p.Inputs) != 1 {
		t.Fatalf("expected 1 input, got %d", len(p.Inputs))
	}
	if p.Inputs[0].Name != "tokens" || p.Inputs[0].DType != TensorInt32 {
		t.Fatalf("unexpected input: %+v", p.Inputs[0])
	}
	if len(p.Outputs) != 1 {
		t.Fatalf("expected 1 output, got %d", len(p.Outputs))
	}
	if p.Outputs[0].Name != "loss" {
		t.Fatalf("unexpected output name: %s", p.Outputs[0].Name)
	}
}

func TestAddOp(t *testing.T) {
	p := NewProgram(0)
	p.AddOp(OpMatMul, []string{"a", "b"}, []string{"c"}, nil, nil)
	if len(p.Ops) != 1 {
		t.Fatalf("expected 1 op, got %d", len(p.Ops))
	}
	if p.Ops[0].Code != OpMatMul {
		t.Fatalf("expected OpMatMul (%d), got %d", OpMatMul, p.Ops[0].Code)
	}
}

func TestWeightName(t *testing.T) {
	if weightName(0) != "w0" {
		t.Fatalf("expected w0, got %s", weightName(0))
	}
	if weightName(42) != "w42" {
		t.Fatalf("expected w42, got %s", weightName(42))
	}
}

func TestTmpName(t *testing.T) {
	if tmpName("x_attn", 3) != "x_attn_3" {
		t.Fatalf("unexpected tmp name: %s", tmpName("x_attn", 3))
	}
}

// --- Convenience method emission tests ---

func TestEmbed(t *testing.T) {
	p := NewProgram(1)
	p.Embed("w0", "tokens", "x_embed")
	if len(p.Ops) != 1 || p.Ops[0].Code != OpEmbed {
		t.Fatalf("unexpected: %+v", p.Ops)
	}
	if p.Ops[0].Inputs[0] != "w0" || p.Ops[0].Inputs[1] != "tokens" {
		t.Fatalf("bad embed inputs: %v", p.Ops[0].Inputs)
	}
}

func TestRMSNorm(t *testing.T) {
	p := NewProgram(1)
	p.RMSNorm("x", "w0", "x_norm", 1e-5)
	if len(p.Ops) != 1 || p.Ops[0].Code != OpRMSNorm {
		t.Fatalf("unexpected: %+v", p.Ops)
	}
	if len(p.Ops[0].FloatParams) != 1 {
		t.Fatalf("expected 1 float param, got %d", len(p.Ops[0].FloatParams))
	}
}

func TestRoPE(t *testing.T) {
	p := NewProgram(0)
	p.RoPE("q", "k", "qr", "kr", 128, 32, 16, 10000.0)
	if len(p.Ops) != 1 || p.Ops[0].Code != OpRoPE {
		t.Fatalf("unexpected: %+v", p.Ops)
	}
	if len(p.Ops[0].Outputs) != 2 {
		t.Fatalf("expected 2 outputs for RoPE, got %d", len(p.Ops[0].Outputs))
	}
	if got := p.Ops[0].IntParams; len(got) != 3 || got[0] != 128 || got[1] != 32 || got[2] != 16 {
		t.Fatalf("bad RoPE int params: %v", got)
	}
}

func TestRoPEIndexed(t *testing.T) {
	p := NewProgram(0)
	p.RoPEIndexed("q", "k", "positions", "qr", "kr", 3, 8, 4, 10000.0)
	if len(p.Ops) != 1 || p.Ops[0].Code != OpRoPEIndexed {
		t.Fatalf("unexpected: %+v", p.Ops)
	}
	op := p.Ops[0]
	if len(op.Inputs) != 3 || op.Inputs[0] != "q" || op.Inputs[1] != "k" || op.Inputs[2] != "positions" {
		t.Fatalf("bad RoPEIndexed inputs: %v", op.Inputs)
	}
	if len(op.Outputs) != 2 || op.Outputs[0] != "qr" || op.Outputs[1] != "kr" {
		t.Fatalf("bad RoPEIndexed outputs: %v", op.Outputs)
	}
	if len(op.FloatParams) != 1 || op.FloatParams[0] != 10000.0 {
		t.Fatalf("bad RoPEIndexed float params: %v", op.FloatParams)
	}
	if len(op.IntParams) != 3 || op.IntParams[0] != 3 || op.IntParams[1] != 8 || op.IntParams[2] != 4 {
		t.Fatalf("bad RoPEIndexed int params: %v", op.IntParams)
	}
}

func TestBroadcast(t *testing.T) {
	p := NewProgram(0)
	p.Broadcast("latents", 4, "latents_b")
	if len(p.Ops) != 3 {
		t.Fatalf("unexpected: %+v", p.Ops)
	}
	if p.Ops[0].Code != OpReshape || p.Ops[1].Code != OpFull || p.Ops[2].Code != OpMul {
		t.Fatalf("unexpected op sequence: %+v", p.Ops)
	}
	if got := p.Ops[1].IntParams; len(got) != 2 || got[0] != 4 || got[1] != 1 {
		t.Fatalf("expected full shape [4 1], got %v", got)
	}
}

func TestScanOp(t *testing.T) {
	p := NewProgram(0)
	p.Scan("h", "decay", "out", 2, 64, 128)
	if len(p.Ops) != 1 {
		t.Fatalf("expected 1 op, got %d", len(p.Ops))
	}
	op := p.Ops[0]
	if op.Code != OpScan {
		t.Fatalf("expected OpScan (%d), got %d", OpScan, op.Code)
	}
	if len(op.Inputs) != 2 || op.Inputs[0] != "h" || op.Inputs[1] != "decay" {
		t.Fatalf("bad scan inputs: %v", op.Inputs)
	}
	if len(op.IntParams) != 3 || op.IntParams[0] != 2 || op.IntParams[1] != 64 || op.IntParams[2] != 128 {
		t.Fatalf("bad scan int params: %v", op.IntParams)
	}
}

func TestPositionIndexingOps(t *testing.T) {
	p := NewProgram(0)
	p.DeclareInput("x", TensorFloat32, []int{2, 5, 3})
	p.DeclareInput("positions", TensorInt32, []int{2})
	p.DeclareOutput("out", TensorFloat32, []int{2, 5, 3})

	p.GatherPositions("x", "positions", "picked", 2, 2, 3)
	p.ScatterPositions("x", "picked", "positions", "out", 2, 5, 2, 3)

	if len(p.Ops) != 2 {
		t.Fatalf("expected 2 ops, got %d", len(p.Ops))
	}

	gather := p.Ops[0]
	if gather.Code != OpGatherPositions {
		t.Fatalf("expected OpGatherPositions (%d), got %d", OpGatherPositions, gather.Code)
	}
	if len(gather.Inputs) != 2 || gather.Inputs[0] != "x" || gather.Inputs[1] != "positions" {
		t.Fatalf("bad gather inputs: %v", gather.Inputs)
	}
	if len(gather.Outputs) != 1 || gather.Outputs[0] != "picked" {
		t.Fatalf("bad gather outputs: %v", gather.Outputs)
	}
	if len(gather.IntParams) != 3 || gather.IntParams[0] != 2 || gather.IntParams[1] != 2 || gather.IntParams[2] != 3 {
		t.Fatalf("bad gather int params: %v", gather.IntParams)
	}

	scatter := p.Ops[1]
	if scatter.Code != OpScatterPositions {
		t.Fatalf("expected OpScatterPositions (%d), got %d", OpScatterPositions, scatter.Code)
	}
	if len(scatter.Inputs) != 3 || scatter.Inputs[0] != "x" || scatter.Inputs[1] != "picked" || scatter.Inputs[2] != "positions" {
		t.Fatalf("bad scatter inputs: %v", scatter.Inputs)
	}
	if len(scatter.Outputs) != 1 || scatter.Outputs[0] != "out" {
		t.Fatalf("bad scatter outputs: %v", scatter.Outputs)
	}
	if len(scatter.IntParams) != 4 || scatter.IntParams[0] != 2 || scatter.IntParams[1] != 5 || scatter.IntParams[2] != 2 || scatter.IntParams[3] != 3 {
		t.Fatalf("bad scatter int params: %v", scatter.IntParams)
	}
}

func TestSliceOp(t *testing.T) {
	p := NewProgram(0)
	p.Slice("proj", 0, 64, 1, 1, "z")
	if len(p.Ops) != 1 {
		t.Fatalf("expected 1 op, got %d", len(p.Ops))
	}
	op := p.Ops[0]
	if op.Code != OpSlice {
		t.Fatalf("expected OpSlice (%d), got %d", OpSlice, op.Code)
	}
	if len(op.IntParams) != 4 || op.IntParams[0] != 0 || op.IntParams[1] != 64 || op.IntParams[2] != 1 || op.IntParams[3] != 1 {
		t.Fatalf("bad slice int params: %v", op.IntParams)
	}
}

func TestSub(t *testing.T) {
	p := NewProgram(0)
	p.Sub("a", "b", "c")
	if len(p.Ops) != 1 {
		t.Fatalf("expected 1 op, got %d", len(p.Ops))
	}
	op := p.Ops[0]
	if op.Code != OpSub {
		t.Fatalf("expected OpSub (%d), got %d", OpSub, op.Code)
	}
	if len(op.Inputs) != 2 || op.Inputs[0] != "a" || op.Inputs[1] != "b" {
		t.Fatalf("bad inputs: %v", op.Inputs)
	}
	if len(op.Outputs) != 1 || op.Outputs[0] != "c" {
		t.Fatalf("bad outputs: %v", op.Outputs)
	}
}

func TestGELU(t *testing.T) {
	p := NewProgram(0)
	p.GELU("a", "b")
	if len(p.Ops) != 1 {
		t.Fatalf("expected 1 op, got %d", len(p.Ops))
	}
	op := p.Ops[0]
	if op.Code != OpGELU {
		t.Fatalf("expected OpGELU (%d), got %d", OpGELU, op.Code)
	}
	if len(op.Inputs) != 1 || op.Inputs[0] != "a" {
		t.Fatalf("bad inputs: %v", op.Inputs)
	}
	if len(op.Outputs) != 1 || op.Outputs[0] != "b" {
		t.Fatalf("bad outputs: %v", op.Outputs)
	}
}

func TestTanh(t *testing.T) {
	p := NewProgram(0)
	p.Tanh("a", "b")
	if len(p.Ops) != 1 {
		t.Fatalf("expected 1 op, got %d", len(p.Ops))
	}
	op := p.Ops[0]
	if op.Code != OpTanh {
		t.Fatalf("expected OpTanh (%d), got %d", OpTanh, op.Code)
	}
	if len(op.Inputs) != 1 || op.Inputs[0] != "a" {
		t.Fatalf("bad inputs: %v", op.Inputs)
	}
}

func TestDiv(t *testing.T) {
	p := NewProgram(0)
	p.Div("a", "b", "c")
	if len(p.Ops) != 1 {
		t.Fatalf("expected 1 op, got %d", len(p.Ops))
	}
	op := p.Ops[0]
	if op.Code != OpDiv {
		t.Fatalf("expected OpDiv (%d), got %d", OpDiv, op.Code)
	}
	if len(op.Inputs) != 2 || op.Inputs[0] != "a" || op.Inputs[1] != "b" {
		t.Fatalf("bad inputs: %v", op.Inputs)
	}
	if op.FloatParams != nil {
		t.Fatalf("Div should have nil float params, got %v", op.FloatParams)
	}
}

func TestNegExp(t *testing.T) {
	p := NewProgram(0)
	p.NegExp("a", "neg_a", "out")
	if len(p.Ops) != 2 {
		t.Fatalf("expected 2 ops (ScalarMul + Exp), got %d", len(p.Ops))
	}
	if p.Ops[0].Code != OpScalarMul {
		t.Fatalf("first op should be ScalarMul, got %d", p.Ops[0].Code)
	}
	if p.Ops[0].FloatParams[0] != -1.0 {
		t.Fatalf("ScalarMul scalar should be -1.0, got %f", p.Ops[0].FloatParams[0])
	}
	if p.Ops[0].Inputs[0] != "a" || p.Ops[0].Outputs[0] != "neg_a" {
		t.Fatalf("bad ScalarMul routing: in=%v out=%v", p.Ops[0].Inputs, p.Ops[0].Outputs)
	}
	if p.Ops[1].Code != OpExp {
		t.Fatalf("second op should be Exp, got %d", p.Ops[1].Code)
	}
	if p.Ops[1].Inputs[0] != "neg_a" || p.Ops[1].Outputs[0] != "out" {
		t.Fatalf("bad Exp routing: in=%v out=%v", p.Ops[1].Inputs, p.Ops[1].Outputs)
	}
}

func countOps(prog *Program, code int) int {
	n := 0
	for _, op := range prog.Ops {
		if op.Code == code {
			n++
		}
	}
	return n
}
