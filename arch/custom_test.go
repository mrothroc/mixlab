package arch

import (
	"testing"
)

// --- resolveShapeSymbol ---

func TestResolveShapeSymbol_AllSymbols(t *testing.T) {
	D, H, T, B, V := 128, 4, 256, 2, 1024
	tests := []struct {
		sym  string
		want int
	}{
		{"D", 128},
		{"H", 4},
		{"HD", 32},   // 128/4
		{"FFN", 341}, // int(2.67*128)
		{"2D", 256},
		{"3D", 384},
		{"4D", 512},
		{"8D", 1024},
		{"T", 256},
		{"B", 2},
		{"V", 1024},
		{"BT", 512},
		{"T/2", 128},
		{"64", 64},    // literal integer
		{"1.5D", 192}, // float multiplier
	}
	for _, tt := range tests {
		got, err := resolveShapeSymbol([]string{tt.sym}, D, H, T, B, V)
		if err != nil {
			t.Fatalf("resolveShapeSymbol(%q): %v", tt.sym, err)
		}
		if got[0] != tt.want {
			t.Errorf("resolveShapeSymbol(%q) = %d, want %d", tt.sym, got[0], tt.want)
		}
	}
}

func TestResolveShapeSymbol_MultiDim(t *testing.T) {
	got, err := resolveShapeSymbol([]string{"D", "FFN"}, 128, 4, 256, 2, 1024)
	if err != nil {
		t.Fatalf("resolveShapeSymbol: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("expected 2 dims, got %d", len(got))
	}
	if got[0] != 128 || got[1] != 341 {
		t.Errorf("got %v, want [128, 341]", got)
	}
}

func TestResolveShapeSymbol_EmptyShape(t *testing.T) {
	_, err := resolveShapeSymbol(nil, 128, 4, 256, 2, 1024)
	if err == nil {
		t.Fatal("expected error for nil shape")
	}
}

func TestResolveShapeSymbol_UnknownSymbol(t *testing.T) {
	_, err := resolveShapeSymbol([]string{"UNKNOWN"}, 128, 4, 256, 2, 1024)
	if err == nil {
		t.Fatal("expected error for unknown symbol")
	}
}

func TestResolveShapeSymbol_InvalidHD(t *testing.T) {
	// D=128, H=3 -> 128%3 != 0
	_, err := resolveShapeSymbol([]string{"HD"}, 128, 3, 256, 2, 1024)
	if err == nil {
		t.Fatal("expected error for invalid HD")
	}
}

func TestResolveShapeSymbol_FloatMultiplier(t *testing.T) {
	got, err := resolveShapeSymbol([]string{"2.67D"}, 100, 4, 256, 2, 1024)
	if err != nil {
		t.Fatalf("resolveShapeSymbol(2.67D): %v", err)
	}
	if got[0] != 267 {
		t.Errorf("resolveShapeSymbol(2.67D) = %d, want 267", got[0])
	}
}

// --- emitCustomBlockIR: GeGLU ---

func TestEmitCustomBlockIR_GeGLU(t *testing.T) {
	spec := BlockSpec{
		Type: "custom",
		Name: "geglu",
		Weights: []WeightSpec{
			{Name: "w_gate", Shape: []string{"D", "FFN"}},
			{Name: "w_up", Shape: []string{"D", "FFN"}},
			{Name: "w_down", Shape: []string{"FFN", "D"}},
		},
		Ops: []OpSpec{
			{Op: "matmul", Inputs: []string{"x", "w_gate"}, Output: "gate"},
			{Op: "silu", Inputs: []string{"gate"}, Output: "gate_act"},
			{Op: "matmul", Inputs: []string{"x", "w_up"}, Output: "up"},
			{Op: "mul", Inputs: []string{"gate_act", "up"}, Output: "ff"},
			{Op: "matmul", Inputs: []string{"ff", "w_down"}, Output: "ff_out"},
			{Op: "add", Inputs: []string{"x", "ff_out"}, Output: "x"},
		},
	}

	prog := NewProgram(3)
	wi, err := emitCustomBlockIR(prog, spec, "x", 0, 128, 64, 1, 1024, 0)
	if err != nil {
		t.Fatalf("emitCustomBlockIR: %v", err)
	}

	// Should consume 3 weights (w_gate, w_up, w_down).
	if wi != 3 {
		t.Fatalf("expected wi=3, got %d", wi)
	}

	// Should emit 6 ops.
	if len(prog.Ops) != 6 {
		t.Fatalf("expected 6 ops, got %d", len(prog.Ops))
	}

	// Verify op order.
	expectedOps := []int{OpMatMul, OpSiLU, OpMatMul, OpMul, OpMatMul, OpAdd}
	for i, want := range expectedOps {
		if prog.Ops[i].Code != want {
			t.Errorf("op[%d] = %d, want %d", i, prog.Ops[i].Code, want)
		}
	}

	// Verify that weight names are resolved: first matmul input[1] should be "w0".
	if prog.Ops[0].Inputs[1] != "w0" {
		t.Errorf("first matmul weight input = %q, want \"w0\"", prog.Ops[0].Inputs[1])
	}

	// Verify that "x" maps to the stream name.
	if prog.Ops[0].Inputs[0] != "x" {
		t.Errorf("first matmul stream input = %q, want \"x\"", prog.Ops[0].Inputs[0])
	}

	// Verify the last op (add) outputs to stream "x".
	if prog.Ops[5].Outputs[0] != "x" {
		t.Errorf("last op output = %q, want \"x\"", prog.Ops[5].Outputs[0])
	}
}

// --- Weight count ---

func TestBlockWeightCount_Custom(t *testing.T) {
	spec := BlockSpec{
		Type: "custom",
		Name: "geglu",
		Weights: []WeightSpec{
			{Name: "w_gate", Shape: []string{"D", "FFN"}},
			{Name: "w_up", Shape: []string{"D", "FFN"}},
			{Name: "w_down", Shape: []string{"FFN", "D"}},
		},
	}
	n, err := BlockWeightCount(spec, false, false)
	if err != nil {
		t.Fatalf("BlockWeightCount: %v", err)
	}
	if n != 3 {
		t.Fatalf("expected 3, got %d", n)
	}
}

func TestBlockWeightCount_CustomEmpty(t *testing.T) {
	spec := BlockSpec{
		Type: "custom",
		Name: "empty",
	}
	n, err := BlockWeightCount(spec, false, false)
	if err != nil {
		t.Fatalf("BlockWeightCount: %v", err)
	}
	if n != 0 {
		t.Fatalf("expected 0, got %d", n)
	}
}

// --- Error cases ---

func TestEmitCustomBlockIR_UnknownOp(t *testing.T) {
	spec := BlockSpec{
		Type: "custom",
		Name: "bad_op",
		Weights: []WeightSpec{
			{Name: "w", Shape: []string{"D", "D"}},
		},
		Ops: []OpSpec{
			{Op: "frobulate", Inputs: []string{"x", "w"}, Output: "y"},
		},
	}
	prog := NewProgram(1)
	_, err := emitCustomBlockIR(prog, spec, "x", 0, 128, 64, 1, 1024, 0)
	if err == nil {
		t.Fatal("expected error for unknown op")
	}
}

func TestEmitCustomBlockIR_MissingOutput(t *testing.T) {
	spec := BlockSpec{
		Type: "custom",
		Name: "no_output",
		Weights: []WeightSpec{
			{Name: "w", Shape: []string{"D", "D"}},
		},
		Ops: []OpSpec{
			{Op: "matmul", Inputs: []string{"x", "w"}},
		},
	}
	prog := NewProgram(1)
	_, err := emitCustomBlockIR(prog, spec, "x", 0, 128, 64, 1, 1024, 0)
	if err == nil {
		t.Fatal("expected error for missing output")
	}
}

func TestEmitCustomBlockIR_MissingWeightName(t *testing.T) {
	spec := BlockSpec{
		Type: "custom",
		Name: "bad_weight",
		Weights: []WeightSpec{
			{Name: "", Shape: []string{"D", "D"}},
		},
		Ops: []OpSpec{
			{Op: "matmul", Inputs: []string{"x", "w"}, Output: "y"},
		},
	}
	prog := NewProgram(1)
	_, err := emitCustomBlockIR(prog, spec, "x", 0, 128, 64, 1, 1024, 0)
	if err == nil {
		t.Fatal("expected error for empty weight name")
	}
}

func TestEmitCustomBlockIR_BadWeightShape(t *testing.T) {
	spec := BlockSpec{
		Type: "custom",
		Name: "bad_shape",
		Weights: []WeightSpec{
			{Name: "w", Shape: []string{"INVALID"}},
		},
		Ops: []OpSpec{
			{Op: "matmul", Inputs: []string{"x", "w"}, Output: "y"},
		},
	}
	prog := NewProgram(1)
	_, err := emitCustomBlockIR(prog, spec, "x", 0, 128, 64, 1, 1024, 0)
	if err == nil {
		t.Fatal("expected error for invalid shape symbol")
	}
}

// --- Dispatch through emitBlockIR ---

func TestEmitBlockIR_Custom(t *testing.T) {
	spec := BlockSpec{
		Type: "custom",
		Name: "simple_ffn",
		Weights: []WeightSpec{
			{Name: "w1", Shape: []string{"D", "4D"}},
			{Name: "w2", Shape: []string{"4D", "D"}},
		},
		Ops: []OpSpec{
			{Op: "matmul", Inputs: []string{"x", "w1"}, Output: "h"},
			{Op: "silu", Inputs: []string{"h"}, Output: "h_act"},
			{Op: "matmul", Inputs: []string{"h_act", "w2"}, Output: "out"},
			{Op: "add", Inputs: []string{"x", "out"}, Output: "x"},
		},
	}
	prog := NewProgram(2)
	wi, err := emitBlockIR(prog, spec, "x", 0, 128, 64, 1, 1024, 0, nil, DefaultFFNMultiplier, false)
	if err != nil {
		t.Fatalf("emitBlockIR(custom): %v", err)
	}
	if wi != 2 {
		t.Fatalf("expected wi=2, got %d", wi)
	}
	if len(prog.Ops) != 4 {
		t.Fatalf("expected 4 ops, got %d", len(prog.Ops))
	}
}

// --- Full program with custom blocks ---

func TestBuildIRProgram_WithCustomBlocks(t *testing.T) {
	geglu := BlockSpec{
		Type: "custom",
		Name: "geglu",
		Weights: []WeightSpec{
			{Name: "w_gate", Shape: []string{"D", "FFN"}},
			{Name: "w_up", Shape: []string{"D", "FFN"}},
			{Name: "w_down", Shape: []string{"FFN", "D"}},
		},
		Ops: []OpSpec{
			{Op: "matmul", Inputs: []string{"x", "w_gate"}, Output: "gate"},
			{Op: "silu", Inputs: []string{"gate"}, Output: "gate_act"},
			{Op: "matmul", Inputs: []string{"x", "w_up"}, Output: "up"},
			{Op: "mul", Inputs: []string{"gate_act", "up"}, Output: "ff"},
			{Op: "matmul", Inputs: []string{"ff", "w_down"}, Output: "ff_out"},
			{Op: "add", Inputs: []string{"x", "ff_out"}, Output: "x"},
		},
	}
	blocks := []BlockSpec{
		{Type: "plain", Heads: 4},
		geglu,
		{Type: "plain", Heads: 4},
		geglu,
	}
	// 3 base + 2*7 plain + 2*3 custom = 3 + 14 + 6 = 23
	prog, err := BuildIRProgram(128, 1024, 128, 1, DefaultFFNMultiplier, false, false, false, false, 0, blocks)
	if err != nil {
		t.Fatalf("BuildIRProgram: %v", err)
	}
	if prog.NumWeights != 23 {
		t.Fatalf("expected 23 weights, got %d", prog.NumWeights)
	}
	// First op should be embedding.
	if prog.Ops[0].Code != OpEmbed {
		t.Fatalf("first op should be Embed, got %d", prog.Ops[0].Code)
	}
	// Last op should be cross-entropy.
	last := prog.Ops[len(prog.Ops)-1]
	if last.Code != OpCrossEntropy {
		t.Fatalf("last op should be CrossEntropy, got %d", last.Code)
	}
}

// --- Custom block with op params (scalar_mul) ---

func TestEmitCustomBlockIR_ScalarMul(t *testing.T) {
	spec := BlockSpec{
		Type: "custom",
		Name: "scaled_ffn",
		Weights: []WeightSpec{
			{Name: "w1", Shape: []string{"D", "D"}},
		},
		Ops: []OpSpec{
			{Op: "matmul", Inputs: []string{"x", "w1"}, Output: "h"},
			{Op: "scalar_mul", Inputs: []string{"h"}, Output: "h_scaled", Params: map[string]interface{}{"scalar": 0.5}},
			{Op: "add", Inputs: []string{"x", "h_scaled"}, Output: "x"},
		},
	}
	prog := NewProgram(1)
	wi, err := emitCustomBlockIR(prog, spec, "x", 0, 128, 64, 1, 1024, 0)
	if err != nil {
		t.Fatalf("emitCustomBlockIR: %v", err)
	}
	if wi != 1 {
		t.Fatalf("expected wi=1, got %d", wi)
	}
	// scalar_mul should have float params.
	scalarOp := prog.Ops[1]
	if scalarOp.Code != OpScalarMul {
		t.Fatalf("expected OpScalarMul, got %d", scalarOp.Code)
	}
	if len(scalarOp.FloatParams) != 1 || scalarOp.FloatParams[0] != 0.5 {
		t.Fatalf("expected scalar=0.5, got %v", scalarOp.FloatParams)
	}
}

// --- Custom block with multiple outputs (e.g. RoPE uses Outputs) ---

func TestEmitCustomBlockIR_MultiOutput(t *testing.T) {
	spec := BlockSpec{
		Type: "custom",
		Name: "multi_out",
		Weights: []WeightSpec{
			{Name: "w", Shape: []string{"D", "D"}},
		},
		Ops: []OpSpec{
			{Op: "matmul", Inputs: []string{"x", "w"}, Outputs: []string{"a", "b"}},
			{Op: "add", Inputs: []string{"a", "b"}, Output: "x"},
		},
	}
	prog := NewProgram(1)
	_, err := emitCustomBlockIR(prog, spec, "x", 0, 128, 64, 1, 1024, 0)
	if err != nil {
		t.Fatalf("emitCustomBlockIR: %v", err)
	}
	if len(prog.Ops[0].Outputs) != 2 {
		t.Fatalf("expected 2 outputs on first op, got %d", len(prog.Ops[0].Outputs))
	}
}

// --- All supported op names ---

func TestOpNameToCode_Coverage(t *testing.T) {
	required := []string{
		"matmul", "add", "sub", "mul", "scalar_mul", "div",
		"sigmoid", "silu", "gelu", "relu", "tanh",
		"softmax", "reshape", "transpose", "rmsnorm", "rope",
	}
	for _, name := range required {
		if _, ok := opNameToCode[name]; !ok {
			t.Errorf("opNameToCode missing %q", name)
		}
	}
}

// --- Custom block with reshape params ---

func TestEmitCustomBlockIR_ReshapeParams(t *testing.T) {
	spec := BlockSpec{
		Type: "custom",
		Name: "reshape_test",
		Weights: []WeightSpec{
			{Name: "w", Shape: []string{"D", "D"}},
		},
		Ops: []OpSpec{
			{Op: "matmul", Inputs: []string{"x", "w"}, Output: "h"},
			{Op: "reshape", Inputs: []string{"h"}, Output: "h_r", Params: map[string]interface{}{
				"shape": []interface{}{"B", "T", "D"},
			}},
			{Op: "reshape", Inputs: []string{"h_r"}, Output: "x", Params: map[string]interface{}{
				"shape": []interface{}{"BT", "D"},
			}},
		},
	}
	prog := NewProgram(1)
	_, err := emitCustomBlockIR(prog, spec, "x", 0, 128, 64, 1, 1024, 0)
	if err != nil {
		t.Fatalf("emitCustomBlockIR: %v", err)
	}
	// Check the reshape has int params resolved.
	reshapeOp := prog.Ops[1]
	if reshapeOp.Code != OpReshape {
		t.Fatalf("expected OpReshape, got %d", reshapeOp.Code)
	}
	// shape should be [B=1, T=64, D=128]
	if len(reshapeOp.IntParams) != 3 {
		t.Fatalf("expected 3 int params, got %d", len(reshapeOp.IntParams))
	}
	if reshapeOp.IntParams[0] != 1 || reshapeOp.IntParams[1] != 64 || reshapeOp.IntParams[2] != 128 {
		t.Errorf("reshape params = %v, want [1, 64, 128]", reshapeOp.IntParams)
	}
}

// --- Param helpers ---

func TestValueToInt(t *testing.T) {
	tests := []struct {
		v    interface{}
		want int
	}{
		{float64(42), 42},
		{float32(7), 7},
		{int(3), 3},
		{int64(100), 100},
		{"99", 99},
	}
	for _, tt := range tests {
		got, err := valueToInt(tt.v)
		if err != nil {
			t.Fatalf("valueToInt(%v): %v", tt.v, err)
		}
		if got != tt.want {
			t.Errorf("valueToInt(%v) = %d, want %d", tt.v, got, tt.want)
		}
	}
}

func TestValueToFloat(t *testing.T) {
	got, err := valueToFloat(float64(1.5))
	if err != nil {
		t.Fatalf("valueToFloat: %v", err)
	}
	if got != 1.5 {
		t.Errorf("valueToFloat(1.5) = %f, want 1.5", got)
	}
}

func TestValueToIntSlice(t *testing.T) {
	got, err := valueToIntSlice([]interface{}{float64(1), float64(2), float64(3)})
	if err != nil {
		t.Fatalf("valueToIntSlice: %v", err)
	}
	if len(got) != 3 || got[0] != 1 || got[1] != 2 || got[2] != 3 {
		t.Errorf("got %v, want [1 2 3]", got)
	}
}

func TestValueToStringSlice(t *testing.T) {
	got, err := valueToStringSlice([]interface{}{"D", "FFN"})
	if err != nil {
		t.Fatalf("valueToStringSlice: %v", err)
	}
	if len(got) != 2 || got[0] != "D" || got[1] != "FFN" {
		t.Errorf("got %v, want [D FFN]", got)
	}
}

func TestValueToIntSlice_BadType(t *testing.T) {
	_, err := valueToIntSlice("not an array")
	if err == nil {
		t.Fatal("expected error for non-array")
	}
}

func TestValueToStringSlice_BadType(t *testing.T) {
	_, err := valueToStringSlice("not an array")
	if err == nil {
		t.Fatal("expected error for non-array")
	}
}

// --- valueToFloat: comprehensive type coverage ---

func TestValueToFloat_Int(t *testing.T) {
	got, err := valueToFloat(int(42))
	if err != nil {
		t.Fatalf("valueToFloat(int): %v", err)
	}
	if got != 42.0 {
		t.Errorf("valueToFloat(int(42)) = %f, want 42.0", got)
	}
}

func TestValueToFloat_Float64(t *testing.T) {
	got, err := valueToFloat(float64(3.14))
	if err != nil {
		t.Fatalf("valueToFloat(float64): %v", err)
	}
	if got != float32(3.14) {
		t.Errorf("valueToFloat(float64(3.14)) = %f, want %f", got, float32(3.14))
	}
}

func TestValueToFloat_Float32(t *testing.T) {
	got, err := valueToFloat(float32(2.5))
	if err != nil {
		t.Fatalf("valueToFloat(float32): %v", err)
	}
	if got != 2.5 {
		t.Errorf("valueToFloat(float32(2.5)) = %f, want 2.5", got)
	}
}

func TestValueToFloat_String(t *testing.T) {
	got, err := valueToFloat("1.23")
	if err != nil {
		t.Fatalf("valueToFloat(string): %v", err)
	}
	if got != float32(1.23) {
		t.Errorf("valueToFloat(\"1.23\") = %f, want %f", got, float32(1.23))
	}
}

func TestValueToFloat_UnsupportedType(t *testing.T) {
	_, err := valueToFloat([]int{1, 2, 3})
	if err == nil {
		t.Fatal("expected error for unsupported type")
	}
}

func TestValueToInt_UnsupportedType(t *testing.T) {
	_, err := valueToInt([]int{1, 2, 3})
	if err == nil {
		t.Fatal("expected error for unsupported type")
	}
}

func TestValueToStringSlice_NonStringElement(t *testing.T) {
	_, err := valueToStringSlice([]interface{}{42})
	if err == nil {
		t.Fatal("expected error for non-string element")
	}
}

func TestValueToIntSlice_BadElement(t *testing.T) {
	_, err := valueToIntSlice([]interface{}{[]int{1}})
	if err == nil {
		t.Fatal("expected error for bad element type")
	}
}

// --- emitCustomBlockIR: sub, gelu, tanh ops ---

func TestEmitCustomBlockIR_SubOp(t *testing.T) {
	spec := BlockSpec{
		Type: "custom",
		Name: "sub_test",
		Weights: []WeightSpec{
			{Name: "w1", Shape: []string{"D", "D"}},
		},
		Ops: []OpSpec{
			{Op: "matmul", Inputs: []string{"x", "w1"}, Output: "h"},
			{Op: "sub", Inputs: []string{"x", "h"}, Output: "x"},
		},
	}
	prog := NewProgram(1)
	wi, err := emitCustomBlockIR(prog, spec, "x", 0, 128, 64, 1, 1024, 0)
	if err != nil {
		t.Fatalf("emitCustomBlockIR: %v", err)
	}
	if wi != 1 {
		t.Fatalf("expected wi=1, got %d", wi)
	}
	if prog.Ops[1].Code != OpSub {
		t.Fatalf("expected OpSub, got %d", prog.Ops[1].Code)
	}
}

func TestEmitCustomBlockIR_GELUOp(t *testing.T) {
	spec := BlockSpec{
		Type: "custom",
		Name: "gelu_test",
		Weights: []WeightSpec{
			{Name: "w1", Shape: []string{"D", "D"}},
		},
		Ops: []OpSpec{
			{Op: "matmul", Inputs: []string{"x", "w1"}, Output: "h"},
			{Op: "gelu", Inputs: []string{"h"}, Output: "h_act"},
			{Op: "add", Inputs: []string{"x", "h_act"}, Output: "x"},
		},
	}
	prog := NewProgram(1)
	wi, err := emitCustomBlockIR(prog, spec, "x", 0, 128, 64, 1, 1024, 0)
	if err != nil {
		t.Fatalf("emitCustomBlockIR: %v", err)
	}
	if wi != 1 {
		t.Fatalf("expected wi=1, got %d", wi)
	}
	if prog.Ops[1].Code != OpGELU {
		t.Fatalf("expected OpGELU, got %d", prog.Ops[1].Code)
	}
}

func TestEmitCustomBlockIR_TanhOp(t *testing.T) {
	spec := BlockSpec{
		Type: "custom",
		Name: "tanh_test",
		Weights: []WeightSpec{
			{Name: "w1", Shape: []string{"D", "D"}},
		},
		Ops: []OpSpec{
			{Op: "matmul", Inputs: []string{"x", "w1"}, Output: "h"},
			{Op: "tanh", Inputs: []string{"h"}, Output: "h_act"},
			{Op: "add", Inputs: []string{"x", "h_act"}, Output: "x"},
		},
	}
	prog := NewProgram(1)
	wi, err := emitCustomBlockIR(prog, spec, "x", 0, 128, 64, 1, 1024, 0)
	if err != nil {
		t.Fatalf("emitCustomBlockIR: %v", err)
	}
	if wi != 1 {
		t.Fatalf("expected wi=1, got %d", wi)
	}
	if prog.Ops[1].Code != OpTanh {
		t.Fatalf("expected OpTanh, got %d", prog.Ops[1].Code)
	}
}

func TestEmitCustomBlockIR_DivOp(t *testing.T) {
	spec := BlockSpec{
		Type: "custom",
		Name: "div_test",
		Weights: []WeightSpec{
			{Name: "w1", Shape: []string{"D", "D"}},
		},
		Ops: []OpSpec{
			{Op: "matmul", Inputs: []string{"x", "w1"}, Output: "h"},
			{Op: "div", Inputs: []string{"x", "h"}, Output: "x"},
		},
	}
	prog := NewProgram(1)
	wi, err := emitCustomBlockIR(prog, spec, "x", 0, 128, 64, 1, 1024, 0)
	if err != nil {
		t.Fatalf("emitCustomBlockIR: %v", err)
	}
	if wi != 1 {
		t.Fatalf("expected wi=1, got %d", wi)
	}
	if prog.Ops[1].Code != OpDiv {
		t.Fatalf("expected OpDiv, got %d", prog.Ops[1].Code)
	}
}

func TestEmitCustomBlockIR_ReLUOp(t *testing.T) {
	spec := BlockSpec{
		Type: "custom",
		Name: "relu_test",
		Weights: []WeightSpec{
			{Name: "w1", Shape: []string{"D", "D"}},
		},
		Ops: []OpSpec{
			{Op: "matmul", Inputs: []string{"x", "w1"}, Output: "h"},
			{Op: "relu", Inputs: []string{"h"}, Output: "h_act"},
			{Op: "add", Inputs: []string{"x", "h_act"}, Output: "x"},
		},
	}
	prog := NewProgram(1)
	wi, err := emitCustomBlockIR(prog, spec, "x", 0, 128, 64, 1, 1024, 0)
	if err != nil {
		t.Fatalf("emitCustomBlockIR: %v", err)
	}
	if wi != 1 {
		t.Fatalf("expected wi=1, got %d", wi)
	}
	if prog.Ops[1].Code != OpReLU {
		t.Fatalf("expected OpReLU, got %d", prog.Ops[1].Code)
	}
}

// --- emitCustomBlockIR: reshape with symbolic params ---

func TestEmitCustomBlockIR_ReshapeWithIntLiterals(t *testing.T) {
	spec := BlockSpec{
		Type: "custom",
		Name: "reshape_int",
		Weights: []WeightSpec{
			{Name: "w", Shape: []string{"D", "D"}},
		},
		Ops: []OpSpec{
			{Op: "matmul", Inputs: []string{"x", "w"}, Output: "h"},
			{Op: "reshape", Inputs: []string{"h"}, Output: "h_r", Params: map[string]interface{}{
				"shape": []interface{}{"B", "T", "H", "HD"},
			}},
			{Op: "reshape", Inputs: []string{"h_r"}, Output: "x", Params: map[string]interface{}{
				"shape": []interface{}{"BT", "D"},
			}},
		},
	}
	prog := NewProgram(1)
	// D=128, H=4, T=64, B=2, V=1024
	_, err := emitCustomBlockIR(prog, spec, "x", 0, 128, 64, 2, 1024, 0)
	if err != nil {
		t.Fatalf("emitCustomBlockIR: %v", err)
	}
	reshapeOp := prog.Ops[1]
	if reshapeOp.Code != OpReshape {
		t.Fatalf("expected OpReshape, got %d", reshapeOp.Code)
	}
	// With heads=1 (default), HD = 128/1 = 128; H = 1
	// shape should be [B=2, T=64, H=1, HD=128]
	if len(reshapeOp.IntParams) != 4 {
		t.Fatalf("expected 4 int params, got %d", len(reshapeOp.IntParams))
	}
	if reshapeOp.IntParams[0] != 2 || reshapeOp.IntParams[1] != 64 {
		t.Errorf("reshape params B,T = %d,%d, want 2,64", reshapeOp.IntParams[0], reshapeOp.IntParams[1])
	}
}

// --- emitCustomBlockIR: transpose with axes param ---

func TestEmitCustomBlockIR_TransposeOp(t *testing.T) {
	spec := BlockSpec{
		Type: "custom",
		Name: "transpose_test",
		Weights: []WeightSpec{
			{Name: "w", Shape: []string{"D", "D"}},
		},
		Ops: []OpSpec{
			{Op: "matmul", Inputs: []string{"x", "w"}, Output: "h"},
			{Op: "transpose", Inputs: []string{"h"}, Output: "h_t", Params: map[string]interface{}{
				"axes": []interface{}{float64(1), float64(0)},
			}},
			{Op: "add", Inputs: []string{"x", "h_t"}, Output: "x"},
		},
	}
	prog := NewProgram(1)
	_, err := emitCustomBlockIR(prog, spec, "x", 0, 128, 64, 1, 1024, 0)
	if err != nil {
		t.Fatalf("emitCustomBlockIR: %v", err)
	}
	transposeOp := prog.Ops[1]
	if transposeOp.Code != OpTranspose {
		t.Fatalf("expected OpTranspose, got %d", transposeOp.Code)
	}
	if len(transposeOp.IntParams) != 2 || transposeOp.IntParams[0] != 1 || transposeOp.IntParams[1] != 0 {
		t.Errorf("transpose axes = %v, want [1 0]", transposeOp.IntParams)
	}
}

// --- emitCustomBlockIR: softmax with axis param ---

func TestEmitCustomBlockIR_SoftmaxWithAxis(t *testing.T) {
	spec := BlockSpec{
		Type: "custom",
		Name: "softmax_test",
		Weights: []WeightSpec{
			{Name: "w", Shape: []string{"D", "D"}},
		},
		Ops: []OpSpec{
			{Op: "matmul", Inputs: []string{"x", "w"}, Output: "h"},
			{Op: "softmax", Inputs: []string{"h"}, Output: "h_s", Params: map[string]interface{}{
				"axis": float64(-1),
			}},
			{Op: "add", Inputs: []string{"x", "h_s"}, Output: "x"},
		},
	}
	prog := NewProgram(1)
	_, err := emitCustomBlockIR(prog, spec, "x", 0, 128, 64, 1, 1024, 0)
	if err != nil {
		t.Fatalf("emitCustomBlockIR: %v", err)
	}
	softmaxOp := prog.Ops[1]
	if softmaxOp.Code != OpSoftmax {
		t.Fatalf("expected OpSoftmax, got %d", softmaxOp.Code)
	}
	if len(softmaxOp.IntParams) != 1 || softmaxOp.IntParams[0] != -1 {
		t.Errorf("softmax axis = %v, want [-1]", softmaxOp.IntParams)
	}
}

// --- emitCustomBlockIR: rmsnorm with eps param ---

func TestEmitCustomBlockIR_RMSNormWithEps(t *testing.T) {
	spec := BlockSpec{
		Type: "custom",
		Name: "rmsnorm_test",
		Weights: []WeightSpec{
			{Name: "scale", Shape: []string{"D"}},
		},
		Ops: []OpSpec{
			{Op: "rmsnorm", Inputs: []string{"x", "scale"}, Output: "x", Params: map[string]interface{}{
				"eps": float64(1e-5),
			}},
		},
	}
	prog := NewProgram(1)
	_, err := emitCustomBlockIR(prog, spec, "x", 0, 128, 64, 1, 1024, 0)
	if err != nil {
		t.Fatalf("emitCustomBlockIR: %v", err)
	}
	op := prog.Ops[0]
	if op.Code != OpRMSNorm {
		t.Fatalf("expected OpRMSNorm, got %d", op.Code)
	}
	if len(op.FloatParams) != 1 || op.FloatParams[0] != 1e-5 {
		t.Errorf("rmsnorm eps = %v, want [1e-5]", op.FloatParams)
	}
}

// --- emitCustomBlockIR: empty name defaults to "custom" ---

func TestEmitCustomBlockIR_EmptyName(t *testing.T) {
	spec := BlockSpec{
		Type: "custom",
		Name: "",
		Weights: []WeightSpec{
			{Name: "w", Shape: []string{"D", "D"}},
		},
		Ops: []OpSpec{
			{Op: "matmul", Inputs: []string{"x", "w"}, Output: "h"},
			{Op: "add", Inputs: []string{"x", "h"}, Output: "x"},
		},
	}
	prog := NewProgram(1)
	_, err := emitCustomBlockIR(prog, spec, "x", 0, 128, 64, 1, 1024, 0)
	if err != nil {
		t.Fatalf("emitCustomBlockIR: %v", err)
	}
	if len(prog.Ops) != 2 {
		t.Fatalf("expected 2 ops, got %d", len(prog.Ops))
	}
}

// --- emitCustomBlockIR: heads > 1 affects HD resolution ---

func TestEmitCustomBlockIR_WithHeads(t *testing.T) {
	spec := BlockSpec{
		Type:  "custom",
		Name:  "headed",
		Heads: 4,
		Weights: []WeightSpec{
			{Name: "w", Shape: []string{"D", "HD"}},
		},
		Ops: []OpSpec{
			{Op: "matmul", Inputs: []string{"x", "w"}, Output: "h"},
			{Op: "add", Inputs: []string{"x", "h"}, Output: "x"},
		},
	}
	prog := NewProgram(1)
	// D=128, H=4 -> HD=32
	_, err := emitCustomBlockIR(prog, spec, "x", 0, 128, 64, 1, 1024, 0)
	if err != nil {
		t.Fatalf("emitCustomBlockIR: %v", err)
	}
	if len(prog.Ops) != 2 {
		t.Fatalf("expected 2 ops, got %d", len(prog.Ops))
	}
}
