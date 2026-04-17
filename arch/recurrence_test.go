package arch

import "testing"

func weightInputForOutput(t *testing.T, prog *Program, code int, output string) string {
	t.Helper()
	const weightInput = 1
	for _, op := range prog.Ops {
		if op.Code != code || len(op.Outputs) == 0 || op.Outputs[0] != output {
			continue
		}
		if len(op.Inputs) <= weightInput {
			t.Fatalf("op for %q has inputs %v, want weight input %d", output, op.Inputs, weightInput)
		}
		return op.Inputs[weightInput]
	}
	t.Fatalf("missing op code %d with output %q", code, output)
	return ""
}

func TestCountWeightsWithRecurrence_ReducesBlockWeights(t *testing.T) {
	blocks := []BlockSpec{
		{Type: "plain", Heads: 4},
		{Type: "swiglu"},
		{Type: "plain", Heads: 4},
		{Type: "swiglu"},
		{Type: "plain", Heads: 4},
		{Type: "swiglu"},
		{Type: "plain", Heads: 4},
		{Type: "swiglu"},
	}
	recurrence := []int{0, 1, 2, 3, 2, 3, 2, 3}

	got, err := CountWeightsWithBigramAndRecurrence(64, DefaultFFNMultiplier, false, false, false, false, 0, 0, blocks, recurrence)
	if err != nil {
		t.Fatalf("CountWeightsWithBigramAndRecurrence: %v", err)
	}
	if got != 25 {
		t.Fatalf("weight count=%d want 25", got)
	}

	untied, err := CountWeights(DefaultFFNMultiplier, false, false, false, false, blocks)
	if err != nil {
		t.Fatalf("CountWeights: %v", err)
	}
	if untied != 47 {
		t.Fatalf("untied weight count=%d want 47", untied)
	}
}

func TestBuildIRProgramWithRecurrence_ReusesWeightNames(t *testing.T) {
	blocks := []BlockSpec{
		{Type: "plain", Heads: 4},
		{Type: "swiglu"},
		{Type: "plain", Heads: 4},
		{Type: "swiglu"},
		{Type: "plain", Heads: 4},
		{Type: "swiglu"},
	}
	recurrence := []int{0, 1, 2, 3, 2, 3}

	prog, err := BuildIRProgramWithRecurrence(64, 256, 32, 1, DefaultFFNMultiplier, false, false, false, false, 0, blocks, recurrence)
	if err != nil {
		t.Fatalf("BuildIRProgramWithRecurrence: %v", err)
	}
	if prog.NumWeights != 25 {
		t.Fatalf("NumWeights=%d want 25", prog.NumWeights)
	}

	plainOriginal := weightInputForOutput(t, prog, OpRMSNorm, "x_attn_2_x_norm")
	plainCopy := weightInputForOutput(t, prog, OpRMSNorm, "x_attn_4_x_norm")
	if plainOriginal != "w14" || plainCopy != plainOriginal {
		t.Fatalf("plain recurrence weights original=%q copy=%q want both w14", plainOriginal, plainCopy)
	}

	swigluOriginal := weightInputForOutput(t, prog, OpRMSNorm, "x_swiglu_3_x_norm")
	swigluCopy := weightInputForOutput(t, prog, OpRMSNorm, "x_swiglu_5_x_norm")
	if swigluOriginal != "w21" || swigluCopy != swigluOriginal {
		t.Fatalf("swiglu recurrence weights original=%q copy=%q want both w21", swigluOriginal, swigluCopy)
	}
}

func TestBuildIRProgramWithRecurrence_BackwardCompatNil(t *testing.T) {
	blocks := []BlockSpec{
		{Type: "plain", Heads: 4},
		{Type: "swiglu"},
	}
	base, err := BuildIRProgram(64, 256, 32, 1, DefaultFFNMultiplier, false, false, false, false, 0, blocks)
	if err != nil {
		t.Fatalf("BuildIRProgram: %v", err)
	}
	withNil, err := BuildIRProgramWithRecurrence(64, 256, 32, 1, DefaultFFNMultiplier, false, false, false, false, 0, blocks, nil)
	if err != nil {
		t.Fatalf("BuildIRProgramWithRecurrence: %v", err)
	}
	if withNil.NumWeights != base.NumWeights || len(withNil.Ops) != len(base.Ops) {
		t.Fatalf("nil recurrence changed program: weights %d/%d ops %d/%d", withNil.NumWeights, base.NumWeights, len(withNil.Ops), len(base.Ops))
	}
}

func TestBuildIRProgramWithRecurrence_UNetDecoderCopiesEncoder(t *testing.T) {
	blocks := []BlockSpec{
		{Type: "plain", Heads: 4},
		{Type: "plain", Heads: 4},
		{Type: "plain", Heads: 4},
		{Type: "plain", Heads: 4},
	}
	recurrence := []int{0, 1, 0, 1}

	prog, err := BuildIRProgramWithRecurrence(64, 256, 32, 1, DefaultFFNMultiplier, false, false, false, true, 0, blocks, recurrence)
	if err != nil {
		t.Fatalf("BuildIRProgramWithRecurrence unet: %v", err)
	}
	if prog.NumWeights != 19 {
		t.Fatalf("NumWeights=%d want 19", prog.NumWeights)
	}

	enc0 := weightInputForOutput(t, prog, OpRMSNorm, "x_attn_0_x_norm")
	dec0 := weightInputForOutput(t, prog, OpRMSNorm, "x_attn_2_x_norm")
	enc1 := weightInputForOutput(t, prog, OpRMSNorm, "x_attn_1_x_norm")
	dec1 := weightInputForOutput(t, prog, OpRMSNorm, "x_attn_3_x_norm")
	if enc0 != "w3" || dec0 != enc0 {
		t.Fatalf("decoder block 2 weight=%q want encoder block 0 weight %q", dec0, enc0)
	}
	if enc1 != "w10" || dec1 != enc1 {
		t.Fatalf("decoder block 3 weight=%q want encoder block 1 weight %q", dec1, enc1)
	}
}
