package arch

import (
	"reflect"
	"testing"
)

func TestBlockDiffusionAttentionMaskKindOverridesPlainMask(t *testing.T) {
	spec := BlockSpec{Type: "plain", Heads: 2, AttentionMask: AttentionMaskCausal}
	if got := resolvedPlainAttentionMaskForObjective(spec, ObjectiveBlockDiffusion, ObjectiveBlockDiffusion); got != AttentionMaskBlockDiffusion {
		t.Fatalf("resolved mask=%q, want %q", got, AttentionMaskBlockDiffusion)
	}

	blocks := resolveBlockAttentionMasksForObjective([]BlockSpec{spec}, ObjectiveBlockDiffusion, ObjectiveBlockDiffusion)
	if len(blocks) != 1 {
		t.Fatalf("resolved blocks len=%d, want 1", len(blocks))
	}
	if got := blocks[0].AttentionMask; got != AttentionMaskBlockDiffusion {
		t.Fatalf("resolved block attention_mask=%q, want %q", got, AttentionMaskBlockDiffusion)
	}
}

func TestBlockDiffusionMaskDispatchDeclaresBoundaryInputs(t *testing.T) {
	p := NewProgram(1)
	got, err := emitPlainAttentionMaskIR(p, "scores", "masked", AttentionMaskBlockDiffusion, 2, 4, 0, false)
	if err != nil {
		t.Fatalf("emitPlainAttentionMaskIR: %v", err)
	}
	if got != "masked" {
		t.Fatalf("mask output=%q, want masked", got)
	}
	if len(p.Ops) != 1 {
		t.Fatalf("ops=%d, want 1", len(p.Ops))
	}
	op := p.Ops[0]
	if op.Code != OpBlockDiffusionMask {
		t.Fatalf("op code=%d, want %d", op.Code, OpBlockDiffusionMask)
	}
	if !reflect.DeepEqual(op.Inputs, []string{"scores", "diffusion_block_start", "diffusion_block_end"}) {
		t.Fatalf("op inputs=%v", op.Inputs)
	}
	if !reflect.DeepEqual(op.IntParams, []int{4}) {
		t.Fatalf("op int params=%v, want [4]", op.IntParams)
	}

	if _, err := emitPlainAttentionMaskIR(p, "scores2", "masked2", AttentionMaskBlockDiffusion, 2, 4, 0, false); err != nil {
		t.Fatalf("second emitPlainAttentionMaskIR: %v", err)
	}
	startInputs := 0
	endInputs := 0
	for _, in := range p.Inputs {
		switch in.Name {
		case "diffusion_block_start":
			startInputs++
			if in.DType != TensorInt32 || !reflect.DeepEqual(in.Shape, []int{2}) {
				t.Fatalf("diffusion_block_start decl=%+v", in)
			}
		case "diffusion_block_end":
			endInputs++
			if in.DType != TensorInt32 || !reflect.DeepEqual(in.Shape, []int{2}) {
				t.Fatalf("diffusion_block_end decl=%+v", in)
			}
		}
	}
	if startInputs != 1 || endInputs != 1 {
		t.Fatalf("boundary input counts start=%d end=%d, want 1/1", startInputs, endInputs)
	}
}
