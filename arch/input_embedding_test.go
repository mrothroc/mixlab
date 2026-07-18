package arch

import (
	"reflect"
	"testing"
)

func TestEmitDiscreteTokenInputIRSimpleLegacyShape(t *testing.T) {
	prog := NewProgram(2)
	next, err := emitDiscreteTokenInputIR(prog, discreteTokenInputOptions{
		BatchSize:           2,
		SeqLen:              4,
		ModelDim:            8,
		TokenWeightIndex:    0,
		NextWeightIndex:     2,
		PositionalEmbedding: PositionalEmbeddingRope,
		Smear:               disabledSmearEmbeddingOptions(),
	})
	if err != nil {
		t.Fatal(err)
	}
	if next != 2 {
		t.Fatalf("next weight index=%d, want 2", next)
	}
	want := []Op{
		{Code: OpEmbed, Inputs: []string{"w0", "tokens"}, Outputs: []string{"x_embed"}},
		{Code: OpReshape, Inputs: []string{"x_embed"}, Outputs: []string{"x"}, IntParams: []int{8, 8}},
	}
	if !reflect.DeepEqual(prog.Ops, want) {
		t.Fatalf("ops=%+v\nwant=%+v", prog.Ops, want)
	}
}

func TestEmitDiscreteTokenInputIRPreservesFeatureOrder(t *testing.T) {
	prog := NewProgram(12)
	next, err := emitDiscreteTokenInputIR(prog, discreteTokenInputOptions{
		BatchSize:           2,
		SeqLen:              4,
		ModelDim:            8,
		TokenWeightIndex:    0,
		NextWeightIndex:     3,
		PositionalEmbedding: PositionalEmbeddingLearnedAbsolute,
		MaxPositions:        8,
		Smear:               disabledSmearEmbeddingOptions(),
		CharVocabSize:       16,
		CharDim:             4,
		CharMaxPerToken:     2,
		BigramVocabSize:     32,
		BigramDim:           8,
		TrigramVocabSize:    64,
		TrigramDim:          4,
		EmbeddingDropout:    0.1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if next != 12 {
		t.Fatalf("next weight index=%d, want 12", next)
	}
	wantCodes := []int{
		OpEmbed,
		OpArange, OpEmbed, OpReshape, OpFull, OpMul, OpAdd,
		OpReshape,
		OpCharFeatureBag, OpMatMul, OpMul, OpAdd,
		OpEmbed, OpReshape, OpMul, OpAdd,
		OpEmbed, OpReshape, OpMatMul, OpMul, OpAdd,
		OpDropout,
	}
	gotCodes := make([]int, len(prog.Ops))
	for i, op := range prog.Ops {
		gotCodes[i] = op.Code
	}
	if !reflect.DeepEqual(gotCodes, wantCodes) {
		t.Fatalf("op codes=%v\nwant=%v", gotCodes, wantCodes)
	}
	for _, want := range []struct {
		op     int
		weight string
	}{
		{0, "w0"},
		{2, "w3"},
		{8, "w4"},
		{9, "w5"},
		{10, "w6"},
		{12, "w7"},
		{14, "w8"},
		{16, "w9"},
		{18, "w10"},
		{19, "w11"},
	} {
		found := false
		for _, input := range prog.Ops[want.op].Inputs {
			if input == want.weight {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("op %d inputs=%v, want weight %s", want.op, prog.Ops[want.op].Inputs, want.weight)
		}
	}
}

func TestEmitDiscreteTokenInputIRRejectsInvalidShape(t *testing.T) {
	if _, err := emitDiscreteTokenInputIR(NewProgram(1), discreteTokenInputOptions{}); err == nil {
		t.Fatal("expected invalid shape error")
	}
}
