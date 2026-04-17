package train

import (
	"reflect"
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

func TestHiddenstatsOutputShapeUsesConfigSeqLen(t *testing.T) {
	const (
		batchTokens = 2048
		seqLen      = 2048
		modelDim    = 448
	)

	cfg := &ArchConfig{
		ModelDim:  modelDim,
		VocabSize: 1024,
		SeqLen:    seqLen,
		Blocks:    []BlockSpec{{Type: "plain", Heads: 7}},
		Training:  TrainingSpec{BatchTokens: batchTokens},
	}
	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig: %v", err)
	}

	wantShape := []int{1, seqLen, modelDim}
	if got := hiddenStatsOutputShape(batchTokens/seqLen, seqLen, modelDim); !reflect.DeepEqual(got, wantShape) {
		t.Fatalf("hiddenStatsOutputShape = %v, want %v", got, wantShape)
	}

	var hiddenShape []int
	for _, out := range prog.Outputs {
		if out.Name == "x_hidden" {
			hiddenShape = out.Shape
			break
		}
	}
	if !reflect.DeepEqual(hiddenShape, wantShape) {
		t.Fatalf("x_hidden output shape = %v, want %v", hiddenShape, wantShape)
	}

	foundReshape := false
	for _, op := range prog.Ops {
		if op.Code == ir.OpReshape && len(op.Inputs) == 1 && op.Inputs[0] == "x_final_norm" &&
			len(op.Outputs) == 1 && op.Outputs[0] == "x_hidden" &&
			reflect.DeepEqual(op.IntParams, wantShape) {
			foundReshape = true
			break
		}
	}
	if !foundReshape {
		t.Fatalf("missing x_final_norm -> x_hidden reshape to %v", wantShape)
	}
}
