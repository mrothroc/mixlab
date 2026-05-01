package arch

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

func smearEmbeddingsTestConfig() *ArchConfig {
	return &ArchConfig{
		Name:      "smear_embeddings",
		ModelDim:  16,
		VocabSize: 64,
		SeqLen:    8,
		Blocks:    []BlockSpec{{Type: "plain", Heads: 4}},
		Training:  TrainingSpec{BatchTokens: 8, Steps: 10, LR: 1e-3},
	}
}

func TestParseArchConfig_SmearEmbeddings(t *testing.T) {
	data := []byte(`{
		"name": "smear",
		"model_dim": 16,
		"vocab_size": 64,
		"seq_len": 8,
		"smear_embeddings": true,
		"blocks": [{"type": "plain", "heads": 4}],
		"training": {"batch_tokens": 8}
	}`)
	cfg, err := ParseArchConfig(data, "smear")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	if !cfg.SmearEmbeddings {
		t.Fatal("SmearEmbeddings=false, want true")
	}
	if got := cfg.EffectiveSmearEmbeddingsGateShape(); got != SmearEmbeddingsGatePR130 {
		t.Fatalf("EffectiveSmearEmbeddingsGateShape=%q want %q", got, SmearEmbeddingsGatePR130)
	}
	out, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	if !strings.Contains(string(out), "smear_embeddings") ||
		!strings.Contains(string(out), SmearEmbeddingsGatePR130) {
		t.Fatalf("round-trip JSON missing smear fields: %s", out)
	}
}

func TestParseArchConfig_RejectsInvalidSmearEmbeddings(t *testing.T) {
	tests := []struct {
		name    string
		patch   func(*ArchConfig)
		wantErr string
	}{
		{
			name: "bad_gate_shape",
			patch: func(cfg *ArchConfig) {
				cfg.SmearEmbeddingsGateShape = "wide"
			},
			wantErr: "smear_embeddings_gate_shape",
		},
		{
			name: "short_sequence",
			patch: func(cfg *ArchConfig) {
				cfg.SeqLen = 1
			},
			wantErr: "seq_len",
		},
		{
			name: "pr130_small_model_dim",
			patch: func(cfg *ArchConfig) {
				cfg.ModelDim = 8
			},
			wantErr: "model_dim",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := smearEmbeddingsTestConfig()
			cfg.SmearEmbeddings = true
			tt.patch(cfg)
			data, err := json.Marshal(cfg)
			if err != nil {
				t.Fatalf("Marshal: %v", err)
			}
			_, err = ParseArchConfig(data, tt.name)
			if err == nil {
				t.Fatal("expected error")
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("error %q does not contain %q", err.Error(), tt.wantErr)
			}
		})
	}
}

func TestBuildIRProgram_SmearEmbeddingsDisabledMatchesBase(t *testing.T) {
	base := smearEmbeddingsTestConfig()
	baseProg, err := BuildIRProgramFromConfig(base)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig(base): %v", err)
	}
	withFalse := smearEmbeddingsTestConfig()
	withFalse.SmearEmbeddings = false
	withFalse.SmearEmbeddingsGateShape = ""
	withFalseProg, err := BuildIRProgramFromConfig(withFalse)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig(withFalse): %v", err)
	}
	if !reflect.DeepEqual(baseProg, withFalseProg) {
		t.Fatal("disabled smear_embeddings changed the IR program")
	}
}

func TestSmearEmbeddingsWeightShapes_PR130(t *testing.T) {
	cfg := smearEmbeddingsTestConfig()
	cfg.SmearEmbeddings = true

	metas, err := CollectWeightShapesFromConfig(cfg)
	if err != nil {
		t.Fatalf("CollectWeightShapesFromConfig: %v", err)
	}
	fixed := fixedWeightCountWithHead(cfg.ReservesUntiedHeadWeight())
	if got, want := metas[fixed].Name, "smear_gate"; got != want {
		t.Fatalf("weight[%d].Name=%q want %q", fixed, got, want)
	}
	if got, want := metas[fixed].Shape, []int{12, 1}; !reflect.DeepEqual(got, want) {
		t.Fatalf("smear_gate shape=%v want %v", got, want)
	}
	if !metas[fixed].InitZero {
		t.Fatalf("smear_gate should be zero-initialized: %+v", metas[fixed])
	}
	if got, want := metas[fixed+1].Name, "smear_scale"; got != want {
		t.Fatalf("weight[%d].Name=%q want %q", fixed+1, got, want)
	}
	if got, want := metas[fixed+1].Shape, []int{1}; !reflect.DeepEqual(got, want) {
		t.Fatalf("smear_scale shape=%v want %v", got, want)
	}
	if !metas[fixed+1].InitZero {
		t.Fatalf("smear_scale should be zero-initialized: %+v", metas[fixed+1])
	}
	n, err := CountIRWeightsFromConfig(cfg)
	if err != nil {
		t.Fatalf("CountIRWeightsFromConfig: %v", err)
	}
	if n != len(metas) {
		t.Fatalf("CountIRWeightsFromConfig=%d want %d", n, len(metas))
	}
	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig: %v", err)
	}
	if prog.NumWeights != len(metas) {
		t.Fatalf("prog.NumWeights=%d want %d", prog.NumWeights, len(metas))
	}
}

func TestSmearEmbeddingsWeightShapes_StaticVariants(t *testing.T) {
	tests := []struct {
		gateShape string
		wantShape []int
	}{
		{SmearEmbeddingsGatePerChannel, []int{16}},
		{SmearEmbeddingsGatePerPositionPerChannel, []int{8, 16}},
	}
	for _, tt := range tests {
		t.Run(tt.gateShape, func(t *testing.T) {
			cfg := smearEmbeddingsTestConfig()
			cfg.SmearEmbeddings = true
			cfg.SmearEmbeddingsGateShape = tt.gateShape
			metas, err := CollectWeightShapesFromConfig(cfg)
			if err != nil {
				t.Fatalf("CollectWeightShapesFromConfig: %v", err)
			}
			fixed := fixedWeightCountWithHead(cfg.ReservesUntiedHeadWeight())
			if got, want := metas[fixed].Name, "smear_gate"; got != want {
				t.Fatalf("weight[%d].Name=%q want %q", fixed, got, want)
			}
			if got := metas[fixed].Shape; !reflect.DeepEqual(got, tt.wantShape) {
				t.Fatalf("smear_gate shape=%v want %v", got, tt.wantShape)
			}
			if !metas[fixed].InitZero {
				t.Fatalf("smear_gate should be zero-initialized: %+v", metas[fixed])
			}
		})
	}
}

func TestBuildIRProgram_SmearEmbeddingsLeavesBOSUnchanged(t *testing.T) {
	cfg := smearEmbeddingsTestConfig()
	cfg.SmearEmbeddings = true
	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig: %v", err)
	}

	foundFirstSlice := false
	foundConcat := false
	foundGateMatMul := false
	for _, op := range prog.Ops {
		if op.Code == OpSlice && len(op.Inputs) == 1 && op.Inputs[0] == "x_embed" &&
			len(op.Outputs) == 1 && op.Outputs[0] == "x_embed_smear_first" &&
			reflect.DeepEqual(op.IntParams, []int{0, 1, 1, 1}) {
			foundFirstSlice = true
		}
		if op.Code == OpConcat && len(op.Inputs) == 2 &&
			op.Inputs[0] == "x_embed_smear_first" &&
			op.Inputs[1] == "x_embed_smear_tail_out" &&
			len(op.Outputs) == 1 && op.Outputs[0] == "x_embed_smeared" &&
			reflect.DeepEqual(op.IntParams, []int{1}) {
			foundConcat = true
		}
		if op.Code == OpMatMul && len(op.Inputs) == 2 &&
			op.Inputs[0] == "x_embed_smear_gate_input" &&
			op.Inputs[1] == "w3" {
			foundGateMatMul = true
		}
	}
	if !foundFirstSlice {
		t.Fatal("missing BOS-preserving first-token slice")
	}
	if !foundConcat {
		t.Fatal("missing concat that reuses the unmodified first-token slice")
	}
	if !foundGateMatMul {
		t.Fatal("missing PR130 dynamic smear gate matmul")
	}
}
