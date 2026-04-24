package arch

import (
	"os"
	"path/filepath"
	"testing"
)

func TestBuildIRProgramFromConfig_Plain3L(t *testing.T) {
	cfg, err := LoadArchConfig(filepath.Join("examples", "plain_3L.json"))
	if err != nil {
		t.Fatalf("LoadArchConfig: %v", err)
	}
	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig: %v", err)
	}

	// plain_3L: 3*(plain+swiglu) = 3*(7+4) + 3 base = 36
	if prog.NumWeights != 36 {
		t.Fatalf("expected 36 weights, got %d", prog.NumWeights)
	}

	if len(prog.Inputs) != 2 {
		t.Fatalf("expected 2 inputs, got %d", len(prog.Inputs))
	}
	if len(prog.Outputs) != 4 || prog.Outputs[0].Name != "loss" || prog.Outputs[1].Name != "per_token_nll" || prog.Outputs[2].Name != "x_hidden" || prog.Outputs[3].Name != "logits" {
		t.Fatalf("expected outputs [loss per_token_nll x_hidden logits], got %+v", prog.Outputs)
	}
}

func TestCountIRWeightsFromConfig_AllExamples(t *testing.T) {
	examples := []struct {
		name        string
		wantWeights int
	}{
		{"plain_3L.json", 36},
	}

	for _, tt := range examples {
		t.Run(tt.name, func(t *testing.T) {
			cfg, err := LoadArchConfig(filepath.Join("examples", tt.name))
			if err != nil {
				t.Fatalf("LoadArchConfig: %v", err)
			}
			n, err := CountIRWeightsFromConfig(cfg)
			if err != nil {
				t.Fatalf("CountIRWeightsFromConfig: %v", err)
			}
			if n != tt.wantWeights {
				t.Fatalf("expected %d weights, got %d", tt.wantWeights, n)
			}
		})
	}
}

func TestBuildIRProgramFromConfig_NilConfig(t *testing.T) {
	_, err := BuildIRProgramFromConfig(nil)
	if err == nil {
		t.Fatal("expected error for nil config")
	}
}

func TestCountIRWeightsFromConfig_NilConfig(t *testing.T) {
	_, err := CountIRWeightsFromConfig(nil)
	if err == nil {
		t.Fatal("expected error for nil config")
	}
}

func TestBuildIRProgramFromConfig_BatchSizeDerivation(t *testing.T) {
	// batch_tokens=1024, seq_len=128 -> batchSize=8
	cfg, err := LoadArchConfig(filepath.Join("examples", "plain_3L.json"))
	if err != nil {
		t.Fatalf("LoadArchConfig: %v", err)
	}
	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig: %v", err)
	}
	// tokens input shape should be [B, T] = [8, 128]
	if len(prog.Inputs) < 1 {
		t.Fatal("no inputs")
	}
	tokShape := prog.Inputs[0].Shape
	if tokShape[0] != 8 || tokShape[1] != 128 {
		t.Fatalf("expected tokens shape [8,128], got %v", tokShape)
	}
}

func TestAllExampleConfigsBuild(t *testing.T) {
	entries, err := os.ReadDir("../examples")
	if err != nil {
		t.Fatalf("ReadDir examples: %v", err)
	}
	for _, e := range entries {
		if e.IsDir() || filepath.Ext(e.Name()) != ".json" {
			continue
		}
		t.Run(e.Name(), func(t *testing.T) {
			cfg, err := LoadArchConfig(filepath.Join("examples", e.Name()))
			if err != nil {
				t.Fatalf("LoadArchConfig: %v", err)
			}
			prog, err := BuildIRProgramFromConfig(cfg)
			if err != nil {
				t.Fatalf("BuildIRProgramFromConfig: %v", err)
			}
			if prog.NumWeights <= 0 {
				t.Fatal("program has no weights")
			}
			if len(prog.Ops) == 0 {
				t.Fatal("program has no ops")
			}
		})
	}
}

func TestConvertBlockSpecs(t *testing.T) {
	specs := []BlockSpec{
		{Type: "plain", Heads: 4, KVHeads: 2},
		{Type: "swiglu"},
	}
	converted := convertBlockSpecs(specs)
	if len(converted) != 2 {
		t.Fatalf("expected 2, got %d", len(converted))
	}
	if converted[0].Type != "plain" || converted[0].Heads != 4 || converted[0].KVHeads != 2 {
		t.Fatalf("unexpected: %+v", converted[0])
	}
	if converted[1].Type != "swiglu" {
		t.Fatalf("unexpected: %+v", converted[1])
	}
}

func TestConvertBlockSpecs_Nil(t *testing.T) {
	if got := convertBlockSpecs(nil); got != nil {
		t.Fatalf("expected nil, got %v", got)
	}
}

func TestBuildIRProgramFromConfig_LogitSoftcap(t *testing.T) {
	cfg := &ArchConfig{
		ModelDim:     64,
		VocabSize:    256,
		SeqLen:       32,
		LogitSoftcap: 8,
		Blocks:       []BlockSpec{{Type: "plain", Heads: 4}},
		Training:     TrainingSpec{BatchTokens: 32},
	}
	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig: %v", err)
	}
	tanhCount := 0
	for _, op := range prog.Ops {
		if op.Code == OpTanh {
			tanhCount++
		}
	}
	if tanhCount != 1 {
		t.Fatalf("expected 1 Tanh op, got %d", tanhCount)
	}
}

func TestBuildIRProgramFromConfig_TiedEmbeddings(t *testing.T) {
	cfg := &ArchConfig{
		ModelDim:      64,
		VocabSize:     256,
		SeqLen:        32,
		TieEmbeddings: true,
		Blocks:        []BlockSpec{{Type: "plain", Heads: 4}},
		Training:      TrainingSpec{BatchTokens: 32},
	}
	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig: %v", err)
	}
	if prog.NumWeights != 9 {
		t.Fatalf("expected 9 weights, got %d", prog.NumWeights)
	}
	hasTranspose := false
	for _, op := range prog.Ops {
		if op.Code == OpTranspose && len(op.Inputs) == 1 && op.Inputs[0] == "w0" {
			hasTranspose = true
			break
		}
	}
	if !hasTranspose {
		t.Fatal("missing tied embedding transpose in IR program")
	}
}
