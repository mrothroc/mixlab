package train

import (
	"os"
	"path/filepath"
	"testing"
)

// embeddedPlainConfig is a minimal config embedded in the test binary so that
// core smoke tests work even without the examples/ directory on disk.
const embeddedPlainConfig = `{
  "name": "embedded_plain",
  "model_dim": 64,
  "vocab_size": 256,
  "seq_len": 32,
  "blocks": [
    {"type": "plain", "heads": 2},
    {"type": "swiglu"}
  ],
  "training": {"steps": 10, "lr": 3e-4, "seed": 1, "batch_tokens": 64}
}`

// TestSmokeEmbeddedConfig verifies that a minimal config embedded in the binary
// parses and builds a valid IR program — no external files needed.
func TestSmokeEmbeddedConfig(t *testing.T) {
	cfg, err := ParseArchConfig([]byte(embeddedPlainConfig), "embedded")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig: %v", err)
	}
	if prog.NumWeights <= 0 {
		t.Error("program has no weights")
	}
	if len(prog.Ops) == 0 {
		t.Error("program has no ops")
	}
	counted, err := CountIRWeightsFromConfig(cfg)
	if err != nil {
		t.Fatalf("CountIRWeightsFromConfig: %v", err)
	}
	if counted != prog.NumWeights {
		t.Errorf("CountIRWeights=%d vs prog.NumWeights=%d", counted, prog.NumWeights)
	}
}

// exampleConfigCase describes a single example config for table-driven tests.
type exampleConfigCase struct {
	filename    string
	wantWeights int
	minOps      int // lower bound on expected IR ops
}

// exampleConfigs is the canonical table of example configs and their expected properties.
//
// Configs marked "needs block impl" will fail until the corresponding block type
// is registered in validateBlockSpec (arch/config.go) and emitBlockIR (arch/builder.go).
// See: mamba, retnet, rwkv, perceiver, custom.
var exampleConfigs = []exampleConfigCase{
	{
		filename:    "plain_3L.json",
		wantWeights: 36,
		minOps:      10,
	},
	{
		filename:    "gqa_8h4kv.json",
		wantWeights: 36,
		minOps:      10,
	},
	{
		filename:    "token_blend_plain.json",
		wantWeights: 27,
		minOps:      10,
	},
	{
		filename:    "bigram_plain.json",
		wantWeights: 39,
		minOps:      12,
	},
	{
		filename:    "softcap_plain.json",
		wantWeights: 36,
		minOps:      10,
	},
	// --- New block types (needs block impl) ---
	{
		filename:    "mamba_2L.json",
		wantWeights: 19, // 3 (embed+head+norm) + 2*(4 mamba + 4 swiglu) = 19
		minOps:      8,
	},
	{
		filename:    "retnet_2L.json",
		wantWeights: 27, // 3 base + retnet+swiglu blocks
		minOps:      8,
	},
	{
		filename:    "rwkv_2L.json",
		wantWeights: 31, // 3 base + rwkv+swiglu blocks
		minOps:      8,
	},
	{
		filename:    "perceiver_2L.json",
		wantWeights: 41, // 3 base + perceiver+swiglu blocks
		minOps:      8,
	},
	{
		filename:    "custom_geglu.json",
		wantWeights: 23, // 3 base + 2*7 plain + 2*3 custom geglu
		minOps:      10,
	},
}

// TestSmokeExampleConfigs_Parse verifies that every example config parses
// and validates successfully.
func TestSmokeExampleConfigs_Parse(t *testing.T) {
	for _, tc := range exampleConfigs {
		t.Run(tc.filename, func(t *testing.T) {
			cfg, err := LoadArchConfig(filepath.Join("examples", tc.filename))
			if err != nil {
				t.Fatalf("LoadArchConfig(%s): %v", tc.filename, err)
			}
			if cfg.ModelDim <= 0 {
				t.Errorf("model_dim should be positive, got %d", cfg.ModelDim)
			}
			if cfg.VocabSize <= 0 {
				t.Errorf("vocab_size should be positive, got %d", cfg.VocabSize)
			}
			if cfg.SeqLen <= 0 {
				t.Errorf("seq_len should be positive, got %d", cfg.SeqLen)
			}
		})
	}
}

// TestSmokeExampleConfigs_BuildIR verifies that every example config can be
// lowered into a well-formed IR program with the expected number of weights.
func TestSmokeExampleConfigs_BuildIR(t *testing.T) {
	for _, tc := range exampleConfigs {
		t.Run(tc.filename, func(t *testing.T) {
			cfg, err := LoadArchConfig(filepath.Join("examples", tc.filename))
			if err != nil {
				t.Fatalf("LoadArchConfig: %v", err)
			}

			prog, err := BuildIRProgramFromConfig(cfg)
			if err != nil {
				t.Fatalf("BuildIRProgramFromConfig: %v", err)
			}

			// Check weight count matches expected.
			if prog.NumWeights != tc.wantWeights {
				t.Errorf("NumWeights = %d, want %d", prog.NumWeights, tc.wantWeights)
			}

			// Check ops are non-empty and exceed minimum.
			if len(prog.Ops) < tc.minOps {
				t.Errorf("len(Ops) = %d, want >= %d", len(prog.Ops), tc.minOps)
			}

			// Check inputs: plain configs use tokens+targets, bigram configs add bigram_ids.
			wantInputs := 2
			if cfg.BigramVocabSize > 0 {
				wantInputs = 3
			}
			if len(prog.Inputs) != wantInputs {
				t.Fatalf("expected %d inputs, got %d", wantInputs, len(prog.Inputs))
			}
			if prog.Inputs[0].Name != "tokens" {
				t.Errorf("input[0].Name = %q, want \"tokens\"", prog.Inputs[0].Name)
			}
			if prog.Inputs[1].Name != "targets" {
				t.Errorf("input[1].Name = %q, want \"targets\"", prog.Inputs[1].Name)
			}
			if cfg.BigramVocabSize > 0 && prog.Inputs[2].Name != "bigram_ids" {
				t.Errorf("input[2].Name = %q, want \"bigram_ids\"", prog.Inputs[2].Name)
			}

			// Check outputs: loss scalar plus hidden-state/logit exports.
			if len(prog.Outputs) != 3 {
				t.Fatalf("expected 3 outputs, got %d", len(prog.Outputs))
			}
			if prog.Outputs[0].Name != "loss" {
				t.Errorf("output[0].Name = %q, want \"loss\"", prog.Outputs[0].Name)
			}
			if prog.Outputs[1].Name != "x_hidden" {
				t.Errorf("output[1].Name = %q, want \"x_hidden\"", prog.Outputs[1].Name)
			}
			if prog.Outputs[2].Name != "logits" {
				t.Errorf("output[2].Name = %q, want \"logits\"", prog.Outputs[2].Name)
			}
		})
	}
}

// TestSmokeExampleConfigs_CountWeightsMatch verifies that CountIRWeightsFromConfig
// agrees with the IR program's NumWeights for every example config.
func TestSmokeExampleConfigs_CountWeightsMatch(t *testing.T) {
	for _, tc := range exampleConfigs {
		t.Run(tc.filename, func(t *testing.T) {
			cfg, err := LoadArchConfig(filepath.Join("examples", tc.filename))
			if err != nil {
				t.Fatalf("LoadArchConfig: %v", err)
			}

			prog, err := BuildIRProgramFromConfig(cfg)
			if err != nil {
				t.Fatalf("BuildIRProgramFromConfig: %v", err)
			}

			counted, err := CountIRWeightsFromConfig(cfg)
			if err != nil {
				t.Fatalf("CountIRWeightsFromConfig: %v", err)
			}

			if counted != prog.NumWeights {
				t.Errorf("CountIRWeightsFromConfig = %d, prog.NumWeights = %d (mismatch)",
					counted, prog.NumWeights)
			}
		})
	}
}

// TestSmokeExampleConfigs_WeightShapes verifies that computeWeightShapes
// produces the same count as the IR program for each example config.
func TestSmokeExampleConfigs_WeightShapes(t *testing.T) {
	for _, tc := range exampleConfigs {
		t.Run(tc.filename, func(t *testing.T) {
			cfg, err := LoadArchConfig(filepath.Join("examples", tc.filename))
			if err != nil {
				t.Fatalf("LoadArchConfig: %v", err)
			}

			shapes, err := computeWeightShapes(cfg)
			if err != nil {
				t.Fatalf("computeWeightShapes: %v", err)
			}

			if len(shapes) != tc.wantWeights {
				t.Errorf("len(shapes) = %d, want %d", len(shapes), tc.wantWeights)
			}

			// Every shape should have at least one dimension > 0.
			for i, ws := range shapes {
				if len(ws.Shape) == 0 {
					t.Errorf("weight %d (%s) has empty shape", i, ws.Name)
					continue
				}
				for _, d := range ws.Shape {
					if d <= 0 {
						t.Errorf("weight %d (%s) has invalid dimension %d in shape %v",
							i, ws.Name, d, ws.Shape)
					}
				}
			}
		})
	}
}

// TestSmokeAllExampleConfigs_Discovery discovers all JSON files in the examples
// directory and verifies they parse + build into IR. This catches newly added
// configs that are not yet in the table above.
func TestSmokeAllExampleConfigs_Discovery(t *testing.T) {
	entries, err := os.ReadDir("../examples")
	if err != nil {
		t.Fatalf("ReadDir examples: %v", err)
	}

	found := 0
	for _, e := range entries {
		if e.IsDir() || filepath.Ext(e.Name()) != ".json" {
			continue
		}
		found++
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
				t.Error("program has no weights")
			}
			if len(prog.Ops) == 0 {
				t.Error("program has no ops")
			}
			if len(prog.Inputs) == 0 {
				t.Error("program has no inputs")
			}
			if len(prog.Outputs) == 0 {
				t.Error("program has no outputs")
			}
		})
	}

	if found == 0 {
		t.Fatal("no example JSON configs found in examples/")
	}
}
