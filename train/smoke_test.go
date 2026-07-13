package train

import (
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/mrothroc/mixlab/arch"
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

func TestSeqLenScheduleTrainingProgramsUseScheduledShapes(t *testing.T) {
	cfg, err := ParseArchConfig([]byte(`{
		"name": "seq_len_schedule_shape",
		"model_dim": 16,
		"vocab_size": 32,
		"seq_len": 8,
		"blocks": [
			{"type": "plain", "heads": 2},
			{"type": "swiglu"}
		],
		"training": {
			"steps": 4,
			"lr": 0.001,
			"batch_tokens": 16,
			"seq_len_schedule": [[0,4],[2,8]]
		}
	}`), "seq_len_schedule_shape")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}

	shortCfg := *cfg
	shortCfg.SeqLen = cfg.Training.EffectiveSeqLenForStep(cfg.SeqLen, 0)
	shortProg, err := BuildTrainingIRProgramFromConfig(&shortCfg, TrainingProgramState{RecurrenceActive: true})
	if err != nil {
		t.Fatalf("BuildTrainingIRProgramFromConfig(short): %v", err)
	}
	maxProg, err := BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{RecurrenceActive: true})
	if err != nil {
		t.Fatalf("BuildTrainingIRProgramFromConfig(max): %v", err)
	}
	if shortProg.NumWeights != maxProg.NumWeights {
		t.Fatalf("scheduled weight count mismatch: short=%d max=%d", shortProg.NumWeights, maxProg.NumWeights)
	}
	if got, want := outputShape(t, shortProg, "x_hidden"), []int{4, 4, 16}; !reflect.DeepEqual(got, want) {
		t.Fatalf("short x_hidden shape = %v, want %v", got, want)
	}
	if got, want := outputShape(t, maxProg, "x_hidden"), []int{2, 8, 16}; !reflect.DeepEqual(got, want) {
		t.Fatalf("max x_hidden shape = %v, want %v", got, want)
	}
	if got, want := outputShape(t, shortProg, "logits"), []int{16, 32}; !reflect.DeepEqual(got, want) {
		t.Fatalf("short logits shape = %v, want %v", got, want)
	}

	evalProg, err := BuildEvalIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildEvalIRProgramFromConfig: %v", err)
	}
	if got, want := outputShape(t, evalProg, "x_hidden"), []int{2, 8, 16}; !reflect.DeepEqual(got, want) {
		t.Fatalf("eval x_hidden shape = %v, want %v", got, want)
	}
}

func outputShape(t *testing.T, prog *Program, name string) []int {
	t.Helper()
	for _, out := range prog.Outputs {
		if out.Name == name {
			return out.Shape
		}
	}
	t.Fatalf("program missing output %q", name)
	return nil
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
		filename:    "char_features_plain.json",
		wantWeights: 39,
		minOps:      12,
	},
	{
		filename:    "softcap_plain.json",
		wantWeights: 36,
		minOps:      10,
	},
	{
		filename:    "qk_norm_tiny.json",
		wantWeights: 29, // 3 base + 2*(9 qk_norm plain + 4 swiglu)
		minOps:      10,
	},
	{
		filename:    "distillation_tiny.json",
		wantWeights: 25,
		minOps:      10,
	},
	{
		filename:    "deberta_relative_tiny.json",
		wantWeights: 31, // 3 base + 2*(10 relative plain + 4 swiglu)
		minOps:      10,
	},
	{
		filename:    "lamb_plain_tiny.json",
		wantWeights: 25, // 3 base + 2*(7 plain + 4 swiglu)
		minOps:      10,
	},
	{
		filename:    "packed_segment_mask_tiny.json",
		wantWeights: 24, // 2 tied-embed base + 2*(7 plain + 4 swiglu)
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
		filename:    "hgrn2_2L.json",
		wantWeights: 23, // 3 base + 2*(6 hgrn2 + 4 swiglu)
		minOps:      8,
	},
	{
		filename:    "ttt_mlp_tiny.json",
		wantWeights: 27, // 3 base + 20 ttt_mlp + 4 swiglu
		minOps:      10,
	},
	{
		filename:    "mlstm_2L.json",
		wantWeights: 33, // 3 base + 2*(11 mlstm + 4 swiglu)
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

			// Check inputs: feature configs append their runtime feature ids.
			wantInputs := 2
			if cfg.Training.Distillation != nil {
				wantInputs++
			}
			if cfg.Training.AttentionSegmentMaskEnabled() {
				wantInputs++
			}
			if cfg.CharVocabSize > 0 {
				wantInputs++
			}
			if cfg.BigramVocabSize > 0 {
				wantInputs++
			}
			if cfg.TrigramVocabSize > 0 {
				wantInputs++
			}
			hasTTTMLP := len(arch.TTTMLPInnerLRScalesForStep(cfg.Blocks, 0)) > 0
			if hasTTTMLP {
				wantInputs++
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
			inputIdx := 2
			if cfg.Training.AttentionSegmentMaskEnabled() {
				if prog.Inputs[inputIdx].Name != "segment_ids" {
					t.Errorf("input[%d].Name = %q, want \"segment_ids\"", inputIdx, prog.Inputs[inputIdx].Name)
				}
				inputIdx++
			}
			if cfg.Training.Distillation != nil {
				if prog.Inputs[inputIdx].Name != "teacher_probs" {
					t.Errorf("input[%d].Name = %q, want \"teacher_probs\"", inputIdx, prog.Inputs[inputIdx].Name)
				}
				inputIdx++
			}
			if hasTTTMLP {
				if prog.Inputs[inputIdx].Name != "ttt_inner_lr_scale" {
					t.Errorf("input[%d].Name = %q, want \"ttt_inner_lr_scale\"", inputIdx, prog.Inputs[inputIdx].Name)
				}
				inputIdx++
			}
			if cfg.CharVocabSize > 0 {
				if prog.Inputs[inputIdx].Name != "char_ids" {
					t.Errorf("input[%d].Name = %q, want \"char_ids\"", inputIdx, prog.Inputs[inputIdx].Name)
				}
				inputIdx++
			}
			if cfg.BigramVocabSize > 0 {
				if prog.Inputs[inputIdx].Name != "bigram_ids" {
					t.Errorf("input[%d].Name = %q, want \"bigram_ids\"", inputIdx, prog.Inputs[inputIdx].Name)
				}
				inputIdx++
			}
			if cfg.TrigramVocabSize > 0 {
				if prog.Inputs[inputIdx].Name != "trigram_ids" {
					t.Errorf("input[%d].Name = %q, want \"trigram_ids\"", inputIdx, prog.Inputs[inputIdx].Name)
				}
			}

			// Check outputs: scalar loss, per-token NLLs, plus hidden-state/logit exports.
			var wantOutputs []string
			for blockIdx, block := range cfg.Blocks {
				if block.Type != "ttt_mlp" {
					continue
				}
				for _, suffix := range []string{
					"ttt_inner_loss_before",
					"ttt_inner_loss_after",
					"ttt_inner_update_norm",
					"ttt_state_drift",
					"ttt_inner_lr_mean",
					"ttt_inner_lr_min",
					"ttt_inner_lr_max",
				} {
					wantOutputs = append(wantOutputs, fmt.Sprintf("block_%d_%s", blockIdx, suffix))
				}
			}
			wantOutputs = append(wantOutputs, "loss", "per_token_nll", "x_hidden", "logits")
			if cfg.Training.Distillation != nil {
				wantOutputs = append(wantOutputs[:len(wantOutputs)-4], "loss", "eval_loss", "per_token_nll", "x_hidden", "logits")
			}
			if len(prog.Outputs) != len(wantOutputs) {
				t.Fatalf("expected %d outputs, got %d", len(wantOutputs), len(prog.Outputs))
			}
			for i, want := range wantOutputs {
				if prog.Outputs[i].Name != want {
					t.Errorf("output[%d].Name = %q, want %q", i, prog.Outputs[i].Name, want)
				}
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
