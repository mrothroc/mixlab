package arch

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestValidPlainConfig(t *testing.T) {
	cfg := ArchConfig{
		Name:      "test_plain",
		ModelDim:  128,
		VocabSize: 1024,
		SeqLen:    128,
		Blocks: []BlockSpec{
			{Type: "plain", Heads: 4},
			{Type: "swiglu"},
			{Type: "plain", Heads: 4},
			{Type: "swiglu"},
		},
		Training: TrainingSpec{Steps: 100, LR: 3e-4},
	}
	data, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	got, err := ParseArchConfig(data, "test_plain")
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if got.Name != "test_plain" {
		t.Errorf("name = %q, want test_plain", got.Name)
	}
	if got.ModelDim != 128 {
		t.Errorf("model_dim = %d, want 128", got.ModelDim)
	}
	if got.VocabSize != 1024 {
		t.Errorf("vocab_size = %d, want 1024", got.VocabSize)
	}
	if got.SeqLen != 128 {
		t.Errorf("seq_len = %d, want 128", got.SeqLen)
	}
	if len(got.Blocks) != 4 {
		t.Errorf("blocks len = %d, want 4", len(got.Blocks))
	}
	if got.Training.Steps != 100 {
		t.Errorf("steps = %d, want 100", got.Training.Steps)
	}
}

func TestMissingModelDim(t *testing.T) {
	cfg := ArchConfig{
		VocabSize: 1024,
		SeqLen:    128,
		Blocks:    []BlockSpec{{Type: "plain", Heads: 4}},
	}
	data, _ := json.Marshal(cfg)
	_, err := ParseArchConfig(data, "test")
	if err == nil {
		t.Fatal("expected error for missing model_dim")
	}
	if !strings.Contains(err.Error(), "model_dim") {
		t.Errorf("error should mention model_dim: %v", err)
	}
}

func TestMissingVocabSize(t *testing.T) {
	cfg := ArchConfig{
		ModelDim: 128,
		SeqLen:   128,
		Blocks:   []BlockSpec{{Type: "plain", Heads: 4}},
	}
	data, _ := json.Marshal(cfg)
	_, err := ParseArchConfig(data, "test")
	if err == nil {
		t.Fatal("expected error for missing vocab_size")
	}
	if !strings.Contains(err.Error(), "vocab_size") {
		t.Errorf("error should mention vocab_size: %v", err)
	}
}

func TestMissingBlocks(t *testing.T) {
	cfg := ArchConfig{
		ModelDim:  128,
		VocabSize: 1024,
		SeqLen:    128,
	}
	data, _ := json.Marshal(cfg)
	_, err := ParseArchConfig(data, "test")
	if err == nil {
		t.Fatal("expected error for missing blocks")
	}
	if !strings.Contains(err.Error(), "at least one block") {
		t.Errorf("error should mention blocks: %v", err)
	}
}

func TestInvalidBlockType(t *testing.T) {
	cfg := ArchConfig{
		ModelDim:  128,
		VocabSize: 1024,
		SeqLen:    128,
		Blocks:    []BlockSpec{{Type: "unknown_type", Heads: 4}},
	}
	data, _ := json.Marshal(cfg)
	_, err := ParseArchConfig(data, "test")
	if err == nil {
		t.Fatal("expected error for invalid block type")
	}
	if !strings.Contains(err.Error(), "invalid type") {
		t.Errorf("error should mention invalid type: %v", err)
	}
}

func TestPlainBlockMissingHeads(t *testing.T) {
	cfg := ArchConfig{
		ModelDim:  128,
		VocabSize: 1024,
		SeqLen:    128,
		Blocks:    []BlockSpec{{Type: "plain"}},
	}
	data, _ := json.Marshal(cfg)
	_, err := ParseArchConfig(data, "test")
	if err == nil {
		t.Fatal("expected error for plain block with no heads")
	}
	if !strings.Contains(err.Error(), "heads") {
		t.Errorf("error should mention heads: %v", err)
	}
}

func TestTrainingDefaults(t *testing.T) {
	cfg := ArchConfig{
		ModelDim:  128,
		VocabSize: 1024,
		Blocks:    []BlockSpec{{Type: "plain", Heads: 4}},
	}
	data, _ := json.Marshal(cfg)
	got, err := ParseArchConfig(data, "test")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	d := DefaultTrainingSpec()
	if got.Training.Steps != d.Steps {
		t.Errorf("default steps = %d, want %d", got.Training.Steps, d.Steps)
	}
	if got.Training.LR != d.LR {
		t.Errorf("default lr = %g, want %g", got.Training.LR, d.LR)
	}
	if got.Training.Seed != d.Seed {
		t.Errorf("default seed = %d, want %d", got.Training.Seed, d.Seed)
	}
	if got.Training.BatchTokens != d.BatchTokens {
		t.Errorf("default batch_tokens = %d, want %d", got.Training.BatchTokens, d.BatchTokens)
	}
	if got.Training.WeightDecay != d.WeightDecay {
		t.Errorf("default weight_decay = %g, want %g", got.Training.WeightDecay, d.WeightDecay)
	}
	if got.Training.EmbedLR != float32(d.LR) {
		t.Errorf("default embed_lr = %g, want %g", got.Training.EmbedLR, float32(d.LR))
	}
	if got.Training.MatrixLR != float32(d.LR) {
		t.Errorf("default matrix_lr = %g, want %g", got.Training.MatrixLR, float32(d.LR))
	}
	if got.Training.ScalarLR != float32(d.LR) {
		t.Errorf("default scalar_lr = %g, want %g", got.Training.ScalarLR, float32(d.LR))
	}
	if got.Training.HeadLR != float32(d.LR) {
		t.Errorf("default head_lr = %g, want %g", got.Training.HeadLR, float32(d.LR))
	}
	if got.Training.MuonMomentum != d.Beta1 {
		t.Errorf("default muon_momentum = %g, want %g", got.Training.MuonMomentum, d.Beta1)
	}
	if got.Training.MuonBackendSteps != d.MuonBackendSteps {
		t.Errorf("default muon_backend_steps = %d, want %d", got.Training.MuonBackendSteps, d.MuonBackendSteps)
	}
	if got.Training.EmbedWeightDecay != d.WeightDecay {
		t.Errorf("default embed_weight_decay = %g, want %g", got.Training.EmbedWeightDecay, d.WeightDecay)
	}
	if got.Training.MatrixWeightDecay != d.WeightDecay {
		t.Errorf("default matrix_weight_decay = %g, want %g", got.Training.MatrixWeightDecay, d.WeightDecay)
	}
	if got.Training.ScalarWeightDecay != d.WeightDecay {
		t.Errorf("default scalar_weight_decay = %g, want %g", got.Training.ScalarWeightDecay, d.WeightDecay)
	}
	if got.Training.HeadWeightDecay != d.WeightDecay {
		t.Errorf("default head_weight_decay = %g, want %g", got.Training.HeadWeightDecay, d.WeightDecay)
	}
	if got.Training.SWADecay != d.SWADecay {
		t.Errorf("default swa_decay = %g, want %g", got.Training.SWADecay, d.SWADecay)
	}
	if got.Training.SWAInterval != d.SWAInterval {
		t.Errorf("default swa_interval = %d, want %d", got.Training.SWAInterval, d.SWAInterval)
	}
	if got.Training.WarmdownSteps != 0 {
		t.Errorf("default warmdown_steps = %d, want 0", got.Training.WarmdownSteps)
	}
	if got.Training.TTTSteps != 0 {
		t.Errorf("default ttt_steps = %d, want 0", got.Training.TTTSteps)
	}
	if got.Training.TTTLR != d.TTTLR {
		t.Errorf("default ttt_lr = %g, want %g", got.Training.TTTLR, d.TTTLR)
	}
	if got.MLPMult != 2.67 {
		t.Errorf("default mlp_mult = %g, want 2.67", got.MLPMult)
	}
	if got.SeqLen != 128 {
		t.Errorf("default seq_len = %d, want 128", got.SeqLen)
	}
}

func TestOptimizerFieldParsing(t *testing.T) {
	// Default: empty string (Muon for matrix weights)
	cfg1, err := ParseArchConfig([]byte(`{
		"model_dim": 128, "vocab_size": 1024,
		"blocks": [{"type": "plain", "heads": 4}],
		"training": {"steps": 10, "lr": 1e-3, "batch_tokens": 128, "seed": 1}
	}`), "default")
	if err != nil {
		t.Fatalf("parse default: %v", err)
	}
	if cfg1.Training.Optimizer != "" {
		t.Errorf("default optimizer = %q, want empty", cfg1.Training.Optimizer)
	}

	// Explicit adamw
	cfg2, err := ParseArchConfig([]byte(`{
		"model_dim": 128, "vocab_size": 1024,
		"blocks": [{"type": "plain", "heads": 4}],
		"training": {"steps": 10, "lr": 1e-3, "batch_tokens": 128, "seed": 1, "optimizer": "adamw"}
	}`), "adamw")
	if err != nil {
		t.Fatalf("parse adamw: %v", err)
	}
	if cfg2.Training.Optimizer != "adamw" {
		t.Errorf("adamw optimizer = %q, want adamw", cfg2.Training.Optimizer)
	}

	// Explicit muon
	cfg3, err := ParseArchConfig([]byte(`{
		"model_dim": 128, "vocab_size": 1024,
		"blocks": [{"type": "plain", "heads": 4}],
		"training": {"steps": 10, "lr": 1e-3, "batch_tokens": 128, "seed": 1, "optimizer": "muon"}
	}`), "muon")
	if err != nil {
		t.Fatalf("parse muon: %v", err)
	}
	if cfg3.Training.Optimizer != "muon" {
		t.Errorf("muon optimizer = %q, want muon", cfg3.Training.Optimizer)
	}
}

func TestMLPMultValidation(t *testing.T) {
	cfg := ArchConfig{
		ModelDim:  128,
		VocabSize: 1024,
		MLPMult:   -1,
		Blocks:    []BlockSpec{{Type: "plain", Heads: 4}},
	}
	data, _ := json.Marshal(cfg)
	_, err := ParseArchConfig(data, "test")
	if err == nil {
		t.Fatal("expected error for negative mlp_mult")
	}
	if !strings.Contains(err.Error(), "mlp_mult") {
		t.Errorf("error should mention mlp_mult: %v", err)
	}
}

func TestNegativeWeightDecay(t *testing.T) {
	cfg := ArchConfig{
		ModelDim:  128,
		VocabSize: 1024,
		Blocks:    []BlockSpec{{Type: "plain", Heads: 4}},
		Training:  TrainingSpec{WeightDecay: -0.01},
	}
	data, _ := json.Marshal(cfg)
	_, err := ParseArchConfig(data, "test")
	if err == nil {
		t.Fatal("expected error for negative weight_decay")
	}
	if !strings.Contains(err.Error(), "weight_decay") {
		t.Errorf("error should mention weight_decay: %v", err)
	}
}

func TestNegativeGradClip(t *testing.T) {
	cfg := ArchConfig{
		ModelDim:  128,
		VocabSize: 1024,
		Blocks:    []BlockSpec{{Type: "plain", Heads: 4}},
		Training:  TrainingSpec{GradClip: -1.0},
	}
	data, _ := json.Marshal(cfg)
	_, err := ParseArchConfig(data, "test")
	if err == nil {
		t.Fatal("expected error for negative grad_clip")
	}
	if !strings.Contains(err.Error(), "grad_clip") {
		t.Errorf("error should mention grad_clip: %v", err)
	}
}

func TestNegativeSWAStart(t *testing.T) {
	cfg := ArchConfig{
		ModelDim:  128,
		VocabSize: 1024,
		Blocks:    []BlockSpec{{Type: "plain", Heads: 4}},
		Training:  TrainingSpec{SWAStart: -1},
	}
	data, _ := json.Marshal(cfg)
	_, err := ParseArchConfig(data, "test")
	if err == nil {
		t.Fatal("expected error for negative swa_start")
	}
	if !strings.Contains(err.Error(), "swa_start") {
		t.Errorf("error should mention swa_start: %v", err)
	}
}

func TestInvalidSWADecay(t *testing.T) {
	cfg := ArchConfig{
		ModelDim:  128,
		VocabSize: 1024,
		Blocks:    []BlockSpec{{Type: "plain", Heads: 4}},
		Training:  TrainingSpec{SWADecay: 1.0},
	}
	data, _ := json.Marshal(cfg)
	_, err := ParseArchConfig(data, "test")
	if err == nil {
		t.Fatal("expected error for invalid swa_decay")
	}
	if !strings.Contains(err.Error(), "swa_decay") {
		t.Errorf("error should mention swa_decay: %v", err)
	}
}

func TestNegativeWarmdownSteps(t *testing.T) {
	cfg := ArchConfig{
		ModelDim:  128,
		VocabSize: 1024,
		Blocks:    []BlockSpec{{Type: "plain", Heads: 4}},
		Training:  TrainingSpec{WarmdownSteps: -1},
	}
	data, _ := json.Marshal(cfg)
	_, err := ParseArchConfig(data, "test")
	if err == nil {
		t.Fatal("expected error for negative warmdown_steps")
	}
	if !strings.Contains(err.Error(), "warmdown_steps") {
		t.Errorf("error should mention warmdown_steps: %v", err)
	}
}

func TestNegativeTargetValLoss(t *testing.T) {
	cfg := ArchConfig{
		ModelDim:  128,
		VocabSize: 1024,
		Blocks:    []BlockSpec{{Type: "plain", Heads: 4}},
		Training:  TrainingSpec{TargetValLoss: -0.1},
	}
	data, _ := json.Marshal(cfg)
	_, err := ParseArchConfig(data, "test")
	if err == nil {
		t.Fatal("expected error for negative target_val_loss")
	}
	if !strings.Contains(err.Error(), "target_val_loss") {
		t.Errorf("error should mention target_val_loss: %v", err)
	}
}

func TestTTTConfigValidation(t *testing.T) {
	cfg := ArchConfig{
		ModelDim:  128,
		VocabSize: 1024,
		Blocks:    []BlockSpec{{Type: "plain", Heads: 4}},
		Training:  TrainingSpec{TTTSteps: 2, TTTLR: 2e-5},
	}
	data, _ := json.Marshal(cfg)
	got, err := ParseArchConfig(data, "test")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.Training.TTTSteps != 2 {
		t.Errorf("ttt_steps = %d, want 2", got.Training.TTTSteps)
	}
	if got.Training.TTTLR != 2e-5 {
		t.Errorf("ttt_lr = %g, want 2e-5", got.Training.TTTLR)
	}

	cfg.Training.TTTSteps = -1
	data, _ = json.Marshal(cfg)
	_, err = ParseArchConfig(data, "test")
	if err == nil {
		t.Fatal("expected error for negative ttt_steps")
	}
	if !strings.Contains(err.Error(), "ttt_steps") {
		t.Errorf("error should mention ttt_steps: %v", err)
	}

	cfg.Training.TTTSteps = 1
	cfg.Training.TTTLR = -1e-5
	data, _ = json.Marshal(cfg)
	_, err = ParseArchConfig(data, "test")
	if err == nil {
		t.Fatal("expected error for negative ttt_lr")
	}
	if !strings.Contains(err.Error(), "ttt_lr") {
		t.Errorf("error should mention ttt_lr: %v", err)
	}
}

func TestNameDefaultsToSource(t *testing.T) {
	cfg := ArchConfig{
		ModelDim:  128,
		VocabSize: 1024,
		Blocks:    []BlockSpec{{Type: "plain", Heads: 4}},
	}
	data, _ := json.Marshal(cfg)
	got, err := ParseArchConfig(data, "my_source.json")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.Name != "my_source.json" {
		t.Errorf("name = %q, want my_source.json", got.Name)
	}
}

func TestInvalidJSON(t *testing.T) {
	_, err := ParseArchConfig([]byte("{invalid"), "test")
	if err == nil {
		t.Fatal("expected error for invalid JSON")
	}
	if !strings.Contains(err.Error(), "parse config") {
		t.Errorf("error should mention parse: %v", err)
	}
}

func TestLoadArchConfigFileNotFound(t *testing.T) {
	_, err := LoadArchConfig("/nonexistent/path/to/config.json")
	if err == nil {
		t.Fatal("expected error for nonexistent file")
	}
	if !strings.Contains(err.Error(), "read config") {
		t.Errorf("error should mention read: %v", err)
	}
}

func TestLoadArchConfigFromFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.json")
	cfg := ArchConfig{
		Name:      "file_test",
		ModelDim:  64,
		VocabSize: 512,
		SeqLen:    64,
		Blocks:    []BlockSpec{{Type: "plain", Heads: 2}},
	}
	data, _ := json.Marshal(cfg)
	if err := os.WriteFile(path, data, 0644); err != nil {
		t.Fatalf("write: %v", err)
	}
	got, err := LoadArchConfig(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if got.Name != "file_test" {
		t.Errorf("name = %q, want file_test", got.Name)
	}
}

func TestSwigluBlockNoHeadsRequired(t *testing.T) {
	cfg := ArchConfig{
		ModelDim:  128,
		VocabSize: 1024,
		SeqLen:    128,
		Blocks:    []BlockSpec{{Type: "swiglu"}},
	}
	data, _ := json.Marshal(cfg)
	_, err := ParseArchConfig(data, "test")
	if err != nil {
		t.Fatalf("swiglu should not require heads: %v", err)
	}
}

func TestExampleConfigsParse(t *testing.T) {
	examples := []string{
		"examples/plain_3L.json",
	}
	for _, rel := range examples {
		t.Run(rel, func(t *testing.T) {
			path := filepath.Join(".", rel)
			_, err := LoadArchConfig(path)
			if err != nil {
				t.Fatalf("failed to load %s: %v", rel, err)
			}
		})
	}
}

func TestValidPerceiverConfig(t *testing.T) {
	cfg := ArchConfig{
		Name:      "test_perceiver",
		ModelDim:  128,
		VocabSize: 1024,
		SeqLen:    128,
		Blocks: []BlockSpec{
			{Type: "perceiver", Heads: 4, NumLatents: 32},
			{Type: "perceiver", Heads: 4, NumLatents: 32},
		},
	}
	data, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	got, err := ParseArchConfig(data, "test_perceiver")
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(got.Blocks) != 2 {
		t.Errorf("blocks len = %d, want 2", len(got.Blocks))
	}
	if got.Blocks[0].Type != "perceiver" {
		t.Errorf("block type = %q, want perceiver", got.Blocks[0].Type)
	}
	if got.Blocks[0].NumLatents != 32 {
		t.Errorf("num_latents = %d, want 32", got.Blocks[0].NumLatents)
	}
}

func TestValidBottleneckConfig(t *testing.T) {
	cfg := ArchConfig{
		Name:      "test_bottleneck",
		ModelDim:  128,
		VocabSize: 1024,
		SeqLen:    128,
		Blocks: []BlockSpec{
			{Type: "bottleneck", Heads: 4, NumLatents: 4},
			{Type: "swiglu"},
		},
	}
	data, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	got, err := ParseArchConfig(data, "test_bottleneck")
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if got.Blocks[0].Type != "bottleneck" {
		t.Errorf("block type = %q, want bottleneck", got.Blocks[0].Type)
	}
	if got.Blocks[0].NumLatents != 4 {
		t.Errorf("num_latents = %d, want 4", got.Blocks[0].NumLatents)
	}
}

func TestPerceiverBlockMissingHeads(t *testing.T) {
	cfg := ArchConfig{
		ModelDim:  128,
		VocabSize: 1024,
		SeqLen:    128,
		Blocks:    []BlockSpec{{Type: "perceiver", NumLatents: 32}},
	}
	data, _ := json.Marshal(cfg)
	_, err := ParseArchConfig(data, "test")
	if err == nil {
		t.Fatal("expected error for perceiver block with no heads")
	}
	if !strings.Contains(err.Error(), "heads") {
		t.Errorf("error should mention heads: %v", err)
	}
}

func TestBottleneckBlockMissingHeads(t *testing.T) {
	cfg := ArchConfig{
		ModelDim:  128,
		VocabSize: 1024,
		SeqLen:    128,
		Blocks:    []BlockSpec{{Type: "bottleneck", NumLatents: 4}},
	}
	data, _ := json.Marshal(cfg)
	_, err := ParseArchConfig(data, "test")
	if err == nil {
		t.Fatal("expected error for bottleneck block with no heads")
	}
	if !strings.Contains(err.Error(), "heads") {
		t.Errorf("error should mention heads: %v", err)
	}
}

func TestValidRetNetConfig(t *testing.T) {
	cfg := ArchConfig{
		Name:      "test_retnet",
		ModelDim:  128,
		VocabSize: 1024,
		SeqLen:    128,
		Blocks: []BlockSpec{
			{Type: "retnet", Heads: 4, Decay: 0.95},
			{Type: "retnet", Heads: 4, Decay: 0.9},
		},
		Training: TrainingSpec{Steps: 100, LR: 3e-4},
	}
	data, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	got, err := ParseArchConfig(data, "test_retnet")
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if got.Name != "test_retnet" {
		t.Errorf("name = %q, want test_retnet", got.Name)
	}
	if len(got.Blocks) != 2 {
		t.Errorf("blocks len = %d, want 2", len(got.Blocks))
	}
	if got.Blocks[0].Type != "retnet" {
		t.Errorf("block[0].type = %q, want retnet", got.Blocks[0].Type)
	}
	if got.Blocks[0].Decay != 0.95 {
		t.Errorf("block[0].decay = %g, want 0.95", got.Blocks[0].Decay)
	}
}

func TestRetNetBlockMissingHeads(t *testing.T) {
	cfg := ArchConfig{
		ModelDim:  128,
		VocabSize: 1024,
		SeqLen:    128,
		Blocks:    []BlockSpec{{Type: "retnet"}},
	}
	data, _ := json.Marshal(cfg)
	_, err := ParseArchConfig(data, "test")
	if err == nil {
		t.Fatal("expected error for retnet block with no heads")
	}
	if !strings.Contains(err.Error(), "heads") {
		t.Errorf("error should mention heads: %v", err)
	}
}

func TestValidTokenBlendConfig(t *testing.T) {
	cfg := ArchConfig{
		Name:      "test_token_blend",
		ModelDim:  128,
		VocabSize: 1024,
		SeqLen:    128,
		Blocks: []BlockSpec{
			{Type: "token_blend"},
			{Type: "swiglu"},
		},
	}
	data, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	got, err := ParseArchConfig(data, "test_token_blend")
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(got.Blocks) != 2 {
		t.Fatalf("blocks len = %d, want 2", len(got.Blocks))
	}
	if got.Blocks[0].Type != "token_blend" {
		t.Fatalf("block[0].type = %q, want token_blend", got.Blocks[0].Type)
	}
}

func TestUnknownFieldRejected(t *testing.T) {
	cfg := `{"name":"test","model_dim":64,"vocab_size":256,"seq_len":32,"bogus_field":true,"blocks":[{"type":"plain","heads":2},{"type":"swiglu"}]}`
	_, err := ParseArchConfig([]byte(cfg), "test")
	if err == nil {
		t.Fatal("expected error for unknown field 'bogus_field', got nil")
	}
	if !strings.Contains(err.Error(), "bogus_field") {
		t.Fatalf("error should mention 'bogus_field', got: %v", err)
	}
}

func TestJSONCComments(t *testing.T) {
	cfg := `{
		// This is a comment
		"name": "test",
		"model_dim": 64, // inline comment
		"vocab_size": 256,
		"seq_len": 32,
		"blocks": [
			// attention block
			{"type": "plain", "heads": 2},
			{"type": "swiglu"}
		]
	}`
	parsed, err := ParseArchConfig([]byte(cfg), "test")
	if err != nil {
		t.Fatalf("ParseArchConfig with JSONC comments: %v", err)
	}
	if parsed.ModelDim != 64 {
		t.Errorf("model_dim = %d, want 64", parsed.ModelDim)
	}
}

func TestJSONCCommentInString(t *testing.T) {
	cfg := `{
		"name": "has // slashes",
		"model_dim": 64,
		"vocab_size": 256,
		"seq_len": 32,
		"blocks": [{"type": "plain", "heads": 2}, {"type": "swiglu"}]
	}`
	parsed, err := ParseArchConfig([]byte(cfg), "test")
	if err != nil {
		t.Fatalf("ParseArchConfig with // inside string: %v", err)
	}
	if parsed.Name != "has // slashes" {
		t.Errorf("name = %q, want %q", parsed.Name, "has // slashes")
	}
}
