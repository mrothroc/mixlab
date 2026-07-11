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

func TestParseArchConfigDataNoShardShuffle(t *testing.T) {
	raw := []byte(`{
		"name": "test_no_shard_shuffle",
		"model_dim": 128,
		"vocab_size": 1024,
		"seq_len": 128,
		"data": {"no_shard_shuffle": true},
		"blocks": [{"type": "plain", "heads": 4}],
		"training": {"steps": 100, "lr": 0.0003, "batch_tokens": 1024}
	}`)
	got, err := ParseArchConfig(raw, "test_no_shard_shuffle")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	if !got.Data.NoShardShuffle {
		t.Fatal("data.no_shard_shuffle = false, want true")
	}

	defaultRaw := []byte(`{
		"name": "test_default_shard_shuffle",
		"model_dim": 128,
		"vocab_size": 1024,
		"seq_len": 128,
		"blocks": [{"type": "plain", "heads": 4}],
		"training": {"steps": 100, "lr": 0.0003, "batch_tokens": 1024}
	}`)
	defaultCfg, err := ParseArchConfig(defaultRaw, "test_default_shard_shuffle")
	if err != nil {
		t.Fatalf("ParseArchConfig default: %v", err)
	}
	if defaultCfg.Data.NoShardShuffle {
		t.Fatal("default data.no_shard_shuffle = true, want false")
	}
}

func TestParseArchConfigSplitDropoutOverrides(t *testing.T) {
	raw := []byte(`{
		"name": "test_split_dropout",
		"model_dim": 128,
		"vocab_size": 1024,
		"seq_len": 128,
		"dropout": 0.1,
		"hidden_dropout": 0.0,
		"attn_dropout": 0.2,
		"blocks": [{"type": "plain", "heads": 4}],
		"training": {"steps": 100, "lr": 0.0003, "batch_tokens": 1024}
	}`)
	got, err := ParseArchConfig(raw, "test_split_dropout")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	if got.EffectiveHiddenDropout() != 0 {
		t.Fatalf("EffectiveHiddenDropout = %g, want explicit 0", got.EffectiveHiddenDropout())
	}
	if got.EffectiveAttnDropout() != 0.2 {
		t.Fatalf("EffectiveAttnDropout = %g, want 0.2", got.EffectiveAttnDropout())
	}

	legacyRaw := []byte(`{
		"name": "test_legacy_dropout",
		"model_dim": 128,
		"vocab_size": 1024,
		"seq_len": 128,
		"dropout": 0.15,
		"blocks": [{"type": "plain", "heads": 4}],
		"training": {"steps": 100, "lr": 0.0003, "batch_tokens": 1024}
	}`)
	legacy, err := ParseArchConfig(legacyRaw, "test_legacy_dropout")
	if err != nil {
		t.Fatalf("ParseArchConfig legacy: %v", err)
	}
	if legacy.EffectiveHiddenDropout() != 0.15 || legacy.EffectiveAttnDropout() != 0.15 {
		t.Fatalf("legacy effective dropout hidden=%g attn=%g, want 0.15/0.15", legacy.EffectiveHiddenDropout(), legacy.EffectiveAttnDropout())
	}
}

func TestParseArchConfig_QKGain(t *testing.T) {
	cfg := ArchConfig{
		Name:      "test_qk_gain",
		ModelDim:  128,
		VocabSize: 1024,
		SeqLen:    128,
		Blocks:    []BlockSpec{{Type: "plain", Heads: 4, QKGain: 5.25}},
		Training:  TrainingSpec{Steps: 100, LR: 3e-4},
	}
	data, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	got, err := ParseArchConfig(data, "test_qk_gain")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	if got.Blocks[0].QKGain != 5.25 {
		t.Fatalf("qk_gain = %g, want 5.25", got.Blocks[0].QKGain)
	}
}

func TestParseArchConfig_XSA(t *testing.T) {
	cfg := ArchConfig{
		Name:      "test_xsa",
		ModelDim:  128,
		VocabSize: 1024,
		SeqLen:    128,
		Blocks:    []BlockSpec{{Type: "plain", Heads: 4, XSA: true}},
		Training:  TrainingSpec{Steps: 100, LR: 3e-4},
	}
	data, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	got, err := ParseArchConfig(data, "test_xsa")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	if !got.Blocks[0].XSA {
		t.Fatal("xsa = false, want true")
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

func parseWeightDecayTestConfig(t *testing.T, training string) *ArchConfig {
	t.Helper()
	cfg, err := ParseArchConfig([]byte(`{
		"model_dim": 128,
		"vocab_size": 1024,
		"blocks": [{"type": "plain", "heads": 4}],
		`+training+`
	}`), "weight_decay")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	return cfg
}

func assertWeightDecays(t *testing.T, cfg *ArchConfig, weight, embed, matrix, scalar, head float32) {
	t.Helper()
	if cfg.Training.WeightDecay != weight {
		t.Fatalf("weight_decay=%g want %g", cfg.Training.WeightDecay, weight)
	}
	if cfg.Training.EmbedWeightDecay != embed {
		t.Fatalf("embed_weight_decay=%g want %g", cfg.Training.EmbedWeightDecay, embed)
	}
	if cfg.Training.MatrixWeightDecay != matrix {
		t.Fatalf("matrix_weight_decay=%g want %g", cfg.Training.MatrixWeightDecay, matrix)
	}
	if cfg.Training.ScalarWeightDecay != scalar {
		t.Fatalf("scalar_weight_decay=%g want %g", cfg.Training.ScalarWeightDecay, scalar)
	}
	if cfg.Training.HeadWeightDecay != head {
		t.Fatalf("head_weight_decay=%g want %g", cfg.Training.HeadWeightDecay, head)
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

	// Explicit MuonEq-R normalizes to canonical spelling
	cfg4, err := ParseArchConfig([]byte(`{
		"model_dim": 128, "vocab_size": 1024,
		"blocks": [{"type": "plain", "heads": 4}],
		"training": {"steps": 10, "lr": 1e-3, "batch_tokens": 128, "seed": 1, "optimizer": " Muon_Eq_R "}
	}`), "muon_eq_r")
	if err != nil {
		t.Fatalf("parse muon_eq_r: %v", err)
	}
	if cfg4.Training.Optimizer != "muon_eq_r" {
		t.Errorf("muon_eq_r optimizer = %q, want muon_eq_r", cfg4.Training.Optimizer)
	}
	roundTrip, err := json.Marshal(cfg4)
	if err != nil {
		t.Fatalf("marshal muon_eq_r: %v", err)
	}
	if !strings.Contains(string(roundTrip), `"optimizer":"muon_eq_r"`) {
		t.Fatalf("round-trip JSON missing optimizer=muon_eq_r: %s", roundTrip)
	}

	cfg5, err := ParseArchConfig([]byte(`{
		"model_dim": 128, "vocab_size": 1024,
		"blocks": [{"type": "plain", "heads": 4}],
		"training": {"steps": 10, "lr": 1e-3, "batch_tokens": 128, "seed": 1, "optimizer": " NorMuon "}
	}`), "normuon")
	if err != nil {
		t.Fatalf("parse normuon: %v", err)
	}
	if cfg5.Training.Optimizer != "normuon" {
		t.Errorf("normuon optimizer = %q, want normuon", cfg5.Training.Optimizer)
	}

	cfg6, err := ParseArchConfig([]byte(`{
		"model_dim": 128, "vocab_size": 1024,
		"blocks": [{"type": "plain", "heads": 4}],
		"training": {"steps": 10, "lr": 1e-3, "batch_tokens": 128, "seed": 1, "optimizer": " LAMB ", "lamb_beta1": 0.0, "lamb_beta2": 0.999, "lamb_eps": 1e-6}
	}`), "lamb")
	if err != nil {
		t.Fatalf("parse lamb: %v", err)
	}
	if cfg6.Training.Optimizer != "lamb" {
		t.Errorf("lamb optimizer = %q, want lamb", cfg6.Training.Optimizer)
	}
	if cfg6.Training.LAMBBeta1 != 0 || cfg6.Training.LAMBBeta2 != 0.999 || cfg6.Training.LAMBEps != 1e-6 || cfg6.Training.LAMBTrustRatioCap != 10 {
		t.Errorf("lamb hyperparams = beta1=%g beta2=%g eps=%g trust_cap=%g", cfg6.Training.LAMBBeta1, cfg6.Training.LAMBBeta2, cfg6.Training.LAMBEps, cfg6.Training.LAMBTrustRatioCap)
	}
	roundTripLAMB, err := json.Marshal(cfg6)
	if err != nil {
		t.Fatalf("marshal lamb: %v", err)
	}
	if !strings.Contains(string(roundTripLAMB), `"optimizer":"lamb"`) ||
		!strings.Contains(string(roundTripLAMB), `"lamb_beta2":0.999`) ||
		!strings.Contains(string(roundTripLAMB), `"lamb_eps":0.000001`) {
		t.Fatalf("round-trip JSON missing LAMB fields: %s", roundTripLAMB)
	}
	cfg7, err := ParseArchConfig([]byte(`{
		"model_dim": 128, "vocab_size": 1024,
		"blocks": [{"type": "plain", "heads": 4}],
		"training": {"steps": 10, "lr": 1e-3, "batch_tokens": 128, "seed": 1, "optimizer": "lamb", "lamb_trust_ratio_cap": 0.0}
	}`), "lamb_uncapped")
	if err != nil {
		t.Fatalf("parse lamb uncapped: %v", err)
	}
	if cfg7.Training.LAMBTrustRatioCap != 0 {
		t.Errorf("explicit lamb_trust_ratio_cap = %g, want 0", cfg7.Training.LAMBTrustRatioCap)
	}
	roundTripLAMBUncapped, err := json.Marshal(cfg7)
	if err != nil {
		t.Fatalf("marshal lamb uncapped: %v", err)
	}
	if !strings.Contains(string(roundTripLAMBUncapped), `"lamb_trust_ratio_cap":0`) {
		t.Fatalf("round-trip JSON missing explicit uncapped LAMB trust-ratio cap: %s", roundTripLAMBUncapped)
	}

	_, err = ParseArchConfig([]byte(`{
		"model_dim": 128, "vocab_size": 1024,
		"blocks": [{"type": "plain", "heads": 4}],
		"training": {"steps": 10, "lr": 1e-3, "batch_tokens": 128, "seed": 1, "optimizer": "sgd"}
	}`), "bad_optimizer")
	if err == nil {
		t.Fatal("expected invalid optimizer error")
	}
}

func TestLAMBValidation(t *testing.T) {
	tests := []struct {
		name     string
		training string
		want     string
	}{
		{"bad_beta1_negative", `"optimizer": "lamb", "lamb_beta1": -0.1`, "lamb_beta1"},
		{"bad_beta1_one", `"optimizer": "lamb", "lamb_beta1": 1.0`, "lamb_beta1"},
		{"bad_beta2_negative", `"optimizer": "lamb", "lamb_beta2": -0.1`, "lamb_beta2"},
		{"bad_beta2_one", `"optimizer": "lamb", "lamb_beta2": 1.0`, "lamb_beta2"},
		{"bad_eps_zero", `"optimizer": "lamb", "lamb_eps": 0.0`, "lamb_eps"},
		{"bad_trust_ratio_cap_negative", `"optimizer": "lamb", "lamb_trust_ratio_cap": -1.0`, "lamb_trust_ratio_cap"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ParseArchConfig([]byte(`{
				"model_dim": 128,
				"vocab_size": 1024,
				"blocks": [{"type": "plain", "heads": 4}],
				"training": {"steps": 10, "lr": 1e-3, "batch_tokens": 128, "seed": 1, `+tt.training+`}
			}`), tt.name)
			if err == nil {
				t.Fatal("ParseArchConfig succeeded, want error")
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error %q does not mention %q", err, tt.want)
			}
		})
	}
}

func TestTrainingPhasesParsingAndTotalSteps(t *testing.T) {
	cfg, err := ParseArchConfig([]byte(`{
		"model_dim": 128, "vocab_size": 1024,
		"blocks": [{"type": "plain", "heads": 4}],
		"training": {
			"steps": 10,
			"lr": 1e-5,
			"phases": [
				{"steps": 100, "lr": 1e-4, "label": "warmup"},
				{"steps": 4000, "lr": 1e-3, "label": "main"},
				{"steps": 900, "lr": 1e-4, "label": "cooldown"}
			]
		}
	}`), "phases")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	if len(cfg.Training.Phases) != 3 {
		t.Fatalf("phases len = %d, want 3", len(cfg.Training.Phases))
	}
	if got := cfg.Training.TotalSteps(); got != 5000 {
		t.Fatalf("TotalSteps() = %d, want 5000", got)
	}
	if cfg.Training.Steps != 5000 {
		t.Fatalf("training.steps = %d, want computed total 5000", cfg.Training.Steps)
	}
	if cfg.Training.Phases[1].Label != "main" {
		t.Fatalf("phases[1].label = %q, want main", cfg.Training.Phases[1].Label)
	}
}

func TestTrainingPhasesValidation(t *testing.T) {
	tests := []struct {
		name    string
		phases  string
		wantErr string
	}{
		{
			name:    "non_positive_steps",
			phases:  `[{"steps": 0, "lr": 1e-4}]`,
			wantErr: "training.phases[0].steps",
		},
		{
			name:    "non_positive_lr",
			phases:  `[{"steps": 10, "lr": 0}]`,
			wantErr: "training.phases[0].lr",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ParseArchConfig([]byte(`{
				"model_dim": 128, "vocab_size": 1024,
				"blocks": [{"type": "plain", "heads": 4}],
				"training": {"phases": `+tt.phases+`}
			}`), tt.name)
			if err == nil {
				t.Fatal("expected error")
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("error = %q, want substring %q", err.Error(), tt.wantErr)
			}
		})
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

func TestMLPActivationConfigParsing(t *testing.T) {
	cfg := `{
		"name": "test",
		"model_dim": 64,
		"vocab_size": 256,
		"seq_len": 32,
		"blocks": [{"type": "mlp", "activation": "leaky_relu_sq", "leaky_slope": 0.25}]
	}`
	got, err := ParseArchConfig([]byte(cfg), "test")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	if got.Blocks[0].Activation != "leaky_relu_sq" {
		t.Fatalf("activation = %q, want leaky_relu_sq", got.Blocks[0].Activation)
	}
	if got.Blocks[0].LeakySlope != 0.25 {
		t.Fatalf("leaky_slope = %g, want 0.25", got.Blocks[0].LeakySlope)
	}
}

func TestTTTConfigValidation(t *testing.T) {
	cfg := ArchConfig{
		ModelDim:  128,
		VocabSize: 1024,
		Blocks:    []BlockSpec{{Type: "plain", Heads: 4}},
		Training:  TrainingSpec{TTTSteps: 2, TTTMode: "lora", TTTLR: 2e-5, TTTRank: 8},
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
	if got.Training.TTTMode != "lora" {
		t.Errorf("ttt_mode = %q, want lora", got.Training.TTTMode)
	}
	if got.Training.TTTRank != 8 {
		t.Errorf("ttt_rank = %d, want 8", got.Training.TTTRank)
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

	cfg.Training.TTTLR = 2e-5
	cfg.Training.TTTMode = "bad"
	data, _ = json.Marshal(cfg)
	_, err = ParseArchConfig(data, "test")
	if err == nil {
		t.Fatal("expected error for invalid ttt_mode")
	}
	if !strings.Contains(err.Error(), "ttt_mode") {
		t.Errorf("error should mention ttt_mode: %v", err)
	}

	cfg.Training.TTTMode = "lora"
	cfg.Training.TTTRank = -1
	data, _ = json.Marshal(cfg)
	_, err = ParseArchConfig(data, "test")
	if err == nil {
		t.Fatal("expected error for negative ttt_rank")
	}
	if !strings.Contains(err.Error(), "ttt_rank") {
		t.Errorf("error should mention ttt_rank: %v", err)
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

func TestGegluBlockNoHeadsRequired(t *testing.T) {
	cfg := ArchConfig{
		ModelDim:  128,
		VocabSize: 1024,
		SeqLen:    128,
		Blocks:    []BlockSpec{{Type: "geglu"}},
	}
	data, _ := json.Marshal(cfg)
	_, err := ParseArchConfig(data, "test")
	if err != nil {
		t.Fatalf("geglu should not require heads: %v", err)
	}
}

func TestExampleConfigsParse(t *testing.T) {
	examples := []string{
		"examples/plain_3L.json",
		"examples/invariance_mlm_tiny.json",
		"examples/pll_margin_mlm_tiny.json",
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

// QAT start tests are in config_qat_test.go.
