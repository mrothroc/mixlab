package arch

import (
	"encoding/json"
	"path/filepath"
	"testing"
)

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
	if got.Training.LAMBBeta1 != d.LAMBBeta1 {
		t.Errorf("default lamb_beta1 = %g, want %g", got.Training.LAMBBeta1, d.LAMBBeta1)
	}
	if got.Training.LAMBBeta2 != d.LAMBBeta2 {
		t.Errorf("default lamb_beta2 = %g, want %g", got.Training.LAMBBeta2, d.LAMBBeta2)
	}
	if got.Training.LAMBEps != d.LAMBEps {
		t.Errorf("default lamb_eps = %g, want %g", got.Training.LAMBEps, d.LAMBEps)
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
	if got.Training.WarmupSteps != 0 {
		t.Errorf("default warmup_steps = %d, want 0", got.Training.WarmupSteps)
	}
	if got.Training.WarmupRatio != 0 {
		t.Errorf("default warmup_ratio = %g, want 0", got.Training.WarmupRatio)
	}
	if got.Training.HoldSteps != 0 {
		t.Errorf("default hold_steps = %d, want 0", got.Training.HoldSteps)
	}
	if got.Training.TTTSteps != 0 {
		t.Errorf("default ttt_steps = %d, want 0", got.Training.TTTSteps)
	}
	if got.Training.TTTMode != d.TTTMode {
		t.Errorf("default ttt_mode = %q, want %q", got.Training.TTTMode, d.TTTMode)
	}
	if got.Training.TTTLR != d.TTTLR {
		t.Errorf("default ttt_lr = %g, want %g", got.Training.TTTLR, d.TTTLR)
	}
	if got.Training.TTTRank != d.TTTRank {
		t.Errorf("default ttt_rank = %d, want %d", got.Training.TTTRank, d.TTTRank)
	}
	if got.MLPMult != 2.67 {
		t.Errorf("default mlp_mult = %g, want 2.67", got.MLPMult)
	}
	if got.SeqLen != 128 {
		t.Errorf("default seq_len = %d, want 128", got.SeqLen)
	}
}

func TestTrainingWeightDecayExplicitZeroDefaults(t *testing.T) {
	tests := []struct {
		name        string
		training    string
		weightDecay float32
		embedDecay  float32
		matrixDecay float32
		scalarDecay float32
		headDecay   float32
		roundTrip   bool
	}{
		{
			name:        "omitted_global_defaults",
			training:    `"training": {"steps": 10, "lr": 1e-3, "batch_tokens": 128}`,
			weightDecay: 0.01, embedDecay: 0.01, matrixDecay: 0.01, scalarDecay: 0.01, headDecay: 0.01,
		},
		{
			name:        "global_zero_cascades",
			training:    `"training": {"steps": 10, "lr": 1e-3, "batch_tokens": 128, "weight_decay": 0.0}`,
			weightDecay: 0, embedDecay: 0, matrixDecay: 0, scalarDecay: 0, headDecay: 0,
			roundTrip: true,
		},
		{
			name:        "matrix_zero_override",
			training:    `"training": {"steps": 10, "lr": 1e-3, "batch_tokens": 128, "weight_decay": 0.2, "matrix_weight_decay": 0.0}`,
			weightDecay: 0.2, embedDecay: 0.2, matrixDecay: 0, scalarDecay: 0.2, headDecay: 0.2,
			roundTrip: true,
		},
		{
			name: "all_group_zero_overrides",
			training: `"training": {
				"steps": 10, "lr": 1e-3, "batch_tokens": 128, "weight_decay": 0.2,
				"embed_weight_decay": 0.0,
				"matrix_weight_decay": 0.0,
				"scalar_weight_decay": 0.0,
				"head_weight_decay": 0.0
			}`,
			weightDecay: 0.2, embedDecay: 0, matrixDecay: 0, scalarDecay: 0, headDecay: 0,
			roundTrip: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := parseWeightDecayTestConfig(t, tt.training)
			assertWeightDecays(t, cfg, tt.weightDecay, tt.embedDecay, tt.matrixDecay, tt.scalarDecay, tt.headDecay)
			if tt.roundTrip {
				data, err := json.Marshal(cfg)
				if err != nil {
					t.Fatalf("marshal config: %v", err)
				}
				roundTrip, err := ParseArchConfig(data, tt.name+"_roundtrip")
				if err != nil {
					t.Fatalf("round-trip ParseArchConfig: %v", err)
				}
				assertWeightDecays(t, roundTrip, tt.weightDecay, tt.embedDecay, tt.matrixDecay, tt.scalarDecay, tt.headDecay)
			}
		})
	}
}

func TestGPT2StrictSmallWeightDecayZero(t *testing.T) {
	cfg, err := LoadArchConfig(filepath.Join("..", "examples", "gpt2_strict_small_2026.json"))
	if err != nil {
		t.Fatalf("LoadArchConfig: %v", err)
	}
	if cfg.Training.WeightInit != "gpt2" {
		t.Fatalf("weight_init=%q, want gpt2", cfg.Training.WeightInit)
	}
	assertWeightDecays(t, cfg, 0, 0, 0, 0, 0)
}
