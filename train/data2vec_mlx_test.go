//go:build mlx && cgo && (darwin || linux)

package train

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestData2VecHybridTrainingSmoke(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}
	cfg, err := ParseArchConfig([]byte(`{
		"name": "data2vec_hybrid_smoke",
		"model_dim": 8,
		"vocab_size": 16,
		"seq_len": 4,
		"blocks": [
			{"type": "plain", "heads": 2, "attention_mask": "bidirectional"},
			{"type": "swiglu"}
		],
		"training": {
			"steps": 4,
			"lr": 0.01,
			"seed": 23,
			"batch_tokens": 8,
			"optimizer": "adamw",
			"weight_decay": 0.0,
			"grad_clip": 1.0,
			"objective": "hybrid",
			"hybrid_clm_fraction": 0.5,
			"hybrid_secondary_objective": "mntp",
			"mlm_mask_token_id": 15,
			"data2vec": {
				"loss_weight": 0.1,
				"ema_tau": 0.95,
				"top_k_layers": 1,
				"smooth_l1_beta": 1.0
			}
		}
	}`), "data2vec_hybrid_smoke")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	dir := t.TempDir()
	trainDir := filepath.Join(dir, "data")
	if err := os.MkdirAll(trainDir, 0o755); err != nil {
		t.Fatalf("MkdirAll(%s): %v", trainDir, err)
	}
	writeInferenceShard(t, filepath.Join(trainDir, "train_000.bin"), []uint16{
		1, 2, 3, 4, 5, 6, 7, 8,
		2, 3, 4, 5, 6, 7, 8, 9,
		3, 4, 5, 6, 7, 8, 9, 10,
		4, 5, 6, 7, 8, 9, 10, 11,
		5, 6, 7, 8, 9, 10, 11, 12,
	})

	result, err := runTrain(cfg, filepath.Join(trainDir, "train_*.bin"), TrainOptions{LogEvery: 0, ValEvery: 0})
	if err != nil {
		t.Fatalf("runTrain: %v", err)
	}
	for name, v := range map[string]float64{
		"first": result.FirstLoss,
		"last":  result.LastLoss,
	} {
		if v <= 0 || math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("%s loss=%g, want finite positive", name, v)
		}
	}
}
