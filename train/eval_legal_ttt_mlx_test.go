//go:build mlx && cgo && (darwin || linux)

package train

import (
	"bytes"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestLegalChunkSGDEvalModeMLXSmoke(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}

	cfg, err := ParseArchConfig([]byte(`{
		"name": "legal_ttt_mlx_smoke",
		"model_dim": 16,
		"vocab_size": 16,
		"seq_len": 4,
		"blocks": [{"type": "plain", "heads": 2}],
		"training": {"steps": 1, "lr": 0.001, "seed": 7, "batch_tokens": 8, "grad_clip": 0, "weight_decay": 0},
		"eval": {"ttt_mode": "legal_chunk_sgd", "chunk_tokens": 8, "ttt_epochs": 1, "ttt_lr": 0.001, "ttt_momentum": 0.0}
	}`), "legal-ttt-mlx-smoke")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}

	dir := t.TempDir()
	configPath := filepath.Join(dir, "config.json")
	configBlob, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		t.Fatalf("Marshal config: %v", err)
	}
	if err := os.WriteFile(configPath, configBlob, 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	tokens := []uint16{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0}
	writeLegalTTTShard(t, filepath.Join(dir, "train_00.bin"), tokens)
	writeLegalTTTShard(t, filepath.Join(dir, "val_00.bin"), tokens)
	writeLegalTTTLUTs(t, dir, cfg.VocabSize)

	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatalf("computeWeightShapes: %v", err)
	}
	weights := initWeightData(shapes, cfg.Training.Seed, cfg.Training.WeightInit, cfg.Training.WeightInitStd)
	weightsPath := filepath.Join(dir, "weights.safetensors")
	if err := exportSafetensors(weightsPath, cfg, shapes, weights); err != nil {
		t.Fatalf("exportSafetensors: %v", err)
	}
	before, err := os.ReadFile(weightsPath)
	if err != nil {
		t.Fatalf("read weights before: %v", err)
	}

	if err := runEvalMode(configPath, filepath.Join(dir, "train_*.bin"), weightsPath, dir); err != nil {
		t.Fatalf("runEvalMode legal TTT: %v", err)
	}

	after, err := os.ReadFile(weightsPath)
	if err != nil {
		t.Fatalf("read weights after: %v", err)
	}
	if !bytes.Equal(before, after) {
		t.Fatal("eval-time legal TTT mutated the safetensors file")
	}

	reloaded, err := loadSafetensorsWeights(weightsPath, shapes)
	if err != nil {
		t.Fatalf("loadSafetensorsWeights after eval: %v", err)
	}
	for i := range weights {
		if len(weights[i]) != len(reloaded[i]) {
			t.Fatalf("weight %d len=%d, want %d", i, len(reloaded[i]), len(weights[i]))
		}
		for j := range weights[i] {
			if math.Abs(float64(weights[i][j]-reloaded[i][j])) > 1e-8 {
				t.Fatalf("weight %d[%d] changed: got %g want %g", i, j, reloaded[i][j], weights[i][j])
			}
		}
	}
}
