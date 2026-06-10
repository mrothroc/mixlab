//go:build mlx && cgo && (darwin || linux)

package train

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

// TestComputeDTypeBF16TrainingSmoke exercises the bf16 mixed-precision runtime
// path end to end: forward/backward compute in bfloat16 with fp32 master
// weights and optimizer state. It asserts the run produces finite losses and
// stays within a sane band of an otherwise-identical fp32 run.
func TestComputeDTypeBF16TrainingSmoke(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}
	configFor := func(dtype string) string {
		return `{
			"name": "compute_dtype_smoke",
			"model_dim": 16,
			"vocab_size": 24,
			"seq_len": 4,
			"blocks": [
				{"type": "plain", "heads": 2},
				{"type": "swiglu"},
				{"type": "moe", "num_experts": 2, "top_k": 1, "expert_block": {"type": "geglu"}}
			],
			"training": {
				"steps": 6,
				"lr": 0.01,
				"seed": 31,
				"batch_tokens": 8,
				"optimizer": "adamw",
				"weight_decay": 0.0,
				"grad_clip": 1.0,
				"compute_dtype": "` + dtype + `"
			}
		}`
	}
	shard := []uint16{
		1, 2, 3, 4, 5, 6, 7, 8,
		2, 3, 4, 5, 6, 7, 8, 9,
		3, 4, 5, 6, 7, 8, 9, 10,
		4, 5, 6, 7, 8, 9, 10, 11,
		5, 6, 7, 8, 9, 10, 11, 12,
	}

	run := func(dtype string) TrainResult {
		cfg, err := ParseArchConfig([]byte(configFor(dtype)), "compute_dtype_smoke")
		if err != nil {
			t.Fatalf("ParseArchConfig(%s): %v", dtype, err)
		}
		dir := t.TempDir()
		trainDir := filepath.Join(dir, "data")
		if err := os.MkdirAll(trainDir, 0o755); err != nil {
			t.Fatalf("MkdirAll: %v", err)
		}
		writeInferenceShard(t, filepath.Join(trainDir, "train_000.bin"), shard)
		result, err := runTrain(cfg, filepath.Join(trainDir, "train_*.bin"), TrainOptions{LogEvery: 0, ValEvery: 0})
		if err != nil {
			t.Fatalf("runTrain(%s): %v", dtype, err)
		}
		for name, v := range map[string]float64{"first": result.FirstLoss, "last": result.LastLoss} {
			if v <= 0 || math.IsNaN(v) || math.IsInf(v, 0) {
				t.Fatalf("%s: %s loss=%g, want finite positive", dtype, name, v)
			}
		}
		return result
	}

	fp32 := run("float32")
	bf16 := run("bf16")

	// bf16 only changes the forward/backward compute precision, so the loss
	// trajectory should land in the same regime as fp32, not diverge.
	if rel := math.Abs(bf16.LastLoss-fp32.LastLoss) / fp32.LastLoss; rel > 0.25 {
		t.Fatalf("bf16 last loss %g diverges from fp32 %g (rel=%.3f)", bf16.LastLoss, fp32.LastLoss, rel)
	}
}
