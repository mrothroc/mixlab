//go:build mlx && cgo && (darwin || linux)

package train

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/mrothroc/mixlab/gpu"
)

func TestResumedTrainingMatchesUninterruptedWithDropout(t *testing.T) {
	assertResumedTrainingMatchesUninterrupted(t, `{
  "name": "resume_dropout_tiny",
  "model_dim": 8,
  "vocab_size": 16,
  "seq_len": 4,
  "dropout": 0.1,
  "embedding_dropout": 0.1,
  "blocks": [{"type": "plain", "heads": 2}],
  "training": {
    "objective": "causal",
    "steps": 4,
    "lr": 0.001,
    "seed": 19,
    "batch_tokens": 8,
    "optimizer": "adamw",
    "warmup_steps": 0,
    "hold_steps": 0,
    "weight_decay": 0.0
  }
}`)
}

// TestResumedTrainingMatchesUninterruptedAcrossSeqLenSchedule crosses a
// seq-len schedule boundary exactly at the checkpoint step, so a resumed run
// must rebuild the trainer with the startStep-derived program (seq_len 4), not
// the step-0 program (seq_len 2). A regression that re-initialized scheduled
// state from step 0 would diverge here while the schedule-less test stays green.
func TestResumedTrainingMatchesUninterruptedAcrossSeqLenSchedule(t *testing.T) {
	assertResumedTrainingMatchesUninterrupted(t, `{
  "name": "resume_seqlen_schedule_tiny",
  "model_dim": 8,
  "vocab_size": 16,
  "seq_len": 4,
  "dropout": 0.1,
  "embedding_dropout": 0.1,
  "blocks": [{"type": "plain", "heads": 2}],
  "training": {
    "objective": "causal",
    "steps": 4,
    "lr": 0.001,
    "seed": 19,
    "batch_tokens": 8,
    "optimizer": "adamw",
    "warmup_steps": 0,
    "hold_steps": 0,
    "weight_decay": 0.0,
    "seq_len_schedule": [[0, 2], [2, 4]]
  }
}`)
}

func assertResumedTrainingMatchesUninterrupted(t *testing.T, configJSON string) {
	t.Helper()
	if !gpu.Available() {
		t.Skip("MLX backend not available")
	}
	dir := t.TempDir()
	configPath := filepath.Join(dir, "model.json")
	if err := os.WriteFile(configPath, []byte(configJSON), 0o644); err != nil {
		t.Fatal(err)
	}
	tokens := make([]uint16, 256)
	for i := range tokens {
		tokens[i] = uint16((i*7 + 3) % 16)
	}
	trainPath := filepath.Join(dir, "train_000.bin")
	writeInferenceShard(t, trainPath, tokens)
	pattern := filepath.Join(dir, "train_*.bin")
	checkpointDir := filepath.Join(dir, "checkpoints")

	loadConfig := func() *ArchConfig {
		t.Helper()
		cfg, err := LoadArchConfig(configPath)
		if err != nil {
			t.Fatalf("LoadArchConfig: %v", err)
		}
		return cfg
	}
	fullPath := filepath.Join(dir, "full.safetensors")
	if _, err := runTrain(loadConfig(), pattern, TrainOptions{
		SafetensorsPath: fullPath,
		CheckpointDir:   checkpointDir,
		CheckpointEvery: 2,
		LogEvery:        100,
		ValEvery:        100,
	}); err != nil {
		t.Fatalf("uninterrupted runTrain: %v", err)
	}

	resumePath := filepath.Join(dir, "resumed.safetensors")
	manifestPath := filepath.Join(checkpointDir, resumeManifestFilename(2))
	if _, err := runTrain(loadConfig(), pattern, TrainOptions{
		SafetensorsPath: resumePath,
		Resume:          manifestPath,
		LogEvery:        100,
		ValEvery:        100,
	}); err != nil {
		t.Fatalf("resumed runTrain: %v", err)
	}

	cfg := loadConfig()
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatal(err)
	}
	full, err := loadSafetensorsWeights(fullPath, shapes)
	if err != nil {
		t.Fatal(err)
	}
	resumed, err := loadSafetensorsWeights(resumePath, shapes)
	if err != nil {
		t.Fatal(err)
	}
	for i := range full {
		for j := range full[i] {
			if diff := math.Abs(float64(full[i][j] - resumed[i][j])); diff > 1e-6 {
				t.Fatalf("%s[%d] resumed=%g uninterrupted=%g diff=%g", shapes[i].Name, j, resumed[i][j], full[i][j], diff)
			}
		}
	}
	manifest, err := resolveResumeManifest(manifestPath)
	if err != nil {
		t.Fatal(err)
	}
	if manifest.Optimizer.AttemptedSteps != 2 || manifest.GlobalStep != 2 {
		t.Fatalf("checkpoint counters=%+v global_step=%d", manifest.Optimizer, manifest.GlobalStep)
	}
	t.Logf("matched %d tensors after resume from %s", len(full), fmt.Sprint(manifestPath))
}
