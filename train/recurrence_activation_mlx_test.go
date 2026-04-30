//go:build mlx && cgo && (darwin || linux)

package train

import (
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	"github.com/mrothroc/mixlab/arch"
)

func TestRecurrenceActivationSwitchPreservesTrainerState(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}

	cfg, err := ParseArchConfig([]byte(`{
		"name": "recurrence_activation_switch",
		"model_dim": 16,
		"vocab_size": 64,
		"seq_len": 4,
		"blocks": [
			{"type": "plain", "heads": 2},
			{"type": "swiglu"},
			{"type": "plain", "heads": 2},
			{"type": "swiglu"}
		],
		"recurrence": [0,1,0,1],
		"training": {
			"steps": 4,
			"lr": 1e-3,
			"seed": 11,
			"batch_tokens": 8,
			"grad_clip": 1.0,
			"weight_decay": 0.0,
			"recurrence_activation_step": 2
		}
	}`), "recurrence_activation_switch")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}

	preProg, err := arch.BuildPreActivationIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildPreActivationIRProgramFromConfig: %v", err)
	}
	postProg, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig: %v", err)
	}

	trainerIface, err := initGPUTrainer(preProg, cfg, nil, nil)
	if err != nil {
		t.Fatalf("initGPUTrainer: %v", err)
	}
	trainer, ok := trainerIface.(*mlxGPUTrainer)
	if !ok {
		t.Fatalf("trainer type=%T, want *mlxGPUTrainer", trainerIface)
	}
	defer trainer.CloseTrainer()

	batchSize := cfg.Training.BatchTokens / cfg.SeqLen
	rng := rand.New(rand.NewSource(cfg.Training.Seed))
	for step := 0; step < cfg.Training.TotalSteps(); step++ {
		if step == cfg.Training.EffectiveRecurrenceActivationStep() {
			if err := trainer.SetProgramGPU(postProg); err != nil {
				t.Fatalf("SetProgramGPU step %d: %v", step, err)
			}
		}
		x, y := generateSyntheticBatch(rng, cfg.Training.BatchTokens, cfg.VocabSize)
		loss, err := trainer.TrainStepGPU(x, y, batchSize, cfg.SeqLen, float32(cfg.Training.LR))
		if err != nil {
			t.Fatalf("TrainStepGPU step %d: %v", step, err)
		}
		if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) || loss <= 0 {
			t.Fatalf("loss step %d=%g, want finite positive", step, loss)
		}
	}
}

func TestRunTrainRecurrenceActivationScheduleSmoke(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}

	cfg, err := ParseArchConfig([]byte(`{
		"name": "recurrence_activation_run_train",
		"model_dim": 16,
		"vocab_size": 64,
		"seq_len": 4,
		"blocks": [
			{"type": "plain", "heads": 2},
			{"type": "swiglu"},
			{"type": "plain", "heads": 2},
			{"type": "swiglu"}
		],
		"recurrence": [0,1,0,1],
		"training": {
			"steps": 4,
			"lr": 1e-3,
			"seed": 13,
			"batch_tokens": 8,
			"grad_clip": 1.0,
			"weight_decay": 0.0,
			"recurrence_activation_frac": 0.5
		}
	}`), "recurrence_activation_run_train")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}

	trainDir := filepath.Join(t.TempDir(), "data")
	if err := os.MkdirAll(trainDir, 0o755); err != nil {
		t.Fatalf("MkdirAll: %v", err)
	}
	writeInferenceShard(t, filepath.Join(trainDir, "train_000.bin"), []uint16{
		1, 2, 3, 4, 5, 6, 7, 8,
		9, 10, 11, 12, 13, 14, 15, 16,
		17, 18, 19, 20, 21, 22, 23, 24,
		25, 26, 27, 28, 29, 30, 31, 32,
		33, 34, 35, 36, 37, 38, 39, 40,
	})

	result, err := runTrain(cfg, filepath.Join(trainDir, "train_*.bin"), TrainOptions{})
	if err != nil {
		t.Fatalf("runTrain: %v", err)
	}
	if math.IsNaN(result.FirstLoss) || math.IsInf(result.FirstLoss, 0) {
		t.Fatalf("first loss is not finite: %g", result.FirstLoss)
	}
	if math.IsNaN(result.LastLoss) || math.IsInf(result.LastLoss, 0) {
		t.Fatalf("last loss is not finite: %g", result.LastLoss)
	}
}
