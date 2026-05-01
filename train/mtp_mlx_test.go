//go:build mlx && cgo && (darwin || linux)

package train

import (
	"math"
	"math/rand"
	"testing"

	"github.com/mrothroc/mixlab/data"
)

func TestMTPUntieScheduleMLXSmoke(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}

	cfg, err := ParseArchConfig([]byte(`{
		"name": "mtp_untie_schedule",
		"model_dim": 16,
		"vocab_size": 32,
		"seq_len": 4,
		"tie_embeddings": true,
		"mtp": {
			"n": 3,
			"loss_weights": [1.0, 0.5, 0.25],
			"untie_embed_at_frac": 0.5
		},
		"blocks": [{"type": "plain", "heads": 2}],
		"training": {
			"steps": 4,
			"lr": 0.001,
			"seed": 13,
			"batch_tokens": 8,
			"grad_clip": 1.0,
			"weight_decay": 0.0
		}
	}`), "mtp_untie_schedule")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}

	initialProg, err := BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{RecurrenceActive: true, HeadUntied: false})
	if err != nil {
		t.Fatalf("BuildTrainingIRProgramFromConfig(initial): %v", err)
	}
	finalProg, err := BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{RecurrenceActive: true, HeadUntied: true})
	if err != nil {
		t.Fatalf("BuildTrainingIRProgramFromConfig(final): %v", err)
	}
	trainerIface, err := initGPUTrainer(initialProg, cfg, nil, nil)
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
		if step == cfg.EffectiveMTPUntieStep() {
			if err := trainer.CopyWeightGPU("head", "embed"); err != nil {
				t.Fatalf("CopyWeightGPU step %d: %v", step, err)
			}
			if err := trainer.SetProgramGPU(finalProg); err != nil {
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

func TestMTPValidationLossPrefersEvalLossOutputMLX(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}

	cfg, err := ParseArchConfig([]byte(`{
		"name": "mtp_val_eval_loss",
		"model_dim": 16,
		"vocab_size": 24,
		"seq_len": 4,
		"tie_embeddings": false,
		"mtp": {"n": 3, "loss_weights": [1.0, 1.0, 1.0]},
		"blocks": [{"type": "plain", "heads": 2}],
		"training": {
			"steps": 1,
			"lr": 0.001,
			"seed": 17,
			"batch_tokens": 8,
			"grad_clip": 1.0,
			"weight_decay": 0.0
		}
	}`), "mtp_val_eval_loss")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig: %v", err)
	}

	evalTrainer, err := initGPUTrainer(prog, cfg, nil, nil)
	if err != nil {
		t.Fatalf("initGPUTrainer(eval): %v", err)
	}
	defer evalTrainer.CloseTrainer()
	trainTrainer, err := initGPUTrainer(prog, cfg, nil, nil)
	if err != nil {
		t.Fatalf("initGPUTrainer(train): %v", err)
	}
	defer trainTrainer.CloseTrainer()

	rng := rand.New(rand.NewSource(99))
	x, y := generateSyntheticBatch(rng, cfg.Training.BatchTokens, cfg.VocabSize)
	valSet := &data.ValSet{Batches: []data.ValBatch{{X: x, Y: y}}}
	batchSize := cfg.Training.BatchTokens / cfg.SeqLen

	got, err := meanValidationLoss(valSet, evalTrainer, batchSize, cfg.SeqLen)
	if err != nil {
		t.Fatalf("meanValidationLoss: %v", err)
	}
	evalOut, err := readTrainerOutput(evalTrainer, "eval_loss", []int{1})
	if err != nil {
		t.Fatalf("ReadOutput(eval_loss): %v", err)
	}
	trainingLoss, err := trainTrainer.TrainStepGPU(x, y, batchSize, cfg.SeqLen, 0)
	if err != nil {
		t.Fatalf("TrainStepGPU: %v", err)
	}

	if math.Abs(got-float64(evalOut[0])) > 1e-6 {
		t.Fatalf("val loss = %.8f, eval_loss output = %.8f", got, evalOut[0])
	}
	if math.Abs(got-float64(trainingLoss)) < 1e-5 {
		t.Fatalf("val loss unexpectedly matched MTP training loss: val=%.8f loss=%.8f", got, trainingLoss)
	}
}

func TestValidationLossFallsBackToLossOutputWithoutMTPMLX(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}

	cfg, err := ParseArchConfig([]byte(`{
		"name": "plain_val_loss",
		"model_dim": 16,
		"vocab_size": 24,
		"seq_len": 4,
		"tie_embeddings": false,
		"blocks": [{"type": "plain", "heads": 2}],
		"training": {
			"steps": 1,
			"lr": 0.001,
			"seed": 17,
			"batch_tokens": 8,
			"grad_clip": 1.0,
			"weight_decay": 0.0
		}
	}`), "plain_val_loss")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig: %v", err)
	}

	trainer, err := initGPUTrainer(prog, cfg, nil, nil)
	if err != nil {
		t.Fatalf("initGPUTrainer: %v", err)
	}
	defer trainer.CloseTrainer()

	rng := rand.New(rand.NewSource(99))
	x, y := generateSyntheticBatch(rng, cfg.Training.BatchTokens, cfg.VocabSize)
	valSet := &data.ValSet{Batches: []data.ValBatch{{X: x, Y: y}}}
	batchSize := cfg.Training.BatchTokens / cfg.SeqLen

	got, err := meanValidationLoss(valSet, trainer, batchSize, cfg.SeqLen)
	if err != nil {
		t.Fatalf("meanValidationLoss: %v", err)
	}
	lossOut, err := readTrainerOutput(trainer, "loss", []int{1})
	if err != nil {
		t.Fatalf("ReadOutput(loss): %v", err)
	}
	if math.Abs(got-float64(lossOut[0])) > 1e-6 {
		t.Fatalf("val loss = %.8f, loss output = %.8f", got, lossOut[0])
	}
}
