//go:build mlx && cgo && (darwin || linux)

package train

import (
	"math"
	"math/rand"
	"testing"

	"github.com/mrothroc/mixlab/arch"
)

func TestDebertaRelativeAttentionCausalAndMLMSmoke(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}

	t.Run("causal", func(t *testing.T) {
		cfg, err := ParseArchConfig([]byte(`{
			"name": "deberta_relative_causal_smoke",
			"model_dim": 32,
			"vocab_size": 64,
			"seq_len": 8,
			"blocks": [
				{"type": "plain", "heads": 4, "relative_attention": "deberta_p2c_c2p", "relative_attention_window": 4},
				{"type": "swiglu"}
			],
			"training": {"steps": 1, "lr": 0.001, "seed": 17, "batch_tokens": 16, "grad_clip": 1.0, "weight_decay": 0.0}
		}`), "deberta_relative_causal_smoke")
		if err != nil {
			t.Fatalf("ParseArchConfig: %v", err)
		}
		prog, err := BuildIRProgramFromConfig(cfg)
		if err != nil {
			t.Fatalf("BuildIRProgramFromConfig: %v", err)
		}
		trainerIface, err := initGPUTrainer(prog, cfg, nil, nil)
		if err != nil {
			t.Fatalf("initGPUTrainer: %v", err)
		}
		trainer, ok := trainerIface.(*mlxGPUTrainer)
		if !ok {
			t.Fatalf("trainer type=%T, want *mlxGPUTrainer", trainerIface)
		}
		defer trainer.CloseTrainer()

		batchSize := cfg.Training.BatchTokens / cfg.SeqLen
		x, y := generateSyntheticBatch(rand.New(rand.NewSource(cfg.Training.Seed)), cfg.Training.BatchTokens, cfg.VocabSize)
		loss, err := trainer.TrainStepGPU(x, y, batchSize, cfg.SeqLen, float32(cfg.Training.LR))
		if err != nil {
			t.Fatalf("TrainStepGPU: %v", err)
		}
		if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) || loss <= 0 {
			t.Fatalf("loss=%g, want finite positive", loss)
		}
	})

	t.Run("mlm", func(t *testing.T) {
		cfg, err := ParseArchConfig([]byte(`{
			"name": "deberta_relative_mlm_smoke",
			"model_dim": 32,
			"vocab_size": 64,
			"seq_len": 8,
			"blocks": [
				{"type": "plain", "heads": 4, "relative_attention": "deberta_p2c_c2p", "relative_attention_window": 4},
				{"type": "swiglu"}
			],
			"training": {
				"steps": 1,
				"lr": 0.001,
				"seed": 23,
				"batch_tokens": 16,
				"grad_clip": 1.0,
				"weight_decay": 0.0,
				"objective": "mlm",
				"mlm_mask_prob": 0.5,
				"mlm_mask_token_id": 63,
				"mlm_mask_token_prob": 1.0,
				"mlm_random_token_prob": 0.0,
				"mlm_kept_unchanged_prob": 0.0
			}
		}`), "deberta_relative_mlm_smoke")
		if err != nil {
			t.Fatalf("ParseArchConfig: %v", err)
		}
		prog, err := BuildIRProgramFromConfig(cfg)
		if err != nil {
			t.Fatalf("BuildIRProgramFromConfig: %v", err)
		}
		trainerIface, err := initGPUTrainer(prog, cfg, nil, nil)
		if err != nil {
			t.Fatalf("initGPUTrainer: %v", err)
		}
		trainer, ok := trainerIface.(*mlxGPUTrainer)
		if !ok {
			t.Fatalf("trainer type=%T, want *mlxGPUTrainer", trainerIface)
		}
		defer trainer.CloseTrainer()

		batchSize := cfg.Training.BatchTokens / cfg.SeqLen
		x, y := generateSyntheticBatch(rand.New(rand.NewSource(cfg.Training.Seed)), cfg.Training.BatchTokens, cfg.VocabSize)
		prepared, err := prepareObjectiveBatch(cfg, trainBatch{x: x, y: y}, 0, arch.ObjectiveMLM)
		if err != nil {
			t.Fatalf("prepareObjectiveBatch: %v", err)
		}
		loss, err := trainer.TrainObjectiveStepGPU(prepared, batchSize, cfg.SeqLen, float32(cfg.Training.LR))
		if err != nil {
			t.Fatalf("TrainObjectiveStepGPU: %v", err)
		}
		if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) || loss <= 0 {
			t.Fatalf("loss=%g, want finite positive", loss)
		}
	})
}

func TestDebertaRelativeHybridMuonProgramSwitchSmoke(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}

	cfg, err := ParseArchConfig([]byte(`{
		"name": "deberta_relative_hybrid_muon_switch",
		"model_dim": 32,
		"vocab_size": 64,
		"seq_len": 8,
		"blocks": [
			{"type": "plain", "heads": 4, "attention_mask": "bidirectional", "relative_attention": "deberta_p2c_c2p", "relative_attention_window": 4},
			{"type": "swiglu"}
		],
		"training": {
			"steps": 12,
			"lr": 0.001,
			"seed": 41,
			"batch_tokens": 16,
			"grad_clip": 1.0,
			"weight_decay": 0.0,
			"optimizer": "muon",
			"objective": "hybrid",
			"hybrid_clm_fraction": 0.5,
			"hybrid_secondary_objective": "mntp",
			"mlm_mask_prob": 0.5,
			"mlm_mask_token_id": 63,
			"mlm_mask_token_prob": 1.0,
			"mlm_random_token_prob": 0.0,
			"mlm_kept_unchanged_prob": 0.0
		}
	}`), "deberta_relative_hybrid_muon_switch")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}

	programs := make(map[string]*arch.Program)
	programForObjective := func(objective string) (*arch.Program, error) {
		if prog := programs[objective]; prog != nil {
			return prog, nil
		}
		prog, err := BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{
			Objective: objective,
		})
		if err != nil {
			return nil, err
		}
		programs[objective] = prog
		return prog, nil
	}

	currentObjective := objectiveForStep(cfg.Training, 0)
	initialProg, err := programForObjective(currentObjective)
	if err != nil {
		t.Fatalf("BuildTrainingIRProgramFromConfig(%s): %v", currentObjective, err)
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

	rng := rand.New(rand.NewSource(cfg.Training.Seed))
	batchSize := cfg.Training.BatchTokens / cfg.SeqLen
	switches := 0
	seen := map[string]bool{currentObjective: true}
	for step := 0; step < cfg.Training.Steps; step++ {
		nextObjective := objectiveForStep(cfg.Training, step)
		seen[nextObjective] = true
		if nextObjective != currentObjective {
			nextProg, err := programForObjective(nextObjective)
			if err != nil {
				t.Fatalf("BuildTrainingIRProgramFromConfig(%s): %v", nextObjective, err)
			}
			if err := trainer.SetProgramGPU(nextProg); err != nil {
				t.Fatalf("SetProgramGPU step %d %s->%s: %v", step, currentObjective, nextObjective, err)
			}
			currentObjective = nextObjective
			switches++
		}

		x, y := generateSyntheticBatch(rng, cfg.Training.BatchTokens, cfg.VocabSize)
		prepared, err := prepareObjectiveBatch(cfg, trainBatch{x: x, y: y}, step, nextObjective)
		if err != nil {
			t.Fatalf("prepareObjectiveBatch step %d objective %s: %v", step, nextObjective, err)
		}
		loss, err := trainer.TrainObjectiveStepGPU(prepared, batchSize, cfg.SeqLen, float32(cfg.Training.LR))
		if err != nil {
			t.Fatalf("TrainObjectiveStepGPU step %d objective %s: %v", step, nextObjective, err)
		}
		if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) || loss <= 0 {
			t.Fatalf("step %d objective %s loss=%g, want finite positive", step, nextObjective, loss)
		}
	}
	if switches == 0 {
		t.Fatalf("hybrid schedule did not switch objectives")
	}
	if !seen[arch.ObjectiveCausal] || !seen[arch.ObjectiveMNTP] {
		t.Fatalf("hybrid schedule objectives seen=%v, want causal and mntp", seen)
	}
}
