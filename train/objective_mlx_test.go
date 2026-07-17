//go:build mlx && cgo && (darwin || linux)

package train

import (
	"math"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/arch"
)

func TestMLMObjectiveLearnsMaskedTokenAboveChance(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}

	cfg, err := ParseArchConfig([]byte(`{
		"name": "mlm_objective_smoke",
		"model_dim": 16,
		"vocab_size": 16,
		"seq_len": 4,
		"blocks": [
			{"type": "plain", "heads": 2, "attention_mask": "bidirectional"},
			{"type": "swiglu"}
		],
		"training": {
			"steps": 20,
			"lr": 0.05,
			"seed": 19,
			"batch_tokens": 8,
			"grad_clip": 1.0,
			"weight_decay": 0.0,
			"objective": "mlm",
			"mlm_mask_prob": 1.0,
			"mlm_mask_token_id": 15,
			"mlm_mask_token_prob": 1.0,
			"mlm_random_token_prob": 0.0,
			"mlm_kept_unchanged_prob": 0.0
		}
	}`), "mlm_objective_smoke")
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
	raw := trainBatch{
		x: []int{3, 3, 3, 3, 3, 3, 3, 3},
		y: []int{0, 0, 0, 0, 0, 0, 0, 0},
	}
	prepared, err := prepareObjectiveBatch(cfg, raw, 0, arch.ObjectiveMLM)
	if err != nil {
		t.Fatalf("prepareObjectiveBatch: %v", err)
	}
	initial, err := evaluatePreparedObjectiveBatch(trainer, prepared, batchSize, cfg.SeqLen)
	if err != nil {
		t.Fatalf("initial evaluate: %v", err)
	}
	for step := 0; step < cfg.Training.Steps; step++ {
		prepared, err := prepareObjectiveBatch(cfg, raw, step, arch.ObjectiveMLM)
		if err != nil {
			t.Fatalf("prepare step %d: %v", step, err)
		}
		if _, err := trainer.TrainObjectiveStepGPU(prepared, batchSize, cfg.SeqLen, float32(cfg.Training.LR)); err != nil {
			t.Fatalf("TrainObjectiveStepGPU step %d: %v", step, err)
		}
	}
	final, err := evaluatePreparedObjectiveBatch(trainer, prepared, batchSize, cfg.SeqLen)
	if err != nil {
		t.Fatalf("final evaluate: %v", err)
	}
	if math.IsNaN(float64(final)) || math.IsInf(float64(final), 0) {
		t.Fatalf("non-finite final loss: %g", final)
	}
	chance := float32(math.Log(float64(cfg.VocabSize)))
	if final >= initial {
		t.Fatalf("final loss = %g, want below initial %g", final, initial)
	}
	if final >= chance {
		t.Fatalf("final loss = %g, want below chance %g", final, chance)
	}
}

func TestMLMWholeWordCurriculumMLXSmoke(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}
	cfg, err := LoadArchConfig("examples/mlm_wwm_curriculum_tiny.json")
	if err != nil {
		t.Fatalf("LoadArchConfig: %v", err)
	}
	cfg.Training.MLMWordStart = make([]uint8, cfg.VocabSize)
	cfg.Training.MLMMaskEligible = make([]uint8, cfg.VocabSize)
	for id := 5; id < cfg.VocabSize; id++ {
		cfg.Training.MLMMaskEligible[id] = 1
		cfg.Training.MLMWordStart[id] = 1
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
	raw := trainBatch{x: make([]int, cfg.Training.BatchTokens), y: make([]int, cfg.Training.BatchTokens)}
	for i := range raw.x {
		raw.x[i] = 5 + i%8
		raw.y[i] = 5 + (i+1)%8
	}
	for step := 0; step < cfg.Training.Steps; step++ {
		prepared, err := prepareObjectiveBatch(cfg, raw, step, arch.ObjectiveMLM)
		if err != nil {
			t.Fatalf("prepare step %d: %v", step, err)
		}
		wantUnit := arch.MLMMaskUnitWholeWord
		if step >= 2 {
			wantUnit = arch.MLMMaskUnitToken
		}
		if prepared.mlmMaskStats.Unit != wantUnit {
			t.Fatalf("step %d unit=%q, want %q", step, prepared.mlmMaskStats.Unit, wantUnit)
		}
		loss, err := trainer.TrainObjectiveStepGPU(prepared, cfg.Training.BatchTokens/cfg.SeqLen, cfg.SeqLen, float32(cfg.Training.LR))
		if err != nil {
			t.Fatalf("step %d: %v", step, err)
		}
		if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
			t.Fatalf("step %d non-finite loss %g", step, loss)
		}
	}
}

func TestWordStructuralMLXSmoke(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}

	cfg, err := ParseArchConfig([]byte(`{
		"name": "word_structural_mlx_smoke",
		"model_dim": 16,
		"vocab_size": 20,
		"seq_len": 6,
		"blocks": [
			{"type": "plain", "heads": 2, "attention_mask": "bidirectional"},
			{"type": "swiglu"}
		],
		"training": {
			"steps": 1,
			"lr": 0.01,
			"seed": 23,
			"batch_tokens": 6,
			"weight_decay": 0.0,
			"objective": "mlm",
			"mlm_mask_prob": 0.0,
			"mlm_mask_token_id": 19,
			"word_structural_objective": {
				"fraction": 0.5,
				"span": 3,
				"loss_weight": 1.0,
				"skip_token_ids": [19]
			}
		}
	}`), "word_structural_mlx_smoke")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	prog, err := BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{Objective: arch.ObjectiveMLM})
	if err != nil {
		t.Fatalf("BuildTrainingIRProgramFromConfig: %v", err)
	}
	trainerIface, err := initGPUTrainer(prog, cfg, nil, nil)
	if err != nil {
		if strings.Contains(err.Error(), "MLX backend unavailable") {
			t.Skip(err.Error())
		}
		t.Fatalf("initGPUTrainer: %v", err)
	}
	trainer, ok := trainerIface.(*mlxGPUTrainer)
	if !ok {
		t.Fatalf("trainer type=%T, want *mlxGPUTrainer", trainerIface)
	}
	defer trainer.CloseTrainer()

	raw := trainBatch{
		x: []int{1, 2, 3, 4, 5, 6},
		y: []int{2, 3, 4, 5, 6, 7},
	}
	prepared, err := prepareObjectiveBatch(cfg, raw, 0, arch.ObjectiveMLM)
	if err != nil {
		t.Fatalf("prepareObjectiveBatch: %v", err)
	}
	if _, err := trainer.EvaluateObjectiveGPUWithOutputs(prepared, 1, cfg.SeqLen, []string{"word_struct_loss"}); err != nil {
		t.Fatalf("EvaluateObjectiveGPUWithOutputs: %v", err)
	}
	wordLoss, err := trainer.ReadOutput("word_struct_loss", []int{1})
	if err != nil {
		t.Fatalf("ReadOutput(word_struct_loss): %v", err)
	}
	if len(wordLoss) != 1 || wordLoss[0] <= 0 || math.IsNaN(float64(wordLoss[0])) || math.IsInf(float64(wordLoss[0]), 0) {
		t.Fatalf("word_struct_loss=%v, want positive finite scalar", wordLoss)
	}
}

func TestBlockDiffusionMLXSmoke(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}

	cfg, err := ParseArchConfig([]byte(`{
		"name": "block_diffusion_mlx_smoke",
		"model_dim": 16,
		"vocab_size": 16,
		"seq_len": 4,
		"blocks": [
			{"type": "plain", "heads": 2},
			{"type": "swiglu"}
		],
		"training": {
			"steps": 2,
			"lr": 0.01,
			"seed": 31,
			"batch_tokens": 8,
			"grad_clip": 1.0,
			"weight_decay": 0.0,
			"objective": "block_diffusion",
			"mlm_mask_token_id": 15,
			"diffusion": {
				"block_size": 2,
				"min_mask_fraction": 1.0,
				"max_mask_fraction": 1.0
			}
		}
	}`), "block_diffusion_mlx_smoke")
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
	raw := trainBatch{
		x: []int{3, 4, 5, 6, 3, 4, 5, 6},
		y: []int{4, 5, 6, 7, 4, 5, 6, 7},
	}
	for step := 0; step < cfg.Training.Steps; step++ {
		prepared, err := prepareObjectiveBatch(cfg, raw, step, arch.ObjectiveBlockDiffusion)
		if err != nil {
			t.Fatalf("prepare block diffusion step %d: %v", step, err)
		}
		loss, err := trainer.TrainObjectiveStepGPU(prepared, batchSize, cfg.SeqLen, float32(cfg.Training.LR))
		if err != nil {
			t.Fatalf("TrainObjectiveStepGPU step %d: %v", step, err)
		}
		if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) || loss <= 0 {
			t.Fatalf("loss step %d=%g, want finite positive", step, loss)
		}
	}
}

func TestHybridBlockDiffusionMLXProgramSwitchSmoke(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}

	cfg, err := ParseArchConfig([]byte(`{
		"name": "hybrid_block_diffusion_mlx_smoke",
		"model_dim": 16,
		"vocab_size": 16,
		"seq_len": 4,
		"blocks": [
			{"type": "plain", "heads": 2},
			{"type": "swiglu"}
		],
		"training": {
			"steps": 2,
			"lr": 0.01,
			"seed": 37,
			"batch_tokens": 8,
			"grad_clip": 1.0,
			"weight_decay": 0.0,
			"objective": "hybrid",
			"hybrid_secondary_objective": "block_diffusion",
			"hybrid_clm_fraction_schedule": [[0,1],[1,0]],
			"mlm_mask_token_id": 15,
			"diffusion": {
				"block_size": 2,
				"min_mask_fraction": 1.0,
				"max_mask_fraction": 1.0
			}
		}
	}`), "hybrid_block_diffusion_mlx_smoke")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	causalProg, err := BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{
		RecurrenceActive: true,
		Objective:        arch.ObjectiveCausal,
	})
	if err != nil {
		t.Fatalf("BuildTrainingIRProgramFromConfig(causal): %v", err)
	}
	diffusionProg, err := BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{
		RecurrenceActive: true,
		Objective:        arch.ObjectiveBlockDiffusion,
	})
	if err != nil {
		t.Fatalf("BuildTrainingIRProgramFromConfig(block_diffusion): %v", err)
	}
	trainerIface, err := initGPUTrainer(causalProg, cfg, nil, nil)
	if err != nil {
		t.Fatalf("initGPUTrainer: %v", err)
	}
	trainer, ok := trainerIface.(*mlxGPUTrainer)
	if !ok {
		t.Fatalf("trainer type=%T, want *mlxGPUTrainer", trainerIface)
	}
	defer trainer.CloseTrainer()

	batchSize := cfg.Training.BatchTokens / cfg.SeqLen
	raw := trainBatch{
		x: []int{3, 4, 5, 6, 6, 5, 4, 3},
		y: []int{4, 5, 6, 7, 5, 4, 3, 2},
	}
	causal, err := prepareObjectiveBatch(cfg, raw, 0, objectiveForStep(cfg.Training, 0))
	if err != nil {
		t.Fatalf("prepare causal step: %v", err)
	}
	loss, err := trainer.TrainObjectiveStepGPU(causal, batchSize, cfg.SeqLen, float32(cfg.Training.LR))
	if err != nil {
		t.Fatalf("TrainObjectiveStepGPU causal: %v", err)
	}
	if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) || loss <= 0 {
		t.Fatalf("causal loss=%g, want finite positive", loss)
	}

	switcher, ok := trainerIface.(gpuProgramSwitcher)
	if !ok {
		t.Fatalf("trainer type=%T does not implement gpuProgramSwitcher", trainerIface)
	}
	if err := switcher.SetProgramGPU(diffusionProg); err != nil {
		t.Fatalf("SetProgramGPU(diffusion): %v", err)
	}
	diffusion, err := prepareObjectiveBatch(cfg, raw, 1, objectiveForStep(cfg.Training, 1))
	if err != nil {
		t.Fatalf("prepare diffusion step: %v", err)
	}
	loss, err = trainer.TrainObjectiveStepGPU(diffusion, batchSize, cfg.SeqLen, float32(cfg.Training.LR))
	if err != nil {
		t.Fatalf("TrainObjectiveStepGPU diffusion: %v", err)
	}
	if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) || loss <= 0 {
		t.Fatalf("diffusion loss=%g, want finite positive", loss)
	}
}

func evaluatePreparedObjectiveBatch(trainer *mlxGPUTrainer, batch objectiveBatch, batchSize, seqLen int) (float32, error) {
	return trainer.EvaluateObjectiveGPU(batch, batchSize, seqLen)
}
