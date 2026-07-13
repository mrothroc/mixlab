//go:build mlx && cgo && (darwin || linux)

package train

import (
	"math"
	"testing"

	"github.com/mrothroc/mixlab/arch"
)

func TestTTTMLPTinyTrainingSmokeUsesOneCompiledProgram(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}
	cfg, err := ParseArchConfig([]byte(`{
		"name":"ttt_mlp_train_smoke",
		"model_dim":8,"vocab_size":16,"seq_len":5,
		"blocks":[
			{"type":"ttt_mlp","heads":2,"chunk_size":3,"inner_lr_warmup_steps":3},
			{"type":"swiglu"}
		],
		"training":{
			"objective":"causal","optimizer":"adamw","steps":3,
			"lr":1e-4,"seed":19,"batch_tokens":5,"grad_clip":1.0,"weight_decay":0.0
		}
	}`), "ttt_mlp_train_smoke")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	prog, err := BuildTrainingIRProgramFromConfig(cfg, arch.TrainingProgramState{Objective: arch.ObjectiveCausal})
	if err != nil {
		t.Fatalf("BuildTrainingIRProgramFromConfig: %v", err)
	}
	trainerIface, err := initGPUTrainer(prog, cfg, nil, nil)
	if err != nil {
		t.Fatalf("initGPUTrainer: %v", err)
	}
	defer trainerIface.CloseTrainer()
	trainer, ok := trainerIface.(*mlxGPUTrainer)
	if !ok {
		t.Fatalf("trainer type=%T, want *mlxGPUTrainer", trainerIface)
	}
	if len(trainer.componentLossOutputs) != 0 {
		t.Fatalf("training-step component outputs=%v, want loss-only TTT step", trainer.componentLossOutputs)
	}
	if len(trainer.tttDiagnosticOutputs) != 7 {
		t.Fatalf("TTT diagnostic outputs=%v, want seven sampled diagnostics", trainer.tttDiagnosticOutputs)
	}
	initialWeights, err := readTrainerWeights(trainer)
	if err != nil {
		t.Fatalf("read initial weights: %v", err)
	}

	raw := trainBatch{x: []int{0, 1, 2, 3, 4}, y: []int{1, 2, 3, 4, 5}}
	probe, err := prepareObjectiveBatch(cfg, raw, 0, arch.ObjectiveCausal)
	if err != nil {
		t.Fatalf("prepare forward probe: %v", err)
	}
	probeLoss, err := trainer.EvaluateObjectiveGPU(probe, 1, cfg.SeqLen)
	if err != nil {
		t.Fatalf("EvaluateObjectiveGPU forward probe: %v", err)
	}
	if !finitePositiveTTT(probeLoss) {
		t.Fatalf("forward probe loss=%g, want finite positive", probeLoss)
	}
	for step := 0; step < 3; step++ {
		batch, err := prepareObjectiveBatch(cfg, raw, step, arch.ObjectiveCausal)
		if err != nil {
			t.Fatalf("prepareObjectiveBatch step %d: %v", step, err)
		}
		if err := trainer.SubmitObjectiveStepGPU(batch, 1, cfg.SeqLen, float32(cfg.Training.LR)); err != nil {
			t.Fatalf("SubmitObjectiveStepGPU step %d: %v", step, err)
		}
		loss, err := trainer.CollectLossGPU()
		if err != nil {
			t.Fatalf("CollectLossGPU step %d: %v", step, err)
		}
		if !finitePositiveTTT(loss) {
			t.Fatalf("step %d loss=%g, want finite positive", step, loss)
		}
		diagnostics, err := trainer.EvaluateTTTDiagnosticsGPU(batch, 1, cfg.SeqLen)
		if err != nil {
			t.Fatalf("EvaluateTTTDiagnosticsGPU step %d: %v", step, err)
		}
		for _, name := range []string{
			"block_0_ttt_inner_loss_before", "block_0_ttt_inner_loss_after",
			"block_0_ttt_inner_update_norm", "block_0_ttt_state_drift",
			"block_0_ttt_inner_lr_mean", "block_0_ttt_inner_lr_min", "block_0_ttt_inner_lr_max",
		} {
			value, found := diagnostics[name]
			if !found || math.IsNaN(value) || math.IsInf(value, 0) {
				t.Fatalf("step %d diagnostic %s=%g found=%v", step, name, value, found)
			}
		}
	}

	stats, err := trainer.CompileStatsGPU()
	if err != nil {
		t.Fatalf("CompileStatsGPU: %v", err)
	}
	if stats.TrainingStepCacheMisses != 1 || stats.TrainingStepCacheHits < 2 {
		t.Fatalf("compile stats=%+v, want one compiled training graph reused across warmup steps", stats)
	}
	if stats.NamedEvalCacheMisses != 1 || stats.NamedEvalCacheHits < 2 {
		t.Fatalf("compile stats=%+v, want one compiled TTT diagnostic graph reused", stats)
	}
	optimizerStats, err := trainer.OptimizerStatsGPU()
	if err != nil {
		t.Fatalf("OptimizerStatsGPU: %v", err)
	}
	if optimizerStats.CommittedSteps != 3 || optimizerStats.SkippedSteps != 0 {
		t.Fatalf("optimizer stats=%+v, want three committed finite updates", optimizerStats)
	}
	weights, err := readTrainerWeights(trainer)
	if err != nil {
		t.Fatalf("readTrainerWeights: %v", err)
	}
	for wi, weight := range weights {
		maxUpdate := float32(0)
		for j, value := range weight {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				t.Fatalf("weight %d[%d]=%g is non-finite", wi, j, value)
			}
			if update := float32(math.Abs(float64(value - initialWeights[wi][j]))); update > maxUpdate {
				maxUpdate = update
			}
		}
		if maxUpdate == 0 {
			t.Fatalf("weight %d received no outer update across three training steps", wi)
		}
	}
}

func finitePositiveTTT(value float32) bool {
	return value > 0 && !math.IsNaN(float64(value)) && !math.IsInf(float64(value), 0)
}
