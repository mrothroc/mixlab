//go:build mlx && cgo && (darwin || linux)

package train

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/gpu"
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

func TestData2VecLossOutputMaskedAndHybridCausalSkip(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}
	cfg, err := ParseArchConfig([]byte(`{
		"name": "data2vec_loss_output",
		"model_dim": 8,
		"vocab_size": 12,
		"seq_len": 3,
		"blocks": [
			{"type": "plain", "heads": 2, "attention_mask": "bidirectional"}
		],
		"training": {
			"steps": 1,
			"lr": 0.001,
			"seed": 31,
			"batch_tokens": 3,
			"optimizer": "adamw",
			"objective": "hybrid",
			"hybrid_clm_fraction": 0.5,
			"hybrid_secondary_objective": "mntp",
			"mlm_mask_token_id": 11,
			"data2vec": {
				"loss_weight": 0.25,
				"top_k_layers": 1,
				"target_norm": "none",
				"smooth_l1_beta": 1.0
			}
		}
	}`), "data2vec_loss_output")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}

	targets := []float32{
		7, -7, 7, -7, 7, -7, 7, -7,
		-5, 5, -5, 5, -5, 5, -5, 5,
		3, 3, -3, -3, 3, -3, 3, -3,
	}
	x := []int{1, 2, 3}
	y := []int{2, 3, 4}

	maskedLoss := runData2VecConcreteLossOutput(t, cfg, arch.ObjectiveMNTP, objectiveBatch{
		x:               x,
		y:               y,
		lossMask:        []float32{1, 0, 1},
		data2vecTargets: targets,
		data2vecMask:    []float32{1, 0, 1},
	})
	if maskedLoss <= 0 || math.IsNaN(float64(maskedLoss)) || math.IsInf(float64(maskedLoss), 0) {
		t.Fatalf("masked data2vec_loss=%g, want finite positive", maskedLoss)
	}

	causalLoss := runData2VecConcreteLossOutput(t, cfg, arch.ObjectiveCausal, objectiveBatch{
		x:               x,
		y:               y,
		data2vecTargets: targets,
		data2vecMask:    []float32{0, 0, 0},
	})
	if causalLoss != 0 {
		t.Fatalf("hybrid causal skip data2vec_loss=%g, want 0", causalLoss)
	}
}

func runData2VecConcreteLossOutput(t *testing.T, cfg *ArchConfig, objective string, batch objectiveBatch) float32 {
	t.Helper()
	prog, err := arch.BuildTrainingIRProgramFromConfig(cfg, arch.TrainingProgramState{
		RecurrenceActive:     true,
		HeadUntied:           cfg.MTPUntieEnabled(),
		DistillationInactive: true,
		Objective:            objective,
	})
	if err != nil {
		t.Fatalf("BuildTrainingIRProgramFromConfig(%s): %v", objective, err)
	}
	gpuProg, err := gpu.LowerIRProgram(prog)
	if err != nil {
		t.Fatalf("LowerIRProgram(%s): %v", objective, err)
	}
	defer gpuProg.Destroy()
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatalf("computeWeightShapes(%s): %v", objective, err)
	}
	weights := initWeightData(shapes, cfg.Training.Seed, cfg.Training.WeightInit, cfg.Training.WeightInitStd)
	handles, err := uploadWeightHandles(shapes, weights)
	if err != nil {
		t.Fatalf("uploadWeightHandles(%s): %v", objective, err)
	}
	defer gpu.FreeHandles(handles)
	need := cfg.SeqLen
	tokens := make([]int32, need)
	targetIDs := make([]int32, need)
	for i := 0; i < need; i++ {
		tokens[i] = int32(batch.x[i])
		targetIDs[i] = int32(batch.y[i])
	}
	inputs := []gpu.TensorInput{
		{Name: "tokens", DType: gpu.TensorInt32, Shape: []int{1, cfg.SeqLen}, Data: tokens},
		{Name: "targets", DType: gpu.TensorInt32, Shape: []int{need}, Data: targetIDs},
		{Name: "data2vec_targets", DType: gpu.TensorFloat32, Shape: []int{need, cfg.ModelDim}, Data: batch.data2vecTargets},
		{Name: "data2vec_loss_mask", DType: gpu.TensorFloat32, Shape: []int{need}, Data: batch.data2vecMask},
	}
	if data2VecProgramDeclaresInput(prog, "loss_mask") {
		inputs = append(inputs, gpu.TensorInput{Name: "loss_mask", DType: gpu.TensorFloat32, Shape: []int{need}, Data: batch.lossMask})
	}
	out, err := gpu.EvalProgramOutput(gpuProg, handles, inputs, "data2vec_loss")
	if err != nil {
		t.Fatalf("EvalProgramOutput(data2vec_loss, %s): %v", objective, err)
	}
	if len(out) != 1 {
		t.Fatalf("data2vec_loss elems=%d want 1", len(out))
	}
	return out[0]
}
