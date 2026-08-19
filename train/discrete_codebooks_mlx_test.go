//go:build mlx && cgo && (darwin || linux)

package train

import (
	"math"
	"testing"
)

func TestDiscreteCodebooksClassificationTinyTrainingSmoke(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}
	cfg, err := ParseArchConfig([]byte(`{
		"name":"codebook_training_smoke",
		"model_dim":8,
		"seq_len":4,
		"positional_embedding":"none",
		"input_adapter":{"kind":"discrete_codebooks","num_codebooks":2,"codebook_vocab_size":8,"fusion":"attention_mlp","fusion_hidden_dim":8,"norm":"none"},
		"blocks":[{"type":"swiglu"}],
		"training":{
			"objective":"classification",
			"classification":{"num_labels":2,"pooling":"mean","classifier_dropout":0,"bias":false},
			"optimizer":"adamw",
			"batch_tokens":8,
			"steps":30,
			"lr":0.01,
			"grad_clip":1,
			"weight_decay":0,
			"seed":19
		}
	}`), "codebook_training_smoke")
	if err != nil {
		t.Fatal(err)
	}
	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	trainerInterface, err := initGPUTrainer(prog, cfg, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	trainer := trainerInterface.(*mlxGPUTrainer)
	defer trainer.CloseTrainer()
	batch := objectiveBatch{
		codebooks: []int32{
			0, 0, 0, 1, 1, 0, 1, 1,
			6, 7, 7, 7, 6, 6, 7, 6,
		},
		classificationLabels: []int32{0, 1},
		classificationMask:   []float32{1, 1, 1, 1, 1, 1, 1, 1},
	}
	first, err := trainer.EvaluateObjectiveGPU(batch, 2, 4)
	if err != nil {
		t.Fatal(err)
	}
	for step := 0; step < 30; step++ {
		loss, err := trainer.TrainObjectiveStepGPU(batch, 2, 4, float32(cfg.Training.LR))
		if err != nil {
			t.Fatalf("step %d: %v", step, err)
		}
		if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
			t.Fatalf("step %d non-finite loss=%g", step, loss)
		}
	}
	last, err := trainer.EvaluateObjectiveGPU(batch, 2, 4)
	if err != nil {
		t.Fatal(err)
	}
	if !(last < first) {
		t.Fatalf("codebook classification loss did not decrease: first=%g last=%g", first, last)
	}
}
