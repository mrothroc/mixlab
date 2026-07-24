//go:build mlx && cgo && (darwin || linux)

package train

import (
	"fmt"
	"math"
	"testing"

	"github.com/mrothroc/mixlab/arch"
)

func TestNativeClassificationTinyTrainingPlainAndRecurrent(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}
	for _, block := range []string{
		`{"type":"plain","heads":2}`,
		`{"type":"gated_deltanet","heads":2,"d_k":4}`,
	} {
		t.Run(block, func(t *testing.T) {
			raw := fmt.Sprintf(`{
				"name":"classification_smoke",
				"model_dim":8,
				"vocab_size":8,
				"seq_len":4,
				"tie_embeddings":true,
				"blocks":[%s],
				"training":{
					"objective":"classification",
					"classification":{"num_labels":2,"pooling":"last"},
					"optimizer":"adamw",
					"steps":30,
					"lr":0.003,
					"grad_clip":1.0,
					"weight_decay":0.0,
					"seed":11,
					"batch_tokens":8
				}
			}`, block)
			cfg, err := ParseArchConfig([]byte(raw), "classification_smoke")
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
			rawBatch := trainBatch{
				x:          []int{1, 3, 3, 2, 1, 6, 6, 2},
				y:          make([]int, 8),
				labels:     []int32{0, 1},
				validMask:  []float32{1, 1, 1, 1, 1, 1, 1, 1},
				segmentIDs: make([]int32, 8),
			}
			batch, err := prepareObjectiveBatch(cfg, rawBatch, 0, arch.ObjectiveClassification)
			if err != nil {
				t.Fatal(err)
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
				t.Fatalf("classification loss did not decrease: first=%g last=%g", first, last)
			}
		})
	}
}
