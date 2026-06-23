//go:build mlx && cgo && (darwin || linux)

package train

import (
	"math"
	"testing"

	"github.com/mrothroc/mixlab/arch"
)

func TestSegmentAttentionMaskMLXTrainingSmoke(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}

	tests := []struct {
		name      string
		objective string
		extra     string
	}{
		{
			name:      "causal",
			objective: arch.ObjectiveCausal,
		},
		{
			name:      "mlm",
			objective: arch.ObjectiveMLM,
			extra: `,
			"objective": "mlm",
			"mlm_mask_prob": 1.0,
			"mlm_mask_token_id": 31,
			"mlm_mask_token_prob": 1.0,
			"mlm_random_token_prob": 0.0,
			"mlm_kept_unchanged_prob": 0.0`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg, err := ParseArchConfig([]byte(`{
				"name": "segment_attention_`+tt.name+`_smoke",
				"model_dim": 16,
				"vocab_size": 32,
				"seq_len": 4,
				"blocks": [
					{"type": "plain", "heads": 2},
					{"type": "swiglu"}
				],
				"training": {
					"steps": 2,
					"lr": 0.001,
					"seed": 23,
					"batch_tokens": 8,
					"grad_clip": 1.0,
					"weight_decay": 0.0,
					"attention_segment_mask": "boundary_token",
					"attention_segment_boundary_token_id": 1`+tt.extra+`
				}
			}`), "segment_attention_"+tt.name+"_smoke")
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
			if !trainer.segmentIDsInput {
				t.Fatal("trainer should declare segment_ids input")
			}

			raw := trainBatch{
				x: []int{2, 3, 1, 4, 5, 1, 6, 7},
				y: []int{3, 1, 4, 8, 1, 6, 7, 9},
			}
			batchSize := cfg.Training.BatchTokens / cfg.SeqLen
			for step := 0; step < cfg.Training.Steps; step++ {
				prepared, err := prepareObjectiveBatch(cfg, raw, step, tt.objective)
				if err != nil {
					t.Fatalf("prepareObjectiveBatch step %d: %v", step, err)
				}
				loss, err := trainer.TrainObjectiveStepGPU(prepared, batchSize, cfg.SeqLen, float32(cfg.Training.LR))
				if err != nil {
					t.Fatalf("TrainObjectiveStepGPU step %d: %v", step, err)
				}
				if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) || loss <= 0 {
					t.Fatalf("loss step %d=%g, want finite positive", step, loss)
				}
			}
		})
	}
}
