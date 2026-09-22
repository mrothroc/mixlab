//go:build mlx && cgo && (darwin || linux)

package train

import (
	"fmt"
	"math"
	"reflect"
	"testing"

	"github.com/mrothroc/mixlab/gpu"
)

func affineOptimizerTestConfig(t *testing.T, optimizer, rates string) *ArchConfig {
	t.Helper()
	raw := fmt.Sprintf(`{"model_dim":16,"vocab_size":32,"seq_len":4,"tie_embeddings":false,
		"blocks":[{"type":"plain","heads":2,"attn_bias":true,"ffn_bias":true}],
		"training":{"steps":4,"batch_tokens":8,"weight_init":"pytorch_linear_all",
		"optimizer":%q,%s}}`, optimizer, rates)
	cfg, err := ParseArchConfig([]byte(raw), "affine_optimizer")
	if err != nil {
		t.Fatal(err)
	}
	return cfg
}

func TestOptimizerSpecExplicitZeroLearningRate(t *testing.T) {
	for _, optimizer := range []string{"adamw", "lamb", "muon"} {
		for _, rates := range []string{`"lr":0`, `"lr":0.001,"embed_lr":0,"head_lr":0,"matrix_lr":0,"scalar_lr":0`} {
			cfg := affineOptimizerTestConfig(t, optimizer, rates)
			shapes, err := computeWeightShapes(cfg)
			if err != nil {
				t.Fatal(err)
			}
			spec, err := buildTrainerOptimizerSpec(cfg, shapes)
			if err != nil {
				t.Fatal(err)
			}
			for _, g := range spec.Groups {
				if g.LR != 0 {
					t.Fatalf("%s %s: group LR=%g want zero", optimizer, rates, g.LR)
				}
			}
		}
	}
}

func TestPyTorchLinearAllMLXTrainingAndZeroLR(t *testing.T) {
	if !gpu.Available() {
		t.Skip("MLX backend not available")
	}
	for _, optimizer := range []string{"adamw", "lamb", "muon"} {
		for _, frozen := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/frozen=%v", optimizer, frozen), func(t *testing.T) {
				lr := 0.0001
				if frozen {
					lr = 0
				}
				cfg := affineOptimizerTestConfig(t, optimizer, fmt.Sprintf(`"lr":%g`, lr))
				prog, err := BuildIRProgramFromConfig(cfg)
				if err != nil {
					t.Fatal(err)
				}
				trainer, err := initGPUTrainer(prog, cfg, nil, nil)
				if err != nil {
					t.Fatal(err)
				}
				defer trainer.CloseTrainer()
				before, err := readTrainerWeights(trainer)
				if err != nil {
					t.Fatal(err)
				}
				for step := 0; step < 4; step++ {
					loss, err := trainer.TrainStepGPU([]int{1, 2, 3, 4, 2, 3, 4, 5}, []int{2, 3, 4, 5, 3, 4, 5, 6}, 2, 4, float32(lr))
					if err != nil || math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
						t.Fatalf("step %d: loss=%g err=%v", step, loss, err)
					}
				}
				after, err := readTrainerWeights(trainer)
				if err != nil {
					t.Fatal(err)
				}
				if reflect.DeepEqual(before, after) != frozen {
					t.Fatalf("weights unchanged=%v want %v", reflect.DeepEqual(before, after), frozen)
				}
			})
		}
	}
}
