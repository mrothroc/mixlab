//go:build mlx && cgo && (darwin || linux)

package train

import (
	"fmt"
	"math"
	"testing"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/gpu"
)

func dynamicsDecayConfig(t *testing.T, block, optimizer, policy string, bidirectional bool) *ArchConfig {
	t.Helper()
	cfg, err := ParseArchConfig([]byte(fmt.Sprintf(`{
		"model_dim":8,"vocab_size":16,"seq_len":4,
		"blocks":[{"type":%q,"inner_dim":8,"state_size":4,"n_groups":2,
			"heads":2,"d_k":4,"d_v":4,"bidirectional":%t}],
		"training":{"objective":"mlm","mlm_mask_token_id":1,
			"optimizer":%q,"lr":0.01,"scalar_lr":0.002,"matrix_lr":0.003,
			"weight_decay":0.05,"scalar_weight_decay":0.07,"matrix_weight_decay":0.09,
			"weight_decay_policy":%q,"batch_tokens":4}
	}`, block, bidirectional, optimizer, policy)), "dynamics-decay")
	if err != nil {
		t.Fatal(err)
	}
	return cfg
}

func TestBuildTrainerOptimizerSpecDynamicsNoDecay(t *testing.T) {
	for _, block := range []string{"mamba3-canonical", "gated_deltanet"} {
		for _, optimizer := range []string{"adamw", "lamb", "muon", "muon_eq_r", "normuon"} {
			for _, policy := range []string{"matrix_only", "all"} {
				for _, bidirectional := range []bool{false, true} {
					t.Run(fmt.Sprintf("%s/%s/%s/bidir=%t", block, optimizer, policy, bidirectional), func(t *testing.T) {
						cfg := dynamicsDecayConfig(t, block, optimizer, policy, bidirectional)
						shapes, err := computeWeightShapes(cfg)
						if err != nil {
							t.Fatal(err)
						}
						spec, err := buildTrainerOptimizerSpec(cfg, shapes)
						if err != nil {
							t.Fatal(err)
						}
						seen := map[string]int{}
						for i, shape := range shapes {
							if shape.Name != "A_log" && shape.Name != "dt_bias" {
								if len(shape.Shape) == 2 && !shape.IsNormScale && !spec.Weights[i].Decay {
									t.Errorf("ordinary matrix %s lost decay", shape.Name)
								}
								continue
							}
							seen[shape.Name]++
							weight := spec.Weights[i]
							group := spec.Groups[weight.GroupIndex]
							if weight.Decay || !shape.ForceNoDecay || group.WeightDecay <= 0 {
								t.Errorf("%s: shape=%+v weight=%+v group=%+v", shape.Name, shape, weight, group)
							}
							wantLR := float32(0.002)
							if len(shape.Shape) == 2 {
								wantLR = 0.003
							}
							if group.LR != wantLR {
								t.Errorf("%s LR=%g want %g", shape.Name, group.LR, wantLR)
							}
						}
						if seen["A_log"] == 0 || seen["dt_bias"] == 0 {
							t.Fatalf("missing dynamics weights: %v", seen)
						}
					})
				}
			}
		}
	}
}

// A zero-gradient loss isolates decay from learning and exercises the native
// optimizer with metadata collected from a real model, including rank-2 A_log.
func TestDynamicsNoDecayNativeUpdate(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}
	for _, block := range []string{"mamba3-canonical", "gated_deltanet"} {
		for _, optimizer := range []string{"adamw", "lamb"} {
			t.Run(block+"/"+optimizer, func(t *testing.T) {
				cfg := dynamicsDecayConfig(t, block, optimizer, "all", true)
				shapes, err := computeWeightShapes(cfg)
				if err != nil {
					t.Fatal(err)
				}
				spec, err := buildTrainerOptimizerSpec(cfg, shapes)
				if err != nil {
					t.Fatal(err)
				}
				prog := arch.NewProgram(len(shapes))
				prog.DeclareInput("zero", arch.TensorFloat32, []int{1})
				prog.Full([]int{1}, 0, "loss")
				handles := make([]int64, len(shapes))
				for i, shape := range shapes {
					n := 1
					for _, dim := range shape.Shape {
						n *= dim
					}
					h, err := gpu.FromDataShape(repeatFloat32Train(n, 1), shape.Shape)
					if err != nil {
						t.Fatal(err)
					}
					defer gpu.FreeHandle(h)
					handles[i] = h
					flat := fmt.Sprintf("flat%d", i)
					term := fmt.Sprintf("term%d", i)
					prog.Reshape(fmt.Sprintf("w%d", i), []int{n}, flat)
					prog.MeanAxis(flat, 0, term)
					prog.Mul(term, "zero", term)
					prog.Add("loss", term, "loss")
				}
				prog.MeanAxis("loss", 0, "loss")
				lowered, err := gpu.LowerIRProgram(prog)
				if err != nil {
					t.Fatal(err)
				}
				defer lowered.Destroy()
				trainer, err := gpu.CreateTrainer(lowered, handles, spec)
				if err != nil {
					t.Fatal(err)
				}
				defer gpu.TrainerDestroy(trainer)
				for step := 0; step < 3; step++ {
					if _, err := gpu.TrainerStep(trainer, []gpu.TensorInput{
						{Name: "zero", DType: gpu.TensorFloat32, Shape: []int{1}, Data: []float32{0}},
					}); err != nil {
						t.Fatal(err)
					}
				}
				// Assert by name, not by ForceNoDecay. Keying the check off the
				// same metadata under test makes it self-consistent with
				// whatever policy is in force, so it passes even when the
				// exemption is missing entirely -- the regression it exists to
				// catch. These tensors must not decay whatever the metadata says.
				protectedDynamics := map[string]bool{"A_log": true, "dt_bias": true}
				seenProtected := 0
				seenDecayed := 0
				for i, shape := range shapes {
					size, err := gpu.TrainerWeightSize(trainer, i)
					if err != nil {
						t.Fatal(err)
					}
					got := make([]float32, size)
					if err := gpu.TrainerReadWeight(trainer, i, got); err != nil {
						t.Fatal(err)
					}
					for _, value := range got {
						if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
							t.Fatalf("%s non-finite weight %g", shape.Name, value)
						}
						if protectedDynamics[shape.Name] && value != 1 {
							t.Fatalf("%s decayed with zero gradient: %g", shape.Name, value)
						}
						if shape.ForceNoDecay && value != 1 {
							t.Fatalf("%s decayed with zero gradient: %g", shape.Name, value)
						}
						if spec.Weights[i].Decay && value >= 1 {
							t.Fatalf("%s ordinary weight did not decay: %g", shape.Name, value)
						}
					}
					if protectedDynamics[shape.Name] {
						seenProtected++
					}
					if spec.Weights[i].Decay {
						seenDecayed++
					}
				}
				if seenProtected != len(protectedDynamics) {
					t.Fatalf("saw %d of %d protected dynamics tensors", seenProtected, len(protectedDynamics))
				}
				// Without this, "protected weights stayed at 1" is satisfied by a
				// run where decay never applied to anything.
				if seenDecayed == 0 {
					t.Fatal("no ordinary weight decayed; the decay term was not active")
				}
			})
		}
	}
}
