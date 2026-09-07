//go:build mlx && cgo && (darwin || linux)

package train

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/mrothroc/mixlab/gpu"
)

func TestBuildTrainerOptimizerSpecDynamicsStateLR(t *testing.T) {
	for _, block := range []string{"mamba3-canonical", "gated_deltanet"} {
		for _, optimizer := range []string{"adamw", "lamb", "muon", "muon_eq_r", "normuon"} {
			for _, bidir := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/%s/bidir=%t", block, optimizer, bidir), func(t *testing.T) {
					cfg := dynamicsDecayConfig(t, block, optimizer, "all", bidir)
					baseShapes, err := computeWeightShapes(cfg)
					if err != nil {
						t.Fatal(err)
					}
					base, err := buildTrainerOptimizerSpec(cfg, baseShapes)
					if err != nil {
						t.Fatal(err)
					}
					lr := 0.001
					cfg.Blocks[0].StateLR = &lr
					shapes, err := computeWeightShapes(cfg)
					if err != nil {
						t.Fatal(err)
					}
					spec, err := buildTrainerOptimizerSpec(cfg, shapes)
					if err != nil {
						t.Fatal(err)
					}
					seen := 0
					for i, shape := range shapes {
						w := spec.Weights[i]
						g := spec.Groups[w.GroupIndex]
						if shape.Name == "A_log" || shape.Name == "dt_bias" {
							seen++
							kind := gpu.OptimizerAdamW
							if optimizer == "lamb" {
								kind = gpu.OptimizerLAMB
							}
							if g.LR != 0.001 || g.Kind != kind || w.Decay {
								t.Fatalf("bad state group: %+v weight=%+v", g, w)
							}
						} else if !reflect.DeepEqual(g, base.Groups[base.Weights[i].GroupIndex]) {
							t.Fatalf("ordinary group changed for %s", shape.Name)
						}
					}
					if seen != 2 {
						t.Fatalf("state tensors=%d", seen)
					}
					cfg.Blocks[0].StateLR = nil
					resetShapes, err := computeWeightShapes(cfg)
					if err != nil {
						t.Fatal(err)
					}
					reset, err := buildTrainerOptimizerSpec(cfg, resetShapes)
					if err != nil {
						t.Fatal(err)
					}
					if !reflect.DeepEqual(base, reset) {
						t.Fatal("omitted state_lr changed grouping")
					}
				})
			}
		}
	}
}

func TestBuildTrainerOptimizerSpecDynamicsDistinctStateLRs(t *testing.T) {
	cfg := dynamicsDecayConfig(t, "mamba3-canonical", "adamw", "all", false)
	first, second := 0.001, 0.0001
	cfg.Blocks[0].StateLR = &first
	other := cfg.Blocks[0]
	other.Type = "gated_deltanet"
	other.StateLR = &second
	cfg.Blocks = append(cfg.Blocks, other)
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatal(err)
	}
	spec, err := buildTrainerOptimizerSpec(cfg, shapes)
	if err != nil {
		t.Fatal(err)
	}
	counts := map[float32]int{}
	groups := map[float32]int{}
	for i, shape := range shapes {
		if shape.OptimizerRole != "ssm_state" {
			continue
		}
		w := spec.Weights[i]
		g := spec.Groups[w.GroupIndex]
		if g.LR != shape.OptimizerLR || w.Decay {
			t.Fatalf("state LR crossed block boundaries: shape=%+v group=%+v", shape, g)
		}
		counts[g.LR]++
		groups[g.LR] = w.GroupIndex
	}
	if counts[float32(first)] != 2 || counts[float32(second)] != 2 || groups[float32(first)] == groups[float32(second)] {
		t.Fatalf("distinct rates did not get separate groups: counts=%v groups=%v", counts, groups)
	}
}
