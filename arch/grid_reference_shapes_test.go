package arch

import (
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"testing"
)

func TestGridReferenceRecipesAndIntermediateShapes(t *testing.T) {
	for _, label := range []string{"two_channel", "three_channel"} {
		for stage := 1; stage <= 2; stage++ {
			t.Run(fmt.Sprintf("%s_stage%d", label, stage), func(t *testing.T) {
				base := fmt.Sprintf("../examples/grid_unet_reference/%s_stage%d", label, stage)
				cfg, err := LoadArchConfig(base + ".json")
				if err != nil {
					t.Fatal(err)
				}
				raw, err := os.ReadFile(base + "_shapes.json")
				if err != nil {
					t.Fatal(err)
				}
				var want map[string][]int
				if err = json.Unmarshal(raw, &want); err != nil {
					t.Fatal(err)
				}
				shapes := map[string][]int{"x": {1, 256, 256, cfg.InputAdapter.Channels}}
				for _, w := range cfg.Blocks[0].Weights {
					shapes[w.Name], err = gridWeightShape(w.Shape, cfg)
					if err != nil {
						t.Fatal(err)
					}
				}
				for _, op := range cfg.Blocks[0].Ops {
					s, e := gridOpShape(op, shapes, map[string]int{})
					if e != nil {
						t.Fatal(e)
					}
					if !reflect.DeepEqual(s, want[op.Output]) {
						t.Fatalf("%s: native %v reference %v", op.Output, s, want[op.Output])
					}
					shapes[op.Output] = s
				}
				if len(want) != len(cfg.Blocks[0].Ops) {
					t.Fatal("incomplete intermediate shape coverage")
				}
				total, _, err := ParameterCountsFromConfig(cfg)
				if err != nil {
					t.Fatal(err)
				}
				wantTotal := int64(13274031)
				if label == "three_channel" {
					wantTotal = 13275315
				}
				if total != wantTotal {
					t.Fatal(total, wantTotal)
				}
				if cfg.Training.WeightDecay != 0 || cfg.Training.MatrixWeightDecay != 0 || cfg.Training.ScalarWeightDecay != 0 {
					t.Fatal("reference recipe decays weights")
				}
			})
		}
	}
}
