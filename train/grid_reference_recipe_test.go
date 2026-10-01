package train

import (
	"path/filepath"
	"testing"

	"github.com/mrothroc/mixlab/gpu"
)

func TestGridReferenceRecipeOptimizer(t *testing.T) {
	files, err := filepath.Glob("../examples/grid_unet_reference/*stage[12].json")
	if err != nil || len(files) != 4 {
		t.Fatal(files, err)
	}
	for _, file := range files {
		cfg, e := LoadArchConfig(file)
		if e != nil {
			t.Fatal(e)
		}
		shapes, e := computeWeightShapes(cfg)
		if e != nil {
			t.Fatal(e)
		}
		spec, e := buildTrainerOptimizerSpec(cfg, shapes)
		if e != nil {
			t.Fatal(e)
		}
		for _, g := range spec.Groups {
			if g.Kind != gpu.OptimizerAdamW || g.WeightDecay != 0 || g.Beta1 != float32(.9) || g.Beta2 != float32(.999) || g.Epsilon != float32(1e-8) {
				t.Fatalf("recipe group: %+v", g)
			}
		}
	}
}
