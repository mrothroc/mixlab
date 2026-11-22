package train

import (
	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/data"
	"math"
	"strings"
	"testing"
)

func TestGridPolarBatchValidation(t *testing.T) {
	cfg, err := LoadArchConfig("../examples/grid_polar_tiny.json")
	if err != nil {
		t.Fatal(err)
	}
	for _, supervised := range []bool{false, true} {
		p, err := arch.BuildGridIRProgram(cfg, supervised)
		if err != nil {
			t.Fatal(err)
		}
		b := data.GridBatch{BatchSize: 2, Count: 1, IDs: []string{"first", "padded"}, Inputs: make([]float32, 256), Targets: make([]float32, 128), LossMask: make([]float32, 128), Vectors: []float32{2.5, 3, 2.5, 3}}
		if _, err = makeGridInputs(p.Inputs, &b, arch.GridPolarInputChecks(p)...); err != nil {
			t.Fatal(err)
		}
		for _, v := range []float32{-1, 8, float32(math.NaN()), float32(math.Inf(1))} {
			b.Vectors[2] = v
			if _, err = makeGridInputs(p.Inputs, &b, arch.GridPolarInputChecks(p)...); err == nil || !strings.Contains(err.Error(), `record "padded"`) {
				t.Fatal(err)
			}
		}
	}
}
