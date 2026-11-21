package train

import (
	"math"
	"reflect"
	"testing"
)

func TestGridRectangularMuonAndInitialization(t *testing.T) {
	cfg, err := LoadArchConfig("../examples/grid_rectangular_tiny.json")
	if err != nil {
		t.Fatal(err)
	}
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatal(err)
	}
	weights := initWeightData(shapes, 42, "", 0)
	for i, w := range weights {
		if shapes[i].PyTorchLinearFanIn != 18 {
			t.Fatal(shapes[i])
		}
		for _, v := range w {
			if math.Abs(float64(v)) > 1/math.Sqrt(18) {
				t.Fatal("rectangular fan-in bound", v)
			}
		}
	}
	cfg.Training.Optimizer = "muon"
	report, _ := optimizerReportForTest(t, cfg, nil)
	for i, want := range map[int][]int{0: {2, 18}, 2: {1, 18}} {
		if !reflect.DeepEqual(report.Tensors[i].MuonMatrixShape, want) {
			t.Fatal(report.Tensors[i])
		}
	}
}
