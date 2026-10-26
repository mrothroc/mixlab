package train

import (
	"reflect"
	"testing"

	"github.com/mrothroc/mixlab/gpu"
)

func TestGridMuonOptimizerReport(t *testing.T) {
	for _, optimizer := range []string{"muon", "muon_eq_r", "normuon", "adamw", "lamb"} {
		t.Run(optimizer, func(t *testing.T) {
			cfg, err := LoadArchConfig("../examples/grid_muon_tiny.json")
			if err != nil {
				t.Fatal(err)
			}
			cfg.Training.Optimizer = optimizer
			cfg.Training.MatrixLR = .023
			cfg.Training.ScalarLR = .004
			cfg.Training.MatrixWeightDecay = .07
			if !cfg.Training.UsesMuon() {
				cfg.Training.MuonExclude = nil
			}
			r, spec := optimizerReportForTest(t, cfg, nil)
			for i, w := range r.Tensors {
				group, lr, wd := "scalar", float32(.004), float32(0)
				var shape []int
				if i%2 == 0 {
					lr, wd = .023, .07
					group = "extra:grid_kernel"
					if cfg.Training.UsesMuon() {
						group = "matrix"
						switch i {
						case 2:
							shape = []int{4, 12}
						case 4:
							shape = []int{3, 16}
						default:
							group = "extra:grid_muon_excluded"
							lr = .004
						}
					}
				}
				if w.Group != group || *w.ConfiguredLR != lr || w.EffectiveWeightDecay != wd || !reflect.DeepEqual(w.MuonMatrixShape, shape) {
					t.Fatalf("tensor %d=%+v want %s lr=%g wd=%g shape=%v", i, w, group, lr, wd, shape)
				}
				kind := gpu.OptimizerAdamW
				if group == "matrix" {
					kind = gpu.OptimizerMuon
				}
				if optimizer == "lamb" {
					kind = gpu.OptimizerLAMB
				}
				if spec.Groups[*w.GroupIndex].Kind != kind {
					t.Fatal("wrong optimizer", w)
				}
			}
		})
	}
}

func TestGridMuonFrozenAndResumeContract(t *testing.T) {
	cfg, err := LoadArchConfig("../examples/grid_muon_tiny.json")
	if err != nil {
		t.Fatal(err)
	}
	_, spec := optimizerReportForTest(t, cfg, nil)
	hash, err := optimizerSpecHash(spec)
	if err != nil {
		t.Fatal(err)
	}
	spec.Weights[4].MuonView = gpu.MuonMatrixOHWI
	changed, err := optimizerSpecHash(spec)
	if err != nil {
		t.Fatal(err)
	}
	if hash == changed {
		t.Fatal("view is missing from resume compatibility hash")
	}
	cfg.Training.Freeze = []string{"network.up.*"}
	r, spec := optimizerReportForTest(t, cfg, nil)
	if !r.Tensors[4].Frozen || !r.Tensors[5].Frozen || spec.Weights[4].MuonView != gpu.MuonMatrixIdentity || r.Tensors[4].GroupIndex != nil {
		t.Fatal("frozen kernel assigned optimizer", r.Tensors[4])
	}
}

func TestGridMuonTorchStateUnchanged(t *testing.T) {
	cfg, err := LoadArchConfig("../examples/grid_muon_tiny.json")
	if err != nil {
		t.Fatal(err)
	}
	muonShapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatal(err)
	}
	w := initWeightData(muonShapes, 42, "", 0)
	m := torchStateMap{Format: "mixlab.torch_state_map.v1"}
	for _, s := range muonShapes {
		entry := torchStateMapping{Source: s.Name, Target: s.Name, Transform: "identity", Shape: s.Shape}
		if len(s.Shape) == 4 {
			entry.Transform = "transpose"
			entry.Axes = []int{0, 3, 1, 2}
			entry.Shape = []int{s.Shape[0], s.Shape[3], s.Shape[1], s.Shape[2]}
		}
		m.Mappings = append(m.Mappings, entry)
	}
	want, err := mapTorchState(m, muonShapes, w)
	if err != nil {
		t.Fatal(err)
	}
	cfg.Training.Optimizer = "adamw"
	cfg.Training.MuonExclude = nil
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatal(err)
	}
	got, err := mapTorchState(m, shapes, initWeightData(shapes, 42, "", 0))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatal("optimizer policy changed exported tensors")
	}
}

func TestGridMuonReferenceRouting(t *testing.T) {
	cfg, err := LoadArchConfig("../arch/testdata/grid_reference/two_channel.json")
	if err != nil {
		t.Fatal(err)
	}
	cfg.Training.Optimizer = "muon"
	r, _ := optimizerReportForTest(t, cfg, nil)
	kernels := 0
	var parameters int64
	for _, w := range r.Tensors {
		if w.Frozen || len(w.Shape) != 4 {
			continue
		}
		kernels++
		parameters += w.Parameters
		if w.Group != "matrix" || w.Optimizer != "muon" || len(w.MuonMatrixShape) != 2 || int64(w.MuonMatrixShape[0])*int64(w.MuonMatrixShape[1]) != w.Parameters {
			t.Fatalf("unrouted reference kernel: %+v", w)
		}
	}
	if kernels != 19 || parameters < 10_000_000 {
		t.Fatalf("unexpected reference coverage: kernels=%d parameters=%d", kernels, parameters)
	}
}
