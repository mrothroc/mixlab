//go:build mlx && cgo && (darwin || linux)

package train

import (
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/data"
)

// Generate with --phase secondU; the actual pinned reference's detach boundaries
// and canonical weight tensors are used, not an approximation of its architecture.
func TestGridPinnedRefinementFreeze(t *testing.T) {
	dir := os.Getenv("GRID_REFINEMENT_REFERENCE_DIR")
	if dir == "" {
		t.Skip("set GRID_REFINEMENT_REFERENCE_DIR from generate_grid_model_reference.py --phase secondU")
	}
	if !mlxAvailable() {
		t.Skip("MLX required")
	}
	load := func(name string) npzTensor {
		raw, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			t.Fatal(err)
		}
		x, err := decodeNPY(raw)
		if err != nil {
			t.Fatal(err)
		}
		return x
	}
	cfg, err := LoadArchConfig(filepath.Join(dir, "model2.json"))
	if err != nil {
		t.Fatal(err)
	}
	if len(cfg.Training.Freeze) == 0 {
		t.Fatal("fixture must use --phase secondU")
	}
	cfg.Training.WeightDecayPolicy = "all"
	cfg.Training.MatrixWeightDecay = .05
	cfg.Training.ScalarWeightDecay = .05
	x, ref := load("input.npy"), load("output2.npy")
	c := x.Shape[1]
	const pixels = 256 * 256
	b := data.GridBatch{Count: 1, BatchSize: 1, Geometry: data.GridGeometry{Channels: c, TargetChannels: 1, Height: 256, Width: 256}, Inputs: make([]float32, c*pixels), Targets: make([]float32, pixels), LossMask: make([]float32, pixels)}
	for j := 0; j < pixels; j++ {
		for channel := 0; channel < c; channel++ {
			b.Inputs[j*c+channel] = x.F32[channel*pixels+j]
		}
		b.LossMask[j] = 1
	}
	p, err := arch.BuildGridIRProgram(cfg, true)
	if err != nil {
		t.Fatal(err)
	}
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatal(err)
	}
	w, _, err := loadGridWeights(filepath.Join(dir, "weights.safetensors"), cfg, shapes, nil)
	if err != nil {
		t.Fatal(err)
	}
	tr, err := initGPUTrainer(p, cfg, w, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer tr.CloseTrainer()
	if _, err = tr.(gpuObjectiveOutputEvaluator).EvaluateObjectiveGPUWithOutputs(objectiveBatch{grid: &b}, 1, 0, []string{"predictions"}); err != nil {
		t.Fatal(err)
	}
	actual, err := readTrainerOutput(tr, "predictions", []int{1, 256, 256, 1})
	if err != nil {
		t.Fatal(err)
	}
	maxDiff := 0.
	for j, v := range actual {
		d := math.Abs(float64(v - ref.F32[j]))
		if math.IsNaN(d) || math.IsInf(d, 0) {
			t.Fatal("nonfinite reference difference")
		}
		maxDiff = math.Max(maxDiff, d)
	}
	if maxDiff > 1e-4 {
		t.Fatalf("refinement max abs difference=%g", maxDiff)
	}
	if err = submitPreparedStepGPU(tr, objectiveBatch{grid: &b}, 1, 0, 1e-4); err != nil {
		t.Fatal(err)
	}
	loss, err := tr.CollectLossGPU()
	if err != nil || math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
		t.Fatal(loss, err)
	}
	after, err := readTrainerWeights(tr)
	if err != nil {
		t.Fatal(err)
	}
	changed, frozen := 0, 0
	for j, s := range shapes {
		if s.Frozen {
			frozen++
			if !reflect.DeepEqual(w[j], after[j]) {
				t.Fatal("frozen reference weight changed", s.Name)
			}
		} else if !reflect.DeepEqual(w[j], after[j]) {
			changed++
		}
	}
	state, err := tr.(gpuTrainerStateReader).ReadTrainerState()
	if err != nil {
		t.Fatal(err)
	}
	for _, s := range state.Tensors {
		if shapes[s.WeightIndex].Frozen {
			t.Fatal("frozen reference weight has optimizer state")
		}
	}
	if frozen == 0 || changed == 0 {
		t.Fatal("reference did not exercise freeze and refinement update")
	}
	t.Logf("actual PyTorch secondU forward max_abs=%g; frozen=%d updated=%d; loss=%g", maxDiff, frozen, changed, loss)
}
