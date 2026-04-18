//go:build mlx && cgo && darwin

package parity_test

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/gpu"
)

type referenceFile struct {
	Config struct {
		Batch    int     `json:"batch"`
		SeqLen   int     `json:"seq_len"`
		ModelDim int     `json:"model_dim"`
		Heads    int     `json:"heads"`
		Vocab    int     `json:"vocab_size"`
		MLPMult  float64 `json:"mlp_mult"`
	} `json:"config"`
	Tokens  []int32       `json:"tokens"`
	Targets []int32       `json:"targets"`
	Loss    float64       `json:"loss"`
	Weights []weightEntry `json:"weights"`
}

type weightEntry struct {
	Name   string    `json:"name"`
	Shape  []int     `json:"shape"`
	Values []float32 `json:"values"`
}

func TestPyTorchForwardLossParity(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("MLX parity test is darwin-only")
	}
	if !gpu.Available() {
		t.Skip("MLX backend not available")
	}

	ref := loadReference(t)
	blocks := []arch.BlockSpec{
		{Type: "plain", Heads: ref.Config.Heads},
		{Type: "swiglu"},
	}
	prog, err := arch.BuildIRProgram(
		ref.Config.ModelDim,
		ref.Config.Vocab,
		ref.Config.SeqLen,
		ref.Config.Batch,
		ref.Config.MLPMult,
		false,
		false,
		false,
		false,
		0,
		blocks,
	)
	if err != nil {
		t.Fatalf("BuildIRProgram: %v", err)
	}

	metas, err := arch.CollectWeightShapes(
		ref.Config.ModelDim,
		ref.Config.Vocab,
		ref.Config.SeqLen,
		ref.Config.MLPMult,
		false,
		false,
		false,
		false,
		blocks,
	)
	if err != nil {
		t.Fatalf("CollectWeightShapes: %v", err)
	}
	if len(ref.Weights) != len(metas) {
		t.Fatalf("reference weight count=%d, want %d", len(ref.Weights), len(metas))
	}
	if prog.NumWeights != len(metas) {
		t.Fatalf("program weight count=%d, want %d", prog.NumWeights, len(metas))
	}

	gpuProg, err := gpu.LowerIRProgram(prog)
	if err != nil {
		t.Fatalf("LowerIRProgram: %v", err)
	}
	defer gpuProg.Destroy()

	handles := make([]int64, len(ref.Weights))
	for i, w := range ref.Weights {
		meta := metas[i]
		if w.Name != meta.Name {
			t.Fatalf("weight %d name=%q, want %q", i, w.Name, meta.Name)
		}
		if !sameShape(w.Shape, meta.Shape) {
			t.Fatalf("weight %d (%s) shape=%v, want %v", i, w.Name, w.Shape, meta.Shape)
		}
		if got, want := len(w.Values), elemCount(meta.Shape); got != want {
			t.Fatalf("weight %d (%s) values=%d, want %d", i, w.Name, got, want)
		}

		rows, cols := 1, len(w.Values)
		if len(meta.Shape) == 2 {
			rows, cols = meta.Shape[0], meta.Shape[1]
		}
		handles[i], err = gpu.FromData(append([]float32(nil), w.Values...), rows, cols)
		if err != nil {
			gpu.FreeHandles(handles[:i])
			t.Fatalf("upload weight %d (%s): %v", i, w.Name, err)
		}
	}
	defer gpu.FreeHandles(handles)

	trainer, err := gpu.CreateTrainer(gpuProg, handles, trivialOptimizerSpec(len(handles)))
	if err != nil {
		t.Fatalf("CreateTrainer: %v", err)
	}
	defer gpu.TrainerDestroy(trainer)

	inputs := []gpu.TensorInput{
		{Name: "tokens", DType: gpu.TensorInt32, Shape: []int{ref.Config.Batch, ref.Config.SeqLen}, Data: ref.Tokens},
		{Name: "targets", DType: gpu.TensorInt32, Shape: []int{ref.Config.Batch * ref.Config.SeqLen}, Data: ref.Targets},
	}
	loss, err := gpu.TrainerEvaluate(trainer, inputs)
	if err != nil {
		t.Fatalf("TrainerEvaluate: %v", err)
	}

	const tolerance = 1e-4
	if diff := math.Abs(float64(loss) - ref.Loss); diff > tolerance {
		t.Fatalf("loss mismatch: mixlab=%.8f pytorch=%.8f diff=%g tolerance=%g", loss, ref.Loss, diff, tolerance)
	}
	t.Logf("loss parity: mixlab=%.8f pytorch=%.8f", loss, ref.Loss)
}

func loadReference(t *testing.T) referenceFile {
	t.Helper()
	path := filepath.Join("reference_weights.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v; run `python3 tests/parity/reference_pytorch.py` from the repo root first", path, err)
	}
	var ref referenceFile
	if err := json.Unmarshal(data, &ref); err != nil {
		t.Fatalf("parse %s: %v", path, err)
	}
	return ref
}

func trivialOptimizerSpec(n int) gpu.TrainerOptimizerSpec {
	weights := make([]gpu.WeightOptimizer, n)
	for i := range weights {
		weights[i] = gpu.WeightOptimizer{GroupIndex: 0}
	}
	return gpu.TrainerOptimizerSpec{
		Groups: []gpu.OptimizerGroup{{
			Kind:    gpu.OptimizerAdamW,
			LR:      1e-3,
			Beta1:   0.9,
			Beta2:   0.95,
			Epsilon: 1e-8,
		}},
		Weights:       weights,
		DefaultBaseLR: 1e-3,
	}
}

func sameShape(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func elemCount(shape []int) int {
	n := 1
	for _, dim := range shape {
		n *= dim
	}
	return n
}
