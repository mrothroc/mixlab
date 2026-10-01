package train

import (
	"encoding/json"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/mrothroc/mixlab/data"
)

func TestGridCustomInitialization(t *testing.T) {
	for _, kind := range []string{"custom_uniform", "custom_normal", "custom_one", "torch_linear_bias_uniform"} {
		s := []WeightShape{{Name: "kernel", Shape: []int{25000, 2, 2, 2}, InitMode: kind, InitScale: .1, PyTorchLinearFanIn: 100}}
		w := initWeightData(s, 42, "", 0)[0]
		again := initWeightData(s, 42, "", 0)[0]
		if !reflect.DeepEqual(w, again) {
			t.Fatal("not deterministic")
		}
		mean, variance, maxAbs := 0., 0., 0.
		for _, v := range w {
			mean += float64(v)
			variance += float64(v * v)
			maxAbs = math.Max(maxAbs, math.Abs(float64(v)))
		}
		mean /= float64(len(w))
		variance = variance/float64(len(w)) - mean*mean
		if kind == "custom_one" {
			if mean != 1 || variance != 0 {
				t.Fatal(mean, variance)
			}
			continue
		}
		want := .01 / 3
		if kind == "custom_normal" {
			want = .01
		} else if maxAbs > .1 || maxAbs < .099 {
			t.Fatal("uniform bound", maxAbs)
		}
		if math.Abs(variance-want) > want*.01 || math.Abs(mean) > .001 {
			t.Fatalf("kind=%s mean=%g variance=%g want=%g", kind, mean, variance, want)
		}
	}
}

func prepareTinyGrid(t *testing.T) string {
	return prepareTinyGridWithMask(t, 1)
}

func prepareTinyGridWithMask(t *testing.T, valid float32) string {
	t.Helper()
	if err := exec.Command("python3", "-c", "import numpy").Run(); err != nil {
		t.Skip("numpy required for grid prepare integration")
	}
	dir := t.TempDir()
	x := make([]float32, 3*2*8*8)
	y := make([]float32, 3*8*8)
	mask := make([]float32, len(y))
	for n := 0; n < 3; n++ {
		for j := 0; j < 64; j++ {
			v := float32((n*17+j)%31) / 31
			x[n*128+j] = v
			x[n*128+64+j] = 1 - v
			y[n*64+j] = v*.4 + .2
			mask[n*64+j] = valid
		}
	}
	x[0] = math.Float32frombits(0x80000000)
	writeNPYFloat32(t, filepath.Join(dir, "x.npy"), []int{3, 2, 8, 8}, x)
	writeNPYFloat32(t, filepath.Join(dir, "y.npy"), []int{3, 1, 8, 8}, y)
	writeNPYFloat32(t, filepath.Join(dir, "m.npy"), []int{3, 1, 8, 8}, mask)
	spec := map[string]any{"records_per_shard": 2, "splits": map[string]any{"train": map[string]string{"inputs": "x.npy", "targets": "y.npy", "masks": "m.npy"}, "val": map[string]string{"inputs": "x.npy", "targets": "y.npy", "masks": "m.npy"}, "predict": map[string]string{"inputs": "x.npy", "targets": "y.npy", "masks": "m.npy"}}}
	b, _ := json.Marshal(spec)
	source := filepath.Join(dir, "source.json")
	if err := os.WriteFile(source, b, 0600); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(dir, "prepared")
	if err := runPrepare(PrepareOptions{Input: source, Output: out, InputFormat: "grid"}); err != nil {
		t.Fatal(err)
	}
	return filepath.Join(out, data.DatasetManifestFilename)
}

func TestGridPrepareBatchAndMetrics(t *testing.T) {
	path := prepareTinyGrid(t)
	ds, err := data.OpenGridDataset(path, "train")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = ds.Close() }()
	if ds.Len() != 3 {
		t.Fatal(ds.Len())
	}
	b, err := ds.ReadBatch([]int{0, 1}, 2)
	if err != nil {
		t.Fatal(err)
	}
	if math.Float32bits(b.Inputs[0]) != 0x80000000 || b.Inputs[1] != 1 {
		t.Fatal("CHW/NHWC or signed-zero roundtrip")
	}
	b, err = ds.ReadBatch([]int{2}, 2)
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range b.LossMask[64:] {
		if v != 0 {
			t.Fatal("padded mask")
		}
	}
	p := append([]float32(nil), b.Targets...)
	for j := 0; j < 64; j++ {
		p[j] += 2
	}
	for j := 64; j < len(p); j++ {
		p[j] = 1000
	}
	m := gridMetrics{}
	if err = m.add(b, p); err != nil {
		t.Fatal(err)
	}
	if m.Count != 64 || math.Abs(m.MaskedRMSE()-2) > 1e-6 || m.UnmaskedCount != 64 {
		t.Fatalf("metrics %+v", m)
	}
}

func TestGridMetricsPoolNotAverage(t *testing.T) {
	m := gridMetrics{}
	b := data.GridBatch{Targets: []float32{0, 0}, LossMask: []float32{1, 0}, Count: 1, BatchSize: 1, Geometry: data.GridGeometry{Height: 1, Width: 2, TargetChannels: 1}}
	if err := m.add(b, []float32{2, 8}); err != nil {
		t.Fatal(err)
	}
	b.LossMask = []float32{1, 1}
	if err := m.add(b, []float32{1, 1}); err != nil {
		t.Fatal(err)
	}
	if m.MaskedMSE() != 2 || m.UnmaskedCount != 4 {
		t.Fatal(m)
	}
}
