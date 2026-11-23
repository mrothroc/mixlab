package train

import (
	"fmt"
	"math"
	"reflect"
	"slices"
	"strconv"
	"testing"

	"github.com/mrothroc/mixlab/data"
)

func gridBenchmarkSteps(value string) (int, error) {
	if value == "" {
		return 25, nil
	}
	steps, err := strconv.Atoi(value)
	if err != nil || steps < 6 || steps > 1000 {
		return 0, fmt.Errorf("MIXLAB_GRID_BENCH_STEPS must be in [6,1000]")
	}
	return steps, nil
}

func TestGridBenchmarkSteps(t *testing.T) {
	for value, want := range map[string]int{"": 25, "6": 6, "100": 100, "1000": 1000} {
		got, err := gridBenchmarkSteps(value)
		if err != nil || got != want {
			t.Fatalf("%q: %d, %v; want %d", value, got, err, want)
		}
	}
	for _, value := range []string{"0", "5", "-1", "1001", "1.5", "invalid", "9999999999999999999999999"} {
		if _, err := gridBenchmarkSteps(value); err == nil {
			t.Fatalf("accepted %q", value)
		}
	}
}

func gridBenchmarkMedian(values []float64) float64 {
	x := slices.Clone(values)
	slices.Sort(x)
	if len(x) == 0 {
		return 0
	}
	return (x[(len(x)-1)/2] + x[len(x)/2]) / 2
}

// Keep equal-name/equal-shape tensors identical across architecture arms. A
// widened kernel retains its own fan-in-correct initialization, not a slice copy.
func gridBenchmarkWeights(cfg, baseline *ArchConfig) ([][]float32, int, error) {
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		return nil, 0, err
	}
	w := initWeightData(shapes, cfg.Training.Seed, cfg.Training.WeightInit, cfg.Training.WeightInitStd)
	if baseline == nil {
		return w, 0, nil
	}
	baseShapes, err := computeWeightShapes(baseline)
	if err != nil {
		return nil, 0, err
	}
	base := initWeightData(baseShapes, baseline.Training.Seed, baseline.Training.WeightInit, baseline.Training.WeightInitStd)
	indices := make(map[string]int, len(baseShapes))
	for i, s := range baseShapes {
		indices[s.Name] = i
	}
	matched := 0
	for i, s := range shapes {
		if j, ok := indices[s.Name]; ok && reflect.DeepEqual(s.Shape, baseShapes[j].Shape) {
			copy(w[i], base[j])
			matched++
		}
	}
	return w, matched, nil
}

func gridBenchmarkBatch(cfg *ArchConfig) data.GridBatch {
	i := cfg.InputAdapter
	n, pixels := cfg.Training.BatchSize, i.Height*i.Width
	b := data.GridBatch{BatchSize: n, Count: n,
		Inputs: make([]float32, n*pixels*i.Channels), Targets: make([]float32, n*pixels*cfg.DenseRegression.TargetChannels),
		LossMask: make([]float32, n*pixels*cfg.DenseRegression.TargetChannels), Vectors: make([]float32, n*i.VectorDim)}
	for j := range b.Inputs {
		b.Inputs[j] = float32((j*17+412)%1009) / 1008
	}
	for j := range b.Targets {
		b.Targets[j] = float32((j*7+412)%251) / 250
		if j%11 != 0 {
			b.LossMask[j] = 1
		}
	}
	return b
}

func TestGridBenchmarkHelpers(t *testing.T) {
	x := []float64{4, 1, 3, 2}
	if gridBenchmarkMedian(x) != 2.5 || x[0] != 4 || gridBenchmarkMedian(nil) != 0 || gridBenchmarkMedian(x[:3]) != 3 {
		t.Fatal("median or input mutation")
	}
	cfg, err := LoadArchConfig("../examples/grid_polar_tiny.json")
	if err != nil {
		t.Fatal(err)
	}
	w, matched, err := gridBenchmarkWeights(cfg, cfg)
	if err != nil || matched != len(w) {
		t.Fatal(matched, err)
	}
	a, b := gridBenchmarkBatch(cfg), gridBenchmarkBatch(cfg)
	if !reflect.DeepEqual(a, b) || a.Count != cfg.Training.BatchSize {
		t.Fatal("synthetic batch is not deterministic")
	}
	for _, row := range w {
		for _, v := range row {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatal("invalid initial weight")
			}
		}
	}
}
