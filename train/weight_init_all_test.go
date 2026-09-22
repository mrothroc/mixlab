package train

import (
	"math"
	"math/rand"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/arch"
)

func fullAffineTestConfig(t *testing.T) *ArchConfig {
	t.Helper()
	raw, err := os.ReadFile("../examples/vit_pytorch_init.json")
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := arch.ParseArchConfig(raw, "vit_init")
	if err != nil {
		t.Fatal(err)
	}
	cfg.ModelDim = 32 // Preserve the full layout with smaller numerical fixtures.
	return cfg
}

func TestPyTorchLinearAllClassifierCoverage(t *testing.T) {
	cfg := fullAffineTestConfig(t)
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if len(shapes) != 86 {
		t.Fatalf("classifier tensor count=%d, want 86", len(shapes))
	}
	for _, tc := range []struct{ mode, matrices, biases string }{
		{"pytorch_linear_all", "38/38", "20/20"},
		{"pytorch_linear", "2/38", "2/20"},
	} {
		summary := weightInitCoverage(shapes, tc.mode)
		if !strings.Contains(summary, tc.matrices+" ordinary affine matrices") || !strings.Contains(summary, tc.biases+" paired biases") {
			t.Fatal(summary)
		}
	}
	got := initWeightData(shapes, 42, "pytorch_linear_all", 0)
	if !reflect.DeepEqual(got, initWeightData(shapes, 42, "pytorch_linear_all", 0)) {
		t.Fatal("initialization is not deterministic")
	}
	for i, shape := range shapes {
		if fanIn := weightInitLinearFanIn(shape, "pytorch_linear_all"); fanIn > 0 {
			assertValuesWithin(t, shape.Name, got[i], 1/math.Sqrt(float64(fanIn)))
			assertAnyNonZero(t, shape.Name, got[i])
		}
	}
	// Adding metadata must not change any existing policy, RNG draw, or layout.
	oldShapes := append([]WeightShape(nil), shapes...)
	for i := range shapes {
		shapes[i].NormalInitStd = nil
		oldShapes[i].NormalInitStd = nil
		oldShapes[i].LinearFanIn = 0
	}
	for _, mode := range []string{"", "xavier_uniform", "normal", "gptbert", "gpt2", "pytorch_linear"} {
		if !reflect.DeepEqual(initWeightData(shapes, 11, mode, 0.02), initWeightData(oldShapes, 11, mode, 0.02)) {
			t.Fatalf("new metadata changed legacy mode %q", mode)
		}
	}
}

func TestPyTorchLinearAllFanInOracle(t *testing.T) {
	shapes := []WeightShape{
		{Name: "rectangular", Shape: []int{64, 1024}, LinearFanIn: 64},
		{Name: "bias", Shape: []int{1024}, LinearFanIn: 64, InitZero: true},
	}
	got := initWeightData(shapes, 17, "pytorch_linear_all", 0)
	rng := rand.New(rand.NewSource(17))
	var squares float64
	for i, value := range got[0] {
		want := float32((2*rng.Float64() - 1) / 8)
		if value != want {
			t.Fatalf("weight[%d]=%g want %g", i, value, want)
		}
		squares += float64(value) * float64(value)
	}
	if gotVar := squares / float64(len(got[0])); math.Abs(gotVar-1.0/(3*64)) > 0.0001 {
		t.Fatalf("second moment=%g, want 1/(3*fan_in)", gotVar)
	}
	assertValuesWithin(t, "bias", got[1], 1.0/8)
	assertAnyNonZero(t, "bias", got[1])
}

func TestPyTorchLinearAllPreservesSpecialTensors(t *testing.T) {
	shapes := []WeightShape{
		{Name: "wq", Shape: []int{8, 16}, LinearFanIn: 8},
		{Name: "bias", Shape: []int{16}, InitZero: true, LinearFanIn: 8},
		{Name: "position_embeddings", Shape: []int{10, 16}},
		{Name: "cls_token", Shape: []int{1, 16}, InitMode: arch.ClassTokenInitMode},
		{Name: "sparse_gate", Shape: []int{8, 16}, InitZero: true},
		{Name: "resid_mix", Shape: []int{2, 16}},
		{Name: "ssm_state", Shape: []int{8, 16}, InitLogArange: true},
		{Name: "s4d_C_real", Shape: []int{8, 16}, InitMode: "s4d_C_normal"},
		{Name: "special_projection", Shape: []int{8, 16}, InitMode: "torch_linear_uniform", LinearFanIn: 8},
		{Name: "norm", Shape: []int{16}, IsNormScale: true},
	}
	got := initWeightData(shapes, 42, "pytorch_linear_all", 0)
	want := initWeightData(shapes, 42, "xavier_uniform", 0)
	for i := 2; i < len(shapes); i++ {
		if !reflect.DeepEqual(got[i], want[i]) {
			t.Fatalf("new mode changed specialized/unmarked tensor %s", shapes[i].Name)
		}
	}
}

func TestEmbeddingInitOverrides(t *testing.T) {
	cfg := fullAffineTestConfig(t)
	cfg.ModelDim = 4096
	cfg.Blocks = cfg.Blocks[:0] // Test embeddings without allocating large blocks.
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatal(err)
	}
	base := append([]WeightShape(nil), shapes...)
	for i := range base {
		base[i].NormalInitStd = nil
	}
	got := initWeightData(shapes, 123, "pytorch_linear_all", 0)
	want := initWeightData(base, 123, "pytorch_linear_all", 0)
	seen := 0
	for i, shape := range shapes {
		if shape.NormalInitStd == nil {
			if !reflect.DeepEqual(got[i], want[i]) {
				t.Fatalf("embedding override changed %s", shape.Name)
			}
			continue
		}
		seen++
		var sq float64
		for _, x := range got[i] {
			sq += float64(x) * float64(x)
		}
		if math.Abs(sq/float64(len(got[i]))-1) > 0.05 {
			t.Fatalf("%s not initialized at unit variance", shape.Name)
		}
		zero := 0.0
		shapes[i].NormalInitStd = &zero
	}
	if seen != 2 {
		t.Fatalf("normal overrides=%d, want 2", seen)
	}
	zeroed := initWeightData(shapes, 123, "pytorch_linear_all", 0)
	for i, shape := range shapes {
		if shape.NormalInitStd != nil {
			assertValuesWithin(t, shape.Name, zeroed[i], 0)
		}
	}
}
