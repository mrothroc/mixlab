package train

import (
	"math"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/arch"
)

func TestData2VecTauForStepAndEMAUpdate(t *testing.T) {
	if got := data2VecTauForStep(nil, 7); got != 0.999 {
		t.Fatalf("nil spec tau=%g want 0.999", got)
	}
	if got := data2VecTauForStep(&arch.Data2VecSpec{EMATau: 0.91}, 7); got != 0.91 {
		t.Fatalf("fixed tau=%g want 0.91", got)
	}

	spec := &arch.Data2VecSpec{
		EMATau:          0.99,
		EMATauStart:     0.5,
		EMATauEnd:       0.9,
		EMATauRampSteps: 10,
	}
	for _, tc := range []struct {
		step int
		want float64
	}{
		{step: 0, want: 0.5},
		{step: 5, want: 0.7},
		{step: 10, want: 0.9},
		{step: 20, want: 0.9},
	} {
		if got := data2VecTauForStep(spec, tc.step); math.Abs(got-tc.want) > 1e-12 {
			t.Fatalf("tau step=%d got=%g want=%g", tc.step, got, tc.want)
		}
	}

	ema := [][]float32{{10, 20}, {-10}}
	current := [][]float32{{2, 4}, {10}}
	updateData2VecEMAWeights(ema, current, spec, 5)
	requireFloat32SliceNear(t, "ema[0]", ema[0], []float32{7.6, 15.2}, 1e-6)
	requireFloat32SliceNear(t, "ema[1]", ema[1], []float32{-4}, 1e-6)
}

func TestNormalizeRowsInPlaceData2Vec(t *testing.T) {
	x := []float32{1, 2, 3, 4, 4, 4}
	normalizeRowsInPlace(x, 2, 3, 1e-5)

	invStd := float32(1 / math.Sqrt(2.0/3.0+1e-5))
	requireFloat32SliceNear(t, "normalized rows", x, []float32{-invStd, 0, invStd, 0, 0, 0}, 1e-6)
}

func TestFillData2VecTargetsFromOutputsAveragesAndNormalizes(t *testing.T) {
	names := []string{"layer_a", "layer_b"}
	outs := map[string][]float32{
		"layer_a": []float32{1, 3, 5, 7, -1, -3},
		"layer_b": []float32{3, 5, 7, 9, 1, 1},
	}

	targets := make([]float32, 8)
	if err := fillData2VecTargetsFromOutputs(targets, names, outs, 2, 3, arch.Data2VecTargetNormNone, 1e-5); err != nil {
		t.Fatalf("fillData2VecTargetsFromOutputs none: %v", err)
	}
	wantAvg := []float32{2, 4, 6, 8, 0, -1}
	requireFloat32SliceNear(t, "averaged targets", targets[:6], wantAvg, 1e-6)

	targets = make([]float32, 6)
	if err := fillData2VecTargetsFromOutputs(targets, names, outs, 2, 3, arch.Data2VecTargetNormLayer, 1e-5); err != nil {
		t.Fatalf("fillData2VecTargetsFromOutputs normalized: %v", err)
	}
	normalizeRowsOracle(wantAvg, 2, 3, 1e-5)
	requireFloat32SliceNear(t, "normalized targets", targets, wantAvg, 1e-6)
}

func TestFillData2VecTargetsFromOutputsRejectsBadOutputs(t *testing.T) {
	targets := make([]float32, 4)
	err := fillData2VecTargetsFromOutputs(targets, []string{"missing"}, map[string][]float32{}, 2, 2, arch.Data2VecTargetNormNone, 1e-5)
	if err == nil || !strings.Contains(err.Error(), "missing") {
		t.Fatalf("missing output error=%v, want missing output", err)
	}

	err = fillData2VecTargetsFromOutputs(targets, []string{"bad"}, map[string][]float32{"bad": []float32{1, 2, 3}}, 2, 2, arch.Data2VecTargetNormNone, 1e-5)
	if err == nil || !strings.Contains(err.Error(), "size") {
		t.Fatalf("bad size error=%v, want size error", err)
	}
}

func TestAttachData2VecTargetsCausalSkipZeros(t *testing.T) {
	teacher := &data2VecTeacher{
		cfg:       &ArchConfig{ModelDim: 3},
		modelDim:  3,
		targetBuf: []float32{1, 2, 3, 4, 5, 6},
		maskBuf:   []float32{1, 1},
	}
	batch := objectiveBatch{
		x:        []int{1, 2},
		y:        []int{2, 3},
		lossMask: []float32{1, 1},
	}

	got, err := attachData2VecTargets(teacher, batch, arch.ObjectiveCausal, 1, 2)
	if err != nil {
		t.Fatalf("attachData2VecTargets causal: %v", err)
	}
	requireFloat32SliceNear(t, "causal skip targets", got.data2vecTargets, []float32{0, 0, 0, 0, 0, 0}, 0)
	requireFloat32SliceNear(t, "causal skip mask", got.data2vecMask, []float32{0, 0}, 0)
}

func normalizeRowsOracle(x []float32, rows, cols int, eps float64) {
	for r := 0; r < rows; r++ {
		start := r * cols
		end := start + cols
		var mean float64
		for _, v := range x[start:end] {
			mean += float64(v)
		}
		mean /= float64(cols)
		var variance float64
		for _, v := range x[start:end] {
			d := float64(v) - mean
			variance += d * d
		}
		variance /= float64(cols)
		invStd := 1 / math.Sqrt(variance+eps)
		for i := start; i < end; i++ {
			x[i] = float32((float64(x[i]) - mean) * invStd)
		}
	}
}

func requireFloat32SliceNear(t *testing.T, label string, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s len got=%d want=%d\ngot=%v\nwant=%v", label, len(got), len(want), got, want)
	}
	for i := range got {
		if diff := math.Abs(float64(got[i] - want[i])); diff > tol {
			t.Fatalf("%s[%d] got=%g want=%g diff=%g tol=%g\ngot=%v\nwant=%v", label, i, got[i], want[i], diff, tol, got, want)
		}
	}
}
