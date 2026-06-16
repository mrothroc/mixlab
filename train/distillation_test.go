package train

import (
	"math"
	"testing"

	"github.com/mrothroc/mixlab/arch"
)

func TestDistillationTeacherProbBufferReuseAndClear(t *testing.T) {
	ensemble := &distillationEnsemble{vocabSize: 4}
	first := ensemble.teacherProbBuffer(3)
	if len(first) != 12 {
		t.Fatalf("first buffer len=%d want 12", len(first))
	}
	for i := range first {
		first[i] = 1
	}

	second := ensemble.teacherProbBuffer(3)
	if len(second) != 12 {
		t.Fatalf("second buffer len=%d want 12", len(second))
	}
	if &first[0] != &second[0] {
		t.Fatal("teacher probability buffer was not reused for same-sized batches")
	}
	for i, v := range second {
		if v != 0 {
			t.Fatalf("second[%d]=%g want cleared zero", i, v)
		}
	}

	larger := ensemble.teacherProbBuffer(5)
	if len(larger) != 20 {
		t.Fatalf("larger buffer len=%d want 20", len(larger))
	}
}

func TestDistillationEnsembleStrategiesMatchOracle(t *testing.T) {
	const vocab = 3
	teachers := [][]float32{
		{
			0.25, -0.50, 1.10,
			-0.30, 0.80, 0.05,
		},
		{
			1.20, -0.10, -0.40,
			0.15, -0.25, 0.95,
		},
	}
	for _, strategy := range []string{arch.DistillationMeanLogits, arch.DistillationMeanLogProbs} {
		t.Run(strategy, func(t *testing.T) {
			got := make([]float32, len(teachers[0]))
			for _, logits := range teachers {
				accumulateTeacherLogits(got, logits, strategy, vocab)
			}
			finalizeTeacherProbs(got, len(teachers), vocab)
			want := distillationEnsembleOracle(teachers, strategy, vocab)
			if diff := maxAbsDiffTrain(got, want); diff > 1e-6 {
				t.Fatalf("%s max diff=%g\ngot=%v\nwant=%v", strategy, diff, got, want)
			}
		})
	}
}

func distillationEnsembleOracle(teachers [][]float32, strategy string, vocab int) []float32 {
	out := make([]float32, len(teachers[0]))
	rows := len(out) / vocab
	for row := 0; row < rows; row++ {
		start := row * vocab
		switch strategy {
		case arch.DistillationMeanLogProbs:
			for col := 0; col < vocab; col++ {
				var logProbSum float64
				for _, logits := range teachers {
					logProbSum += logSoftmaxAt(logits[start:start+vocab], col)
				}
				out[start+col] = float32(logProbSum / float64(len(teachers)))
			}
			softmaxFloat32(out[start : start+vocab])
		default:
			avg := make([]float32, vocab)
			for col := 0; col < vocab; col++ {
				var sum float64
				for _, logits := range teachers {
					sum += float64(logits[start+col])
				}
				avg[col] = float32(sum / float64(len(teachers)))
			}
			softmaxFloat32(avg)
			copy(out[start:start+vocab], avg)
		}
	}
	return out
}

func logSoftmaxAt(row []float32, col int) float64 {
	maxVal := float64(row[0])
	for _, v := range row[1:] {
		if float64(v) > maxVal {
			maxVal = float64(v)
		}
	}
	var sum float64
	for _, v := range row {
		sum += math.Exp(float64(v) - maxVal)
	}
	return float64(row[col]) - maxVal - math.Log(sum)
}

func softmaxFloat32(row []float32) {
	maxVal := float64(row[0])
	for _, v := range row[1:] {
		if float64(v) > maxVal {
			maxVal = float64(v)
		}
	}
	var sum float64
	for i, v := range row {
		ev := math.Exp(float64(v) - maxVal)
		row[i] = float32(ev)
		sum += ev
	}
	for i := range row {
		row[i] /= float32(sum)
	}
}

func maxAbsDiffTrain(a, b []float32) float64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var out float64
	for i := 0; i < n; i++ {
		if d := math.Abs(float64(a[i] - b[i])); d > out {
			out = d
		}
	}
	return out
}
