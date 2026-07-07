package train

import (
	"math"
	"os"
	"path/filepath"
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
		for _, temperature := range []float64{1.0, 2.0} {
			t.Run(strategy, func(t *testing.T) {
				got := make([]float32, len(teachers[0]))
				for _, logits := range teachers {
					accumulateTeacherLogits(got, logits, strategy, vocab, temperature)
				}
				finalizeTeacherProbs(got, len(teachers), vocab, strategy, temperature)
				want := distillationEnsembleOracle(teachers, strategy, vocab, temperature)
				if diff := maxAbsDiffTrain(got, want); diff > 1e-6 {
					t.Fatalf("%s T=%g max diff=%g\ngot=%v\nwant=%v", strategy, temperature, diff, got, want)
				}
			})
		}
	}
}

func TestDistillationKLZeroDoesNotInstantiateTeachers(t *testing.T) {
	cfg := &ArchConfig{
		VocabSize: 16,
		SeqLen:    4,
		Training: TrainingSpec{
			BatchTokens: 8,
			Distillation: &DistillationSpec{
				LossWeightKL: 0,
				LossWeightCE: 1,
			},
		},
	}
	ensemble, err := newDistillationEnsemble(cfg)
	if err != nil {
		t.Fatalf("newDistillationEnsemble: %v", err)
	}
	if ensemble != nil {
		t.Fatalf("newDistillationEnsemble returned %+v, want nil", ensemble)
	}
}

func TestAttachDistillationSkipsHybridCausalAndZeroMaskedRows(t *testing.T) {
	ensemble := &distillationEnsemble{
		objective: arch.ObjectiveMNTP,
	}
	causalBatch := objectiveBatch{
		x: []int{1, 2, 3, 4},
		y: []int{2, 3, 4, 5},
	}
	got, err := attachDistillationTeacherProbs(ensemble, causalBatch, 1, 4)
	if err != nil {
		t.Fatalf("attachDistillationTeacherProbs causal skip: %v", err)
	}
	if got.teacherProbs != nil {
		t.Fatal("hybrid causal skip attached teacher_probs")
	}

	zeroMaskedBatch := objectiveBatch{
		x:        []int{1, 2, 3, 4},
		y:        []int{2, 3, 4, 5},
		lossMask: []float32{0, 0, 0, 0},
	}
	got, err = attachDistillationTeacherProbs(ensemble, zeroMaskedBatch, 1, 4)
	if err != nil {
		t.Fatalf("attachDistillationTeacherProbs zero-mask skip: %v", err)
	}
	if got.teacherProbs != nil {
		t.Fatal("zero masked rows attached teacher_probs")
	}
}

func TestDistillationTokenizerHashMismatch(t *testing.T) {
	dir := t.TempDir()
	studentDir := filepath.Join(dir, "student")
	teacherDir := filepath.Join(dir, "teacher")
	if err := os.MkdirAll(studentDir, 0o755); err != nil {
		t.Fatalf("MkdirAll student: %v", err)
	}
	if err := os.MkdirAll(teacherDir, 0o755); err != nil {
		t.Fatalf("MkdirAll teacher: %v", err)
	}
	studentCfg := filepath.Join(studentDir, "config.json")
	teacherCfg := filepath.Join(teacherDir, "config.json")
	if err := os.WriteFile(studentCfg, []byte("{}"), 0o644); err != nil {
		t.Fatalf("WriteFile student config: %v", err)
	}
	if err := os.WriteFile(teacherCfg, []byte("{}"), 0o644); err != nil {
		t.Fatalf("WriteFile teacher config: %v", err)
	}
	if err := os.WriteFile(filepath.Join(studentDir, "tokenizer.json"), []byte(`{"model":{"vocab":{"a":0}}}`), 0o644); err != nil {
		t.Fatalf("WriteFile student tokenizer: %v", err)
	}
	if err := os.WriteFile(filepath.Join(teacherDir, "tokenizer.json"), []byte(`{"model":{"vocab":{"b":0}}}`), 0o644); err != nil {
		t.Fatalf("WriteFile teacher tokenizer: %v", err)
	}
	if err := validateDistillationTokenizerMatch(studentCfg, "", teacherCfg, ""); err == nil {
		t.Fatal("expected tokenizer mismatch error")
	}
	if err := os.WriteFile(filepath.Join(teacherDir, "tokenizer.json"), []byte(`{"model":{"vocab":{"a":0}}}`), 0o644); err != nil {
		t.Fatalf("Rewrite teacher tokenizer: %v", err)
	}
	if err := validateDistillationTokenizerMatch(studentCfg, "", teacherCfg, ""); err != nil {
		t.Fatalf("matching tokenizer hashes rejected: %v", err)
	}
}

func distillationEnsembleOracle(teachers [][]float32, strategy string, vocab int, temperature float64) []float32 {
	out := make([]float32, len(teachers[0]))
	rows := len(out) / vocab
	for row := 0; row < rows; row++ {
		start := row * vocab
		switch strategy {
		case arch.DistillationMeanLogProbs:
			for col := 0; col < vocab; col++ {
				var logProbSum float64
				for _, logits := range teachers {
					logProbSum += logSoftmaxAt(logits[start:start+vocab], col, temperature)
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
				avg[col] = float32(sum / float64(len(teachers)) / temperature)
			}
			softmaxFloat32(avg)
			copy(out[start:start+vocab], avg)
		}
	}
	return out
}

func logSoftmaxAt(row []float32, col int, temperature float64) float64 {
	maxVal := float64(row[0]) / temperature
	for _, v := range row[1:] {
		if scaled := float64(v) / temperature; scaled > maxVal {
			maxVal = scaled
		}
	}
	var sum float64
	for _, v := range row {
		sum += math.Exp(float64(v)/temperature - maxVal)
	}
	return float64(row[col])/temperature - maxVal - math.Log(sum)
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
