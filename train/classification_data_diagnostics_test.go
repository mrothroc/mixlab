package train

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/data"
)

func TestClassificationLabelDiversityWarnsForClassPureBatches(t *testing.T) {
	monitor := newClassificationLabelDiversityMonitor(35, 20)
	for batch := 0; batch < defaultClassificationLabelDiversityBatches-1; batch++ {
		labels := []int32{int32(batch), int32(batch), int32(batch), int32(batch)}
		if warning := monitor.observe(trainBatch{labels: labels}); warning != "" {
			t.Fatalf("early warning after batch %d: %q", batch+1, warning)
		}
	}
	warning := monitor.observe(trainBatch{labels: []int32{9, 9, 9, 9}})
	if !strings.Contains(warning, "sampled 10 training batches averaged 1.0 distinct labels") ||
		!strings.Contains(warning, "of 4 possible per batch, 35 classes; min=1 max=1") ||
		!strings.Contains(warning, "shuffle records and labels together before prepare") {
		t.Fatalf("warning=%q", warning)
	}
	if repeated := monitor.observe(trainBatch{labels: []int32{10, 10, 10, 10}}); repeated != "" {
		t.Fatalf("warning repeated: %q", repeated)
	}
}

func TestClassificationLabelDiversityDoesNotWarnForMixedBatches(t *testing.T) {
	monitor := newClassificationLabelDiversityMonitor(4, defaultClassificationLabelDiversityBatches)
	for batch := 0; batch < defaultClassificationLabelDiversityBatches; batch++ {
		labels := []int32{0, 1, 2, 3}
		if warning := monitor.observe(trainBatch{labels: labels}); warning != "" {
			t.Fatalf("mixed labels warning after batch %d: %q", batch+1, warning)
		}
	}
}

func TestClassificationLabelDiversityWarnsAcrossShardBoundaryBatches(t *testing.T) {
	monitor := newClassificationLabelDiversityMonitor(35, 10)
	distinctPerBatch := []int{1, 1, 1, 2, 1, 1, 1, 3, 1, 1}
	var warning string
	for _, distinct := range distinctPerBatch {
		labels := make([]int32, 16)
		for row := range labels {
			labels[row] = int32(row % distinct)
		}
		warning = monitor.observe(trainBatch{labels: labels})
	}
	if !strings.Contains(warning, "averaged 1.3 distinct labels") ||
		!strings.Contains(warning, "of 16 possible per batch, 35 classes; min=1 max=3") {
		t.Fatalf("boundary-batch warning=%q", warning)
	}
}

// The threshold is a quarter of the diversity *above one*, since every batch
// necessarily contains at least one label. With 16 achievable it sits at 4.75.
func TestClassificationLabelDiversityQuarterThresholdBoundary(t *testing.T) {
	observeUniform := func(distinctPerBatch int) string {
		monitor := newClassificationLabelDiversityMonitor(35, 10)
		var warning string
		for batch := 0; batch < 10; batch++ {
			labels := make([]int32, 16)
			for row := range labels {
				labels[row] = int32(row % distinctPerBatch)
			}
			warning = monitor.observe(trainBatch{labels: labels})
		}
		return warning
	}
	if warning := observeUniform(4); !strings.Contains(warning, "averaged 4.0 distinct labels") {
		t.Fatalf("below threshold should warn: %q", warning)
	}
	if warning := observeUniform(5); warning != "" {
		t.Fatalf("at threshold should stay quiet: %q", warning)
	}
}

// A correctly shuffled but imbalanced corpus produces some class-pure batches
// purely by chance. Demanding every binary batch hold both labels made the
// diagnostic fire on 86% of well-prepared 5%-minority runs, so the threshold
// must tolerate the occasional pure batch.
func TestClassificationLabelDiversityDoesNotWarnForImbalancedShuffledData(t *testing.T) {
	monitor := newClassificationLabelDiversityMonitor(2, 10)
	// 5% minority over 32 rows leaves roughly a fifth of batches class-pure.
	pureBatches := map[int]bool{2: true, 6: true}
	var warning string
	for batch := 0; batch < 10; batch++ {
		labels := make([]int32, 32)
		if !pureBatches[batch] {
			labels[batch] = 1
		}
		warning = monitor.observe(trainBatch{labels: labels})
	}
	if warning != "" {
		t.Fatalf("shuffled imbalanced data warned: %q", warning)
	}
}

func TestClassificationLabelDiversityUsesRealRows(t *testing.T) {
	monitor := newClassificationLabelDiversityMonitor(3, 1)
	warning := monitor.observe(trainBatch{
		labels:       []int32{2, 0, 2, 1},
		batchSize:    4,
		exampleCount: 2,
		exampleMask:  []float32{1, 0, 1, 0},
	})
	if !strings.Contains(warning, "averaged 1.0 distinct labels") ||
		!strings.Contains(warning, "of 2 possible per batch, 3 classes") {
		t.Fatalf("warning=%q", warning)
	}
}

func TestClassificationLabelDiversitySkipsSingleRowBatches(t *testing.T) {
	monitor := newClassificationLabelDiversityMonitor(3, 1)
	if warning := monitor.observe(trainBatch{labels: []int32{1}}); warning != "" {
		t.Fatalf("single-row warning=%q", warning)
	}
}

func TestClassificationLabelDiversityDetectsPreparedRecordOrder(t *testing.T) {
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not found")
	}
	if err := exec.Command("python3", "-c", "import numpy").Run(); err != nil {
		t.Skip("python3 numpy library not available")
	}
	scriptPath := filepath.Join("..", "scripts", "prepare.py")
	if _, err := os.Stat(scriptPath); err != nil {
		scriptPath = filepath.Join("scripts", "prepare.py")
	}

	dir := t.TempDir()
	input := filepath.Join(dir, "features.npy")
	const (
		numLabels       = 35
		recordsPerLabel = 62
		recordCount     = numLabels * recordsPerLabel
	)
	frames := make([]float32, recordCount*4)
	for i := range frames {
		frames[i] = float32(i)
	}
	writeNPYFloat32(t, input, []int{recordCount, 4, 1}, frames)

	prepare := func(name string, labelForRow func(int) int) string {
		t.Helper()
		labelsPath := filepath.Join(dir, name+".tsv")
		var labels strings.Builder
		for row := 0; row < recordCount; row++ {
			fmt.Fprintf(&labels, "%d\t%d\n", row, labelForRow(row))
		}
		if err := os.WriteFile(labelsPath, []byte(labels.String()), 0o600); err != nil {
			t.Fatal(err)
		}
		output := filepath.Join(dir, name)
		cmd := exec.Command(
			"python3", scriptPath,
			"--input", input,
			"--output", output,
			"--input-format", "continuous",
			"--label-file", labelsPath,
			"--val-split", "0",
			"--tokens-per-shard", "248",
		)
		if combined, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("prepare %s labels: %v\n%s", name, err, combined)
		}
		return filepath.Join(output, "train_*.bin")
	}

	warningFor := func(pattern string, numLabels int) string {
		t.Helper()
		loader, err := data.NewLoader(pattern, 7)
		if err != nil {
			t.Fatal(err)
		}
		monitor := newClassificationLabelDiversityMonitor(numLabels, 10)
		var warning string
		for i := 0; i < 10; i++ {
			batch, err := loader.NextBatchDetailed(64, 4)
			if err != nil {
				t.Fatal(err)
			}
			warning = monitor.observe(trainBatchFromDataBatch(batch, nil))
		}
		return warning
	}

	ordered := prepare("class-ordered", func(row int) int { return row / recordsPerLabel })
	if warning := warningFor(ordered, numLabels); !strings.Contains(warning, "records may be class-ordered within shards") {
		t.Fatalf("class-ordered warning=%q", warning)
	}

	mixed := prepare("mixed", func(row int) int { return row % numLabels })
	if warning := warningFor(mixed, numLabels); warning != "" {
		t.Fatalf("mixed prepared data warning=%q", warning)
	}
}
