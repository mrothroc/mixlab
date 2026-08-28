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
	if !strings.Contains(warning, "sampled 10 training batches each contained 1 distinct label (of 35)") ||
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

func TestClassificationLabelDiversityUsesRealRows(t *testing.T) {
	monitor := newClassificationLabelDiversityMonitor(3, 1)
	warning := monitor.observe(trainBatch{
		labels:       []int32{2, 0, 2, 1},
		batchSize:    4,
		exampleCount: 2,
		exampleMask:  []float32{1, 0, 1, 0},
	})
	if !strings.Contains(warning, "1 distinct label (of 3)") {
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
	frames := make([]float32, 32*4)
	for i := range frames {
		frames[i] = float32(i)
	}
	writeNPYFloat32(t, input, []int{32, 4, 1}, frames)

	prepare := func(name string, labelForRow func(int) int) string {
		t.Helper()
		labelsPath := filepath.Join(dir, name+".tsv")
		var labels strings.Builder
		for row := 0; row < 32; row++ {
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
			"--tokens-per-shard", "32",
		)
		if combined, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("prepare %s labels: %v\n%s", name, err, combined)
		}
		return filepath.Join(output, "train_*.bin")
	}

	warningFor := func(pattern string) string {
		t.Helper()
		loader, err := data.NewLoader(pattern, 7)
		if err != nil {
			t.Fatal(err)
		}
		monitor := newClassificationLabelDiversityMonitor(2, 10)
		var warning string
		for i := 0; i < 10; i++ {
			batch, err := loader.NextBatchDetailed(16, 4)
			if err != nil {
				t.Fatal(err)
			}
			warning = monitor.observe(trainBatchFromDataBatch(batch, nil))
		}
		return warning
	}

	ordered := prepare("class-ordered", func(row int) int {
		if row < 16 {
			return 0
		}
		return 1
	})
	if warning := warningFor(ordered); !strings.Contains(warning, "records may be class-ordered within shards") {
		t.Fatalf("class-ordered warning=%q", warning)
	}

	mixed := prepare("mixed", func(row int) int { return row % 2 })
	if warning := warningFor(mixed); warning != "" {
		t.Fatalf("mixed prepared data warning=%q", warning)
	}
}
