package train

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestTrainingProgressFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "progress.json")
	t.Setenv("MIXLAB_PROGRESS_FILE", path)
	p, err := newTrainingProgressFile()
	if err != nil {
		t.Fatal(err)
	}
	for _, update := range []struct{ step, committed int }{{-1, 0}, {53025, 53026}, {53026, 53026}, {53027, 53027}} {
		if err := p.update(update.step, uint64(update.committed)); err != nil {
			t.Fatal(err)
		}
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		var got trainingProgressRecord
		if err := json.Unmarshal(b, &got); err != nil {
			t.Fatal(err)
		}
		if got.PID != os.Getpid() || got.Step != update.step || got.OptimizerSteps != uint64(update.committed) {
			t.Fatalf("wrong progress: %+v", got)
		}
	}
	files, err := os.ReadDir(filepath.Dir(path))
	if err != nil || len(files) != 1 {
		t.Fatalf("temporary files leaked: %v %v", files, err)
	}
}

func TestTrainingProgressFileDisabledAndErrors(t *testing.T) {
	t.Setenv("MIXLAB_PROGRESS_FILE", "")
	p, err := newTrainingProgressFile()
	if err != nil || p != nil || p.update(1, 1) != nil {
		t.Fatal("disabled progress must be a no-op")
	}
	t.Setenv("MIXLAB_PROGRESS_FILE", filepath.Join(t.TempDir(), "missing", "progress"))
	if _, err := newTrainingProgressFile(); err == nil {
		t.Fatal("missing progress directory must fail clearly")
	}
}
