package train

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/gpu"
	"github.com/mrothroc/mixlab/internal/gridpoints"
)

func TestGridPointsRunProvenanceAndResumeIdentity(t *testing.T) {
	c, err := LoadArchConfig("../examples/grid_polar_tiny.json")
	if err != nil {
		t.Fatal(err)
	}
	c.Training.AllowTF32 = nil // provenance/hash tests also run without a GPU backend
	base, err := resumeConfigHash(c)
	if err != nil {
		t.Fatal(err)
	}
	for _, points := range []gridpoints.Columns{nil, {{0, 1}}} {
		c.InputAdapter.VectorPoints = points
		dir := t.TempDir()
		if err = startTrainingPrecision(c, TrainOptions{CheckpointDir: dir}); err != nil {
			t.Fatal(err)
		}
		raw, err := os.ReadFile(filepath.Join(dir, "run.json"))
		if err != nil {
			t.Fatal(err)
		}
		var m map[string]json.RawMessage
		if err = json.Unmarshal(raw, &m); err != nil {
			t.Fatal(err)
		}
		if points == nil {
			if _, ok := m["vector_points"]; ok {
				t.Fatal("omission changed manifest")
			}
			continue
		}
		var actual gridpoints.Columns
		if err = json.Unmarshal(m["vector_points"], &actual); err != nil || !reflect.DeepEqual(points, actual) {
			t.Fatal(actual, err)
		}
	}
	hash, err := resumeConfigHash(c)
	if err != nil || hash == base {
		t.Fatal("points absent from resume identity", err)
	}
	dir := t.TempDir()
	for _, name := range []string{"weights.st", "state.st"} {
		if err = os.WriteFile(filepath.Join(dir, name), []byte("must not be read"), 0600); err != nil {
			t.Fatal(err)
		}
	}
	precision, err := trainingPrecision(c)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, resumeManifestFilename(1))
	m := resumeManifest{Format: resumeCheckpointFormat, GlobalStep: 1, ModelFile: "weights.st", StateFile: "state.st", ConfigHash: hash, Precision: &precision, Optimizer: gpu.TrainerOptimizerStats{AttemptedSteps: 1, CommittedSteps: 1}}
	if err = atomicWriteJSON(path, m); err != nil {
		t.Fatal(err)
	}
	if err = startTrainingPrecision(c, TrainOptions{CheckpointDir: dir}); err != nil {
		t.Fatal(err)
	}
	manifestPath := filepath.Join(dir, "run.json")
	original, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatal(err)
	}
	for _, changed := range []gridpoints.Columns{nil, {{1, 0}}} {
		c.InputAdapter.VectorPoints = changed
		if _, err = prepareResumeRun(c, "unused", path, nil); err == nil || !strings.Contains(err.Error(), "config does not match") {
			t.Fatal("changed points accepted", err)
		}
		if err = startTrainingPrecision(c, TrainOptions{CheckpointDir: dir, Resume: path}); err == nil || !strings.Contains(err.Error(), "config does not match") {
			t.Fatal("changed points overwrote provenance", err)
		}
		preserved, err := os.ReadFile(manifestPath)
		if err != nil || string(preserved) != string(original) {
			t.Fatal("rejected resume changed run manifest", err)
		}
	}
}
