package train

import (
	"os"
	"reflect"
	"strings"
	"testing"
)

func TestGridBestArtifactSurvivesCheckpointRewind(t *testing.T) {
	cfg, err := LoadArchConfig("../examples/grid_regression_tiny.json")
	if err != nil {
		t.Fatal(err)
	}
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatal(err)
	}
	w := initWeightData(shapes, 42, "", 0)
	dir := t.TempDir()
	b, err := newGridBestArtifact(cfg, dir, "dataset", false)
	if err != nil {
		t.Fatal(err)
	}
	if err = b.save(cfg, fakeWeightReader{w}, shapes, .2); err != nil {
		t.Fatal(err)
	}
	before, err := os.ReadFile(b.path)
	if err != nil {
		t.Fatal(err)
	}
	// The resumed checkpoint had best=1. Replayed validation at .5 improves
	// its logical metric state, but must not overwrite the newer .2 artifact.
	b, err = newGridBestArtifact(cfg, dir, "dataset", true)
	if err != nil {
		t.Fatal(err)
	}
	w[0][0] += 1
	if err = b.save(cfg, fakeWeightReader{w}, shapes, .5); err != nil {
		t.Fatal(err)
	}
	after, err := os.ReadFile(b.path)
	if err != nil || !reflect.DeepEqual(before, after) {
		t.Fatal("checkpoint rewind replaced better artifact", err)
	}
	if err = b.save(cfg, fakeWeightReader{w}, shapes, .1); err != nil {
		t.Fatal(err)
	}
	restored, err := newGridBestArtifact(cfg, dir, "dataset", true)
	if err != nil || restored.score != .1 {
		t.Fatal(restored, err)
	}
	got, err := loadSafetensorsWeights(b.path, shapes)
	if err != nil || !reflect.DeepEqual(got, w) {
		t.Fatal("score and weights not published together", err)
	}
	if _, err = newGridBestArtifact(cfg, dir, "other-dataset", true); err == nil {
		t.Fatal("accepted unrelated best artifact")
	}
	if err = exportSafetensors(b.path, cfg, shapes, w); err != nil {
		t.Fatal(err)
	}
	if _, err = newGridBestArtifact(cfg, dir, "dataset", true); err == nil || !strings.Contains(err.Error(), "selection metadata") {
		t.Fatal("accepted ambiguous legacy best artifact", err)
	}
	if _, err = newGridBestArtifact(cfg, t.TempDir(), "dataset", true); err != nil {
		t.Fatal("resume into fresh directory", err)
	}
}
