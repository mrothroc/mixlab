package train

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/arch"
)

func TestGridWarmStartLogicalIdentity(t *testing.T) {
	cfg, err := LoadArchConfig("../examples/grid_regression_tiny.json")
	if err != nil {
		t.Fatal(err)
	}
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatal(err)
	}
	w := initWeightData(shapes, 4, "", 0)
	path := filepath.Join(t.TempDir(), "reordered.safetensors")
	// Physical indices deliberately disagree with the target config.
	if err = exportSafetensors(path, cfg, []WeightShape{shapes[1], shapes[0]}, [][]float32{w[1], w[0]}); err != nil {
		t.Fatal(err)
	}
	got, report, err := loadGridWeights(path, cfg, shapes, nil)
	if err != nil || !reflect.DeepEqual(got, w) || len(report.Loaded) != 2 {
		t.Fatalf("weights=%v report=%+v err=%v", got, report, err)
	}
	extended := append(append([]WeightShape(nil), shapes...), WeightShape{Name: "network.refinement.weight", Shape: []int{1}, InitOne: true})
	if _, _, err = loadGridWeights(path, cfg, extended, nil); err == nil || !strings.Contains(err.Error(), "missing logical") {
		t.Fatal(err)
	}
	got, report, err = loadGridWeights(path, cfg, extended, []string{"network.refinement.*"})
	if err != nil || len(report.New) != 1 || got[2][0] != 1 || !reflect.DeepEqual(got[:2], w) {
		t.Fatalf("%+v %v %v", report, got, err)
	}
	bad := append([]WeightShape(nil), shapes...)
	bad[0].Shape = []int{1, 1, 1, 18}
	if _, _, err = loadGridWeights(path, cfg, bad, []string{"*"}); err == nil || !strings.Contains(err.Error(), "shape mismatch") {
		t.Fatal(err)
	}
	if _, _, err = loadGridWeights(path, cfg, shapes[:1], nil); err == nil || !strings.Contains(err.Error(), "unexpected") {
		t.Fatal(err)
	}
	for _, value := range []float32{float32(math.NaN()), float32(math.Inf(1))} {
		w[0][0] = value
		if err = exportSafetensors(path, cfg, shapes, w); err != nil {
			t.Fatal(err)
		}
		if _, _, err = loadGridWeights(path, cfg, shapes, []string{"*"}); err == nil || !strings.Contains(err.Error(), "non-finite") {
			t.Fatalf("accepted non-finite warm start: %v", err)
		}
	}
}

func TestGridWarmStartLegacyAndMalformedIdentity(t *testing.T) {
	cfg, err := LoadArchConfig("../examples/grid_regression_tiny.json")
	if err != nil {
		t.Fatal(err)
	}
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatal(err)
	}
	w := initWeightData(shapes, 42, "", 0)
	path := filepath.Join(t.TempDir(), "weights.safetensors")
	tensors := []namedFloatTensor{{Name: "w0_network.conv.weight", Shape: shapes[0].Shape, Data: w[0]}, {Name: "w1_network.conv.bias", Shape: shapes[1].Shape, Data: w[1]}}
	if err = writeNamedFloatSafetensorsAtomic(path, tensors, nil); err != nil {
		t.Fatal(err)
	}
	if _, _, err = loadGridWeights(path, cfg, shapes, nil); err != nil {
		t.Fatal(err)
	}
	if _, _, err = loadGridWeights(path, cfg, shapes, []string{"*"}); err == nil || !strings.Contains(err.Error(), "legacy") {
		t.Fatal(err)
	}
	for _, identities := range []map[string]string{
		{"network.conv.weight": tensors[0].Name, "network.conv.bias": tensors[0].Name},
		{"network.conv.weight": tensors[0].Name},
		{"network.conv.weight": "missing", "network.conv.bias": tensors[1].Name},
	} {
		encoded, _ := json.Marshal(identities)
		if err = writeNamedFloatSafetensorsAtomic(path, tensors, map[string]string{"logical_weights": string(encoded), "task": arch.ObjectiveDenseRegression}); err != nil {
			t.Fatal(err)
		}
		if _, _, err = loadGridWeights(path, cfg, shapes, nil); err == nil {
			t.Fatal("accepted malformed identity", identities)
		}
	}
	tensors = append(tensors, namedFloatTensor{Name: "unexpected", Shape: []int{1}, Data: []float32{0}})
	if err = writeNamedFloatSafetensorsAtomic(path, tensors, nil); err != nil {
		t.Fatal(err)
	}
	if _, _, err = loadGridWeights(path, cfg, shapes, nil); err == nil || !strings.Contains(err.Error(), "unexpected") {
		t.Fatal(err)
	}
}

func TestGridWarmStartPathsAndResumeConflicts(t *testing.T) {
	cfg, err := LoadArchConfig("../examples/grid_regression_tiny.json")
	if err != nil {
		t.Fatal(err)
	}
	cfg.SourcePath = filepath.Join(t.TempDir(), "model.json")
	cfg.Training.InitFrom = "stage1.safetensors"
	got, err := gridWarmStartPath(cfg, TrainOptions{})
	if err != nil || got != filepath.Join(filepath.Dir(cfg.SourcePath), cfg.Training.InitFrom) {
		t.Fatal(got, err)
	}
	for _, opts := range []TrainOptions{{SafetensorsLoad: "cli"}, {Resume: "resume"}} {
		if _, err = gridWarmStartPath(cfg, opts); err == nil {
			t.Fatal("accepted conflict")
		}
	}
	before, _ := resumeConfigHash(cfg)
	cfg.Training.InitFrom = ""
	after, _ := resumeConfigHash(cfg)
	if before != after {
		t.Fatal("one-time warm start affects resume hash")
	}
	got, err = gridWarmStartPath(cfg, TrainOptions{SafetensorsLoad: "relative-cli"})
	if err != nil || got != "relative-cli" {
		t.Fatal(got, err)
	}
	cfg.Training.InitAllowMissing = []string{"*"}
	if _, err = gridWarmStartPath(cfg, TrainOptions{}); err == nil {
		t.Fatal("accepted missing warm start")
	}
	if _, err = gridWarmStartPath(cfg, TrainOptions{Resume: "resume"}); err == nil {
		t.Fatal("accepted missing patterns on resume")
	}
}

func TestGridConfigRelativeWarmStartFallbackPath(t *testing.T) {
	root := t.TempDir()
	if err := os.Mkdir(filepath.Join(root, "child"), 0700); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile("../examples/grid_regression_tiny.json")
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filepath.Join(root, "grid.json"), raw, 0600); err != nil {
		t.Fatal(err)
	}
	t.Chdir(filepath.Join(root, "child"))
	cfg, err := LoadArchConfig("grid.json")
	if err != nil {
		t.Fatal(err)
	}
	cfg.Training.InitFrom = "stage1.safetensors"
	got, err := gridWarmStartPath(cfg, TrainOptions{})
	if err != nil || got != filepath.Join("..", "stage1.safetensors") {
		t.Fatal(got, err)
	}
}
