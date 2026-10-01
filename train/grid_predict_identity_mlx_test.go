//go:build mlx && cgo && (darwin || linux)

package train

import (
	"math"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/data"
)

func TestGridPredictionLogicalCheckpointIdentity(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX unavailable")
	}
	cfg, err := LoadArchConfig("../examples/grid_regression_tiny.json")
	if err != nil {
		t.Fatal(err)
	}
	ds, err := data.OpenGridDataset(prepareTinyGrid(t), "predict")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = ds.Close() }()
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatal(err)
	}
	weights := initWeightData(shapes, 4, "", 0)
	dir := t.TempDir()
	save := func(name string, specs []WeightShape, values [][]float32) string {
		t.Helper()
		path := filepath.Join(dir, name+".safetensors")
		if err := exportSafetensors(path, cfg, specs, values); err != nil {
			t.Fatal(err)
		}
		return path
	}
	predict := func(path string) ([]float32, error) {
		var result []float32
		err := visitGridPredictions(cfg, path, ds, func(b data.GridBatch, values []float32) error {
			n := b.Count * b.Geometry.Height * b.Geometry.Width * cfg.DenseRegression.TargetChannels
			result = append(result, values[:n]...)
			return nil
		})
		return result, err
	}
	t.Run("reordered", func(t *testing.T) {
		want, err := predict(save("original", shapes, weights))
		if err != nil {
			t.Fatal(err)
		}
		got, err := predict(save("reordered", []WeightShape{shapes[1], shapes[0]}, [][]float32{weights[1], weights[0]}))
		if err != nil || !reflect.DeepEqual(got, want) {
			t.Fatalf("reordered logical checkpoint prediction mismatch: %v", err)
		}
	})
	t.Run("unexpected", func(t *testing.T) {
		extra := append(append([]WeightShape(nil), shapes...), WeightShape{Name: "unexpected", Shape: []int{1}})
		values := append(append([][]float32(nil), weights...), []float32{0})
		if _, err := predict(save("unexpected", extra, values)); err == nil || !strings.Contains(err.Error(), "unexpected") {
			t.Fatalf("expected unexpected checkpoint tensor rejection, got %v", err)
		}
	})
	t.Run("nonfinite_unused", func(t *testing.T) {
		cfg.Blocks[0].Weights = append(cfg.Blocks[0].Weights, arch.WeightSpec{Name: "unused", Shape: []string{"1"}})
		extra, err := computeWeightShapes(cfg)
		if err != nil {
			t.Fatal(err)
		}
		values := append(append([][]float32(nil), weights...), []float32{float32(math.NaN())})
		if _, err := predict(save("nonfinite", extra, values)); err == nil || !strings.Contains(err.Error(), "logical weight") || !strings.Contains(err.Error(), "non-finite") {
			t.Fatalf("expected pre-upload non-finite logical weight rejection, got %v", err)
		}
	})
}
