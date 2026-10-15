package train

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/mrothroc/mixlab/artifact/tensors"
)

// The coordinator's pure F32 reader must accept real native checkpoint headers
// (including metadata and numerically ordered wN names), not only its own writer.
func TestCoordinatorF32NativeCheckpointCompatibility(t *testing.T) {
	cfg, err := ParseArchConfig([]byte(`{"model_dim":8,"vocab_size":16,"seq_len":4,"blocks":[{"type":"plain","heads":2},{"type":"swiglu"}]}`), "tensor_adapter")
	if err != nil {
		t.Fatal(err)
	}
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatal(err)
	}
	weights := initWeightData(shapes, 19, cfg.Training.WeightInit, cfg.Training.WeightInitStd)
	path := filepath.Join(t.TempDir(), "native.safetensors")
	if err = exportSafetensors(path, cfg, shapes, weights); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = f.Close() }()
	info, err := f.Stat()
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := tensors.Read(f, uint64(info.Size()))
	if err != nil {
		t.Fatal(err)
	}
	if len(decoded) != len(weights) {
		t.Fatal("weight count mismatch")
	}
	for i, v := range decoded {
		if !reflect.DeepEqual(v.Shape, shapes[i].Shape) || !reflect.DeepEqual(v.Values, weights[i]) {
			t.Fatal("native payload order changed", i)
		}
	}
	out := filepath.Join(t.TempDir(), "coordinator.safetensors")
	w, err := os.Create(out)
	if err != nil {
		t.Fatal(err)
	}
	err = tensors.Write(w, decoded)
	closeErr := w.Close()
	if err != nil || closeErr != nil {
		t.Fatal(err, closeErr)
	}
	reloaded, err := loadSafetensorsWeights(out, shapes)
	if err != nil || !reflect.DeepEqual(weights, reloaded) {
		t.Fatal("native import mismatch", err)
	}
}
