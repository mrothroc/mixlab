package train

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/data"
)

func prepareTinyGridVectors(t *testing.T) string {
	t.Helper()
	if err := exec.Command("python3", "-c", "import numpy").Run(); err != nil {
		t.Skip("numpy required")
	}
	dir := t.TempDir()
	for name, values := range map[string][]float32{"x": {1, 2, 3}, "y": {1.25, 2.25, 3.25}, "m": {1, 1, 1}, "v": {1.125, 2.125, 3.125}} {
		shape := []int{3, 1, 1, 1}
		if name == "v" {
			shape = []int{3, 1}
		}
		writeNPYFloat32(t, filepath.Join(dir, name+".npy"), shape, values)
	}
	entry := map[string]string{"inputs": "x.npy", "targets": "y.npy", "masks": "m.npy", "vectors": "v.npy"}
	spec := map[string]any{"dtype": "float16", "records_per_shard": 2, "splits": map[string]any{"train": entry, "val": entry, "predict": entry}}
	source := filepath.Join(dir, "source.json")
	if err := atomicWriteJSON(source, spec); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(dir, "prepared")
	if err := runPrepare(PrepareOptions{Input: source, Output: out, InputFormat: "grid"}); err != nil {
		t.Fatal(err)
	}
	return filepath.Join(out, data.DatasetManifestFilename)
}

func TestGridVectorsPrepareInputsAndIdentity(t *testing.T) {
	manifest := prepareTinyGridVectors(t)
	cfg, err := LoadArchConfig("../examples/grid_vectors_tiny.json")
	if err != nil {
		t.Fatal(err)
	}
	for _, split := range []string{"train", "val", "predict"} {
		ds, err := data.OpenGridDataset(manifest, split)
		if err != nil {
			t.Fatal(err)
		}
		if err = checkGridGeometry(cfg, ds.Geometry, true); err != nil {
			t.Fatal(err)
		}
		b, err := ds.ReadBatch([]int{2}, 2)
		_ = ds.Close()
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(b.Vectors, []float32{3.125, 3.125}) || !reflect.DeepEqual(b.LossMask, []float32{1, 0}) {
			t.Fatal(b)
		}
		for _, supervised := range []bool{true, false} {
			p, err := arch.BuildGridIRProgram(cfg, supervised)
			if err != nil {
				t.Fatal(err)
			}
			inputs, err := makeGridInputs(p.Inputs, &b)
			if err != nil {
				t.Fatal(err)
			}
			found := false
			for _, in := range inputs {
				if in.Name == "grid_vectors" {
					found = true
					if !reflect.DeepEqual(in.Data, b.Vectors) {
						t.Fatal(in)
					}
				}
			}
			if !found {
				t.Fatal("not bound")
			}
			bad := b
			bad.Vectors = bad.Vectors[:1]
			if _, err = makeGridInputs(p.Inputs, &bad); err == nil {
				t.Fatal("accepted wrong vector batch")
			}
		}
		g := b.Geometry
		g.VectorDim++
		if err = checkGridGeometry(cfg, g, true); err == nil {
			t.Fatal("accepted mismatched vectors")
		}
	}
	before, err := data.GridDatasetIdentity(manifest)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(filepath.Dir(manifest), "train_00000.grid")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	raw[len(raw)-1] ^= 1 // Change only a vector byte, preserving maps and IDs.
	if err = os.WriteFile(path, raw, 0600); err != nil {
		t.Fatal(err)
	}
	after, err := data.GridDatasetIdentity(manifest)
	if err != nil || before == after {
		t.Fatal("vector mutation not bound to replay identity", err)
	}
}

func TestGridVectorManifestValidation(t *testing.T) {
	path := prepareTinyGridVectors(t)
	original, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"hash", "width", "source", "missing", "format", "geometry"} {
		t.Run(kind, func(t *testing.T) {
			var m data.DatasetManifest
			if err := json.Unmarshal(original, &m); err != nil {
				t.Fatal(err)
			}
			s := m.GridProvenance.Vectors["train"]
			switch kind {
			case "hash":
				s.SHA256 = "bad"
			case "width":
				s.Width++
			case "source":
				s.Name = " "
			case "format":
				m.ShardFormat = data.GridShardFormat
			case "geometry":
				m.Grid.VectorDim++
			}
			m.GridProvenance.Vectors["train"] = s
			if kind == "missing" {
				delete(m.GridProvenance.Vectors, "val")
			}
			if err := atomicWriteJSON(path, m); err != nil {
				t.Fatal(err)
			}
			if _, err := data.OpenGridDataset(path, "train"); err == nil {
				t.Fatal("accepted invalid vector manifest")
			}
		})
	}
	if err := os.WriteFile(path, original, 0600); err != nil {
		t.Fatal(err)
	}
	// Valid metadata but incompatible shard width must also fail.
	var wider data.DatasetManifest
	if err := json.Unmarshal(original, &wider); err != nil {
		t.Fatal(err)
	}
	wider.Grid.VectorDim = 2
	for split, source := range wider.GridProvenance.Vectors {
		source.Width = 2
		wider.GridProvenance.Vectors[split] = source
	}
	if err := atomicWriteJSON(path, wider); err != nil {
		t.Fatal(err)
	}
	if _, err := data.OpenGridDataset(path, "train"); err == nil {
		t.Fatal("accepted shard width mismatch")
	}
}
