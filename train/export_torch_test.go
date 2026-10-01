package train

import (
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"testing"
)

func TestTorchStateMapping(t *testing.T) {
	shapes := []WeightShape{{Name: "conv", Shape: []int{2, 1, 2, 3}}, {Name: "bias", Shape: []int{2}}}
	w := [][]float32{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, {2, 3}}
	m := torchStateMap{Mappings: []torchStateMapping{{Source: "conv", Target: "conv.weight", Transform: "transpose", Axes: []int{0, 3, 1, 2}, Shape: []int{2, 3, 1, 2}}, {Source: "bias", Target: "conv.bias", Transform: "identity", Shape: []int{2}}}}
	got, err := mapTorchState(m, shapes, w)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got[0].Data, []float32{0, 3, 1, 4, 2, 5, 6, 9, 7, 10, 8, 11}) {
		t.Fatal(got[0])
	}
	for _, kind := range []string{"missing", "duplicate-source", "duplicate-target", "axes", "rank", "shape", "code", "unknown", "reason", "exclude-conflict", "identity-axes"} {
		t.Run(kind, func(t *testing.T) {
			raw, _ := json.Marshal(m)
			var bad torchStateMap
			_ = json.Unmarshal(raw, &bad)
			switch kind {
			case "missing":
				bad.Mappings = bad.Mappings[:1]
			case "duplicate-source":
				bad.Mappings[1].Source = "conv"
			case "duplicate-target":
				bad.Mappings[1].Target = "conv.weight"
			case "axes":
				bad.Mappings[0].Axes = []int{0, 0, 2, 3}
			case "rank":
				bad.Mappings[0].Axes = []int{0}
			case "shape":
				bad.Mappings[0].Shape = []int{12}
			case "code":
				bad.Mappings[0].Transform = "eval"
			case "unknown":
				bad.Mappings[0].Source = "typo"
			case "reason":
				bad.Mappings = bad.Mappings[:1]
				bad.Excluded = []torchStateExclusion{{Source: "bias"}}
			case "exclude-conflict":
				bad.Excluded = []torchStateExclusion{{Source: "bias", Reason: "duplicate"}}
			case "identity-axes":
				bad.Mappings[1].Axes = []int{0}
			}
			if _, err := mapTorchState(bad, shapes, w); err == nil {
				t.Fatal("accepted invalid map")
			}
		})
	}
	m.Mappings = m.Mappings[:1]
	m.Excluded = []torchStateExclusion{{Source: "bias", Reason: "consumer has no bias"}}
	if _, err = mapTorchState(m, shapes, w); err != nil {
		t.Fatal(err)
	}
}

func TestExportTorchStatePackage(t *testing.T) {
	cfgPath := "../examples/grid_regression_tiny.json"
	cfg, err := LoadArchConfig(cfgPath)
	if err != nil {
		t.Fatal(err)
	}
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	checkpoint := filepath.Join(dir, "weights.st")
	if err = exportSafetensors(checkpoint, cfg, shapes, initWeightData(shapes, 42, "", 0)); err != nil {
		t.Fatal(err)
	}
	m := torchStateMap{Format: "mixlab.torch_state_map.v1", Mappings: []torchStateMapping{
		{Source: shapes[0].Name, Target: "conv.weight", Transform: "transpose", Axes: []int{0, 3, 1, 2}, Shape: []int{1, 2, 3, 3}},
		{Source: shapes[1].Name, Target: "conv.bias", Transform: "identity", Shape: []int{1}},
	}}
	mapPath := filepath.Join(dir, "map.json")
	if err = atomicWriteJSON(mapPath, m); err != nil {
		t.Fatal(err)
	}
	opts := ExportTorchStateOptions{ConfigPath: cfgPath, SafetensorsLoad: checkpoint, MapPath: mapPath, OutputDir: filepath.Join(dir, "export")}
	if err = RunExportTorchState(opts); err != nil {
		t.Fatal(err)
	}
	for _, file := range []string{"model.safetensors", "export.json", "infer.py", "config.json", "export-map.json"} {
		if _, err = os.Stat(filepath.Join(opts.OutputDir, file)); err != nil {
			t.Fatal(err)
		}
	}
	tensors, err := loadSafetensors(filepath.Join(opts.OutputDir, "model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(tensors["conv.weight"].Shape, []int{1, 2, 3, 3}) || len(tensors) != 2 {
		t.Fatal(tensors)
	}
	blob, err := os.ReadFile(filepath.Join(opts.OutputDir, "model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	headerLen := binary.LittleEndian.Uint64(blob[:8])
	var header map[string]json.RawMessage
	if err = json.Unmarshal(blob[8:8+headerLen], &header); err != nil {
		t.Fatal(err)
	}
	var entries []safetensorHeaderEntry
	for name, raw := range header {
		if name == "__metadata__" {
			continue
		}
		var e safetensorHeaderEntry
		if err = json.Unmarshal(raw, &e); err != nil {
			t.Fatal(err)
		}
		entries = append(entries, e)
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].DataOffsets[0] < entries[j].DataOffsets[0] })
	var end uint64
	for _, e := range entries {
		if e.DataOffsets[0] != end {
			t.Fatal("standard export has payload gap")
		}
		end = e.DataOffsets[1]
	}
	if end != uint64(len(blob))-8-headerLen {
		t.Fatal("standard export has trailing padding")
	}
	if err = RunExportTorchState(opts); err == nil {
		t.Fatal("overwrote export")
	}
	if err = RunExportTorchState(ExportTorchStateOptions{}); err == nil {
		t.Fatal("accepted missing flags")
	}
	for _, raw := range []string{`{"format":"bad"}`, `{"format":"mixlab.torch_state_map.v1","code":"evil"}`, `{"format":"mixlab.torch_state_map.v1"} {}`} {
		if err = os.WriteFile(mapPath, []byte(raw), 0600); err != nil {
			t.Fatal(err)
		}
		if _, err = loadTorchStateMap(mapPath); err == nil {
			t.Fatal("accepted invalid JSON map")
		}
	}
}
