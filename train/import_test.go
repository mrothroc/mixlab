package train

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

// TestSafetensorsRoundTrip verifies that export followed by load produces
// identical weight data. This is a pure file-format test with no GPU dependency.
func TestSafetensorsRoundTrip(t *testing.T) {
	cfg := &ArchConfig{
		Name:      "roundtrip_test",
		ModelDim:  32,
		VocabSize: 64,
		SeqLen:    16,
		Blocks:    []BlockSpec{{Type: "plain", Heads: 2}},
		Training:  TrainingSpec{Steps: 1, LR: 1e-3, Seed: 42, BatchTokens: 16},
	}

	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatalf("computeWeightShapes: %v", err)
	}
	if len(shapes) == 0 {
		t.Fatal("no weight shapes computed")
	}

	// Generate deterministic weights
	original := initWeightData(shapes, 123)
	if len(original) != len(shapes) {
		t.Fatalf("initWeightData returned %d weights, expected %d", len(original), len(shapes))
	}

	// Export to safetensors
	dir := t.TempDir()
	path := filepath.Join(dir, "test_weights.safetensors")
	if err := exportSafetensors(path, cfg, shapes, original); err != nil {
		t.Fatalf("exportSafetensors: %v", err)
	}

	// Verify file exists and is non-empty
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat safetensors file: %v", err)
	}
	if info.Size() == 0 {
		t.Fatal("safetensors file is empty")
	}

	// Load back
	loaded, err := loadSafetensorsWeights(path, shapes)
	if err != nil {
		t.Fatalf("loadSafetensorsWeights: %v", err)
	}
	if len(loaded) != len(original) {
		t.Fatalf("loaded %d weights, expected %d", len(loaded), len(original))
	}

	// Compare values exactly
	for i := range original {
		if len(loaded[i]) != len(original[i]) {
			t.Fatalf("weight %d (%s): loaded %d values, expected %d", i, shapes[i].Name, len(loaded[i]), len(original[i]))
		}
		for j := range original[i] {
			if loaded[i][j] != original[i][j] {
				t.Errorf("weight %d (%s)[%d]: loaded=%v expected=%v", i, shapes[i].Name, j, loaded[i][j], original[i][j])
				break // one error per tensor is enough
			}
		}
	}
}

// TestLoadSafetensorsErrors verifies proper error handling for invalid inputs.
func TestLoadSafetensorsErrors(t *testing.T) {
	// Non-existent file
	_, err := loadSafetensors("/nonexistent/file.safetensors")
	if err == nil {
		t.Error("expected error for non-existent file")
	}

	// Empty file
	dir := t.TempDir()
	empty := filepath.Join(dir, "empty.safetensors")
	if err := os.WriteFile(empty, []byte{}, 0o644); err != nil {
		t.Fatal(err)
	}
	_, err = loadSafetensors(empty)
	if err == nil {
		t.Error("expected error for empty file")
	}

	// Too-small file
	small := filepath.Join(dir, "small.safetensors")
	if err := os.WriteFile(small, []byte{1, 2, 3}, 0o644); err != nil {
		t.Fatal(err)
	}
	_, err = loadSafetensors(small)
	if err == nil {
		t.Error("expected error for too-small file")
	}
}

// TestShapesEqual tests the shapesEqual helper.
func TestShapesEqual(t *testing.T) {
	tests := []struct {
		a, b []int
		want bool
	}{
		{nil, nil, true},
		{[]int{}, []int{}, true},
		{[]int{3, 4}, []int{3, 4}, true},
		{[]int{3, 4}, []int{3, 5}, false},
		{[]int{3}, []int{3, 4}, false},
	}
	for _, tt := range tests {
		got := shapesEqual(tt.a, tt.b)
		if got != tt.want {
			t.Errorf("shapesEqual(%v, %v) = %v, want %v", tt.a, tt.b, got, tt.want)
		}
	}
}

// TestShapeProduct tests the shapeProduct helper.
func TestShapeProduct(t *testing.T) {
	tests := []struct {
		shape []int
		want  int
	}{
		{[]int{3, 4}, 12},
		{[]int{2, 3, 4}, 24},
		{[]int{5}, 5},
		{[]int{}, 1},
	}
	for _, tt := range tests {
		got := shapeProduct(tt.shape)
		if got != tt.want {
			t.Errorf("shapeProduct(%v) = %d, want %d", tt.shape, got, tt.want)
		}
	}
}

// TestLoadSafetensorsWeightsMismatch verifies error when tensor name is missing.
func TestLoadSafetensorsWeightsMismatch(t *testing.T) {
	cfg := &ArchConfig{
		Name:      "mismatch_test",
		ModelDim:  16,
		VocabSize: 32,
		SeqLen:    8,
		Blocks:    []BlockSpec{{Type: "plain", Heads: 2}},
		Training:  TrainingSpec{Steps: 1, LR: 1e-3, Seed: 1, BatchTokens: 8},
	}

	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatalf("computeWeightShapes: %v", err)
	}
	original := initWeightData(shapes, 1)

	dir := t.TempDir()
	path := filepath.Join(dir, "weights.safetensors")
	if err := exportSafetensors(path, cfg, shapes, original); err != nil {
		t.Fatalf("exportSafetensors: %v", err)
	}

	// Try loading with different shapes (extra weight)
	badShapes := append([]WeightShape(nil), shapes...)
	badShapes = append(badShapes, WeightShape{Name: "extra", Shape: []int{16}})
	_, err = loadSafetensorsWeights(path, badShapes)
	if err == nil {
		t.Error("expected error when loading with mismatched shapes")
	}
}

// TestDecodeSafetensorFloat32_WrongDType verifies error for non-F32 dtype.
func TestDecodeSafetensorFloat32_WrongDType(t *testing.T) {
	tensors := map[string]safeTensorBlob{
		"test": {DType: "I8", Shape: []int{4}, Data: make([]byte, 4)},
	}
	_, err := decodeSafetensorFloat32("test", []int{4}, tensors)
	if err == nil {
		t.Error("expected error for wrong dtype")
	}
}

// TestDecodeSafetensorFloat32_WrongShape verifies error for shape mismatch.
func TestDecodeSafetensorFloat32_WrongShape(t *testing.T) {
	tensors := map[string]safeTensorBlob{
		"test": {DType: "F32", Shape: []int{4, 2}, Data: make([]byte, 32)},
	}
	_, err := decodeSafetensorFloat32("test", []int{8}, tensors)
	if err == nil {
		t.Error("expected error for shape mismatch")
	}
}

// TestDecodeSafetensorFloat32_EmptyShape verifies error for zero-product shape.
func TestDecodeSafetensorFloat32_EmptyShape(t *testing.T) {
	tensors := map[string]safeTensorBlob{
		"test": {DType: "F32", Shape: []int{0}, Data: make([]byte, 0)},
	}
	_, err := decodeSafetensorFloat32("test", []int{0}, tensors)
	if err == nil {
		t.Error("expected error for empty/zero shape")
	}
}

// TestDecodeSafetensorFloat32_MissingTensor verifies error for absent tensor name.
func TestDecodeSafetensorFloat32_MissingTensor(t *testing.T) {
	tensors := map[string]safeTensorBlob{}
	_, err := decodeSafetensorFloat32("missing", []int{4}, tensors)
	if err == nil {
		t.Error("expected error for missing tensor")
	}
}

// TestDecodeSafetensorFloat32_PayloadSizeMismatch verifies error when data length != shape*4.
func TestDecodeSafetensorFloat32_PayloadSizeMismatch(t *testing.T) {
	tensors := map[string]safeTensorBlob{
		"test": {DType: "F32", Shape: []int{4}, Data: make([]byte, 8)}, // want 16 bytes
	}
	_, err := decodeSafetensorFloat32("test", []int{4}, tensors)
	if err == nil {
		t.Error("expected error for payload size mismatch")
	}
}

// TestRoundTripNaNInf verifies that special float values survive round-trip.
func TestRoundTripNaNInf(t *testing.T) {
	cfg := &ArchConfig{
		Name:      "special_values",
		ModelDim:  4,
		VocabSize: 8,
		SeqLen:    4,
		Blocks:    []BlockSpec{{Type: "plain", Heads: 1}},
		Training:  TrainingSpec{Steps: 1, LR: 1e-3, Seed: 1, BatchTokens: 4},
	}

	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatalf("computeWeightShapes: %v", err)
	}

	weights := initWeightData(shapes, 1)
	// Inject special values into first weight
	if len(weights[0]) >= 4 {
		weights[0][0] = float32(math.Inf(1))
		weights[0][1] = float32(math.Inf(-1))
		weights[0][2] = float32(math.NaN())
		weights[0][3] = 0.0
	}

	dir := t.TempDir()
	path := filepath.Join(dir, "special.safetensors")
	if err := exportSafetensors(path, cfg, shapes, weights); err != nil {
		t.Fatalf("exportSafetensors: %v", err)
	}

	loaded, err := loadSafetensorsWeights(path, shapes)
	if err != nil {
		t.Fatalf("loadSafetensorsWeights: %v", err)
	}

	if len(loaded[0]) >= 4 {
		if !math.IsInf(float64(loaded[0][0]), 1) {
			t.Errorf("expected +Inf, got %v", loaded[0][0])
		}
		if !math.IsInf(float64(loaded[0][1]), -1) {
			t.Errorf("expected -Inf, got %v", loaded[0][1])
		}
		if !math.IsNaN(float64(loaded[0][2])) {
			t.Errorf("expected NaN, got %v", loaded[0][2])
		}
		if loaded[0][3] != 0.0 {
			t.Errorf("expected 0.0, got %v", loaded[0][3])
		}
	}
}
