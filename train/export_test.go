package train

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// TestEncodeFloat32Scalar verifies encoding a single float32 value.
func TestEncodeFloat32Scalar(t *testing.T) {
	tests := []struct {
		name string
		val  float32
	}{
		{"zero", 0.0},
		{"positive", 3.14},
		{"negative", -2.718},
		{"one", 1.0},
		{"small", 1e-30},
		{"large", 1e30},
		{"neg_zero", float32(math.Copysign(0, -1))},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := encodeFloat32Scalar(tt.val)
			if len(got) != 4 {
				t.Fatalf("expected 4 bytes, got %d", len(got))
			}
			// Decode back and verify.
			bits := binary.LittleEndian.Uint32(got)
			decoded := math.Float32frombits(bits)
			if decoded != tt.val {
				t.Errorf("round-trip mismatch: encoded %v, decoded %v", tt.val, decoded)
			}
		})
	}
}

// TestEncodeFloat32Scalar_SpecialValues verifies NaN and Inf encoding.
func TestEncodeFloat32Scalar_SpecialValues(t *testing.T) {
	// +Inf
	got := encodeFloat32Scalar(float32(math.Inf(1)))
	bits := binary.LittleEndian.Uint32(got)
	decoded := math.Float32frombits(bits)
	if !math.IsInf(float64(decoded), 1) {
		t.Errorf("expected +Inf, got %v", decoded)
	}

	// -Inf
	got = encodeFloat32Scalar(float32(math.Inf(-1)))
	bits = binary.LittleEndian.Uint32(got)
	decoded = math.Float32frombits(bits)
	if !math.IsInf(float64(decoded), -1) {
		t.Errorf("expected -Inf, got %v", decoded)
	}

	// NaN
	got = encodeFloat32Scalar(float32(math.NaN()))
	bits = binary.LittleEndian.Uint32(got)
	decoded = math.Float32frombits(bits)
	if !math.IsNaN(float64(decoded)) {
		t.Errorf("expected NaN, got %v", decoded)
	}
}

// TestEncodeFloat32Data verifies encoding a slice of float32 values.
func TestEncodeFloat32Data(t *testing.T) {
	vals := []float32{1.0, -2.0, 3.5, 0.0}
	got := encodeFloat32Data(vals)
	if len(got) != 16 {
		t.Fatalf("expected 16 bytes, got %d", len(got))
	}
	for i, v := range vals {
		bits := binary.LittleEndian.Uint32(got[i*4:])
		decoded := math.Float32frombits(bits)
		if decoded != v {
			t.Errorf("index %d: encoded %v, decoded %v", i, v, decoded)
		}
	}
}

func TestSafetensorsHeadersAreEightByteAligned(t *testing.T) {
	cfg := &ArchConfig{
		Name:      "aligned_header_test",
		ModelDim:  32,
		VocabSize: 64,
		SeqLen:    8,
		Blocks:    []BlockSpec{{Type: "plain", Heads: 2}},
		Training:  TrainingSpec{Steps: 1, LR: 1e-3, Seed: 42, BatchTokens: 8},
	}
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatalf("computeWeightShapes: %v", err)
	}
	weights := initWeightData(shapes, cfg.Training.Seed, "", 0)
	dir := t.TempDir()
	plainPath := filepath.Join(dir, "plain.safetensors")
	if err := exportSafetensors(plainPath, cfg, shapes, weights); err != nil {
		t.Fatalf("exportSafetensors: %v", err)
	}
	if headerLen := safetensorHeaderLength(t, plainPath); headerLen%8 != 0 {
		t.Fatalf("plain header length = %d, want 8-byte aligned", headerLen)
	}
	assertSafetensorTensorOffsetsAligned(t, plainPath)

	quantPath := filepath.Join(dir, "quant.safetensors")
	if err := exportSafetensorsQuantized(quantPath, cfg, shapes, weights, "int8", "quantile", 0, 0); err != nil {
		t.Fatalf("exportSafetensorsQuantized: %v", err)
	}
	if headerLen := safetensorHeaderLength(t, quantPath); headerLen%8 != 0 {
		t.Fatalf("quantized header length = %d, want 8-byte aligned", headerLen)
	}
	assertSafetensorTensorOffsetsAligned(t, quantPath)
}

func safetensorHeaderLength(t *testing.T, path string) uint64 {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	if len(data) < 8 {
		t.Fatalf("%s too small", path)
	}
	return binary.LittleEndian.Uint64(data[:8])
}

func assertSafetensorTensorOffsetsAligned(t *testing.T, path string) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	headerLen := binary.LittleEndian.Uint64(data[:8])
	raw := map[string]json.RawMessage{}
	if err := json.Unmarshal(data[8:8+headerLen], &raw); err != nil {
		t.Fatalf("parse header %s: %v", path, err)
	}
	for name, msg := range raw {
		if name == "__metadata__" {
			continue
		}
		var entry safetensorHeaderEntry
		if err := json.Unmarshal(msg, &entry); err != nil {
			t.Fatalf("parse entry %s: %v", name, err)
		}
		if len(entry.DataOffsets) != 2 {
			t.Fatalf("%s offsets = %v, want 2 entries", name, entry.DataOffsets)
		}
		if entry.DataOffsets[0]%8 != 0 {
			t.Fatalf("%s start offset = %d, want 8-byte aligned", name, entry.DataOffsets[0])
		}
	}
}
