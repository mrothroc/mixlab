package train

import (
	"encoding/binary"
	"math"
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
