package train

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

// TestLoadUint16LUT_Valid creates a small binary LUT file and verifies loading.
func TestLoadUint16LUT_Valid(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test_u16.bin")

	// Write 4 uint16 values: 10, 20, 30, 40
	vals := []uint16{10, 20, 30, 40}
	buf := make([]byte, len(vals)*2)
	for i, v := range vals {
		binary.LittleEndian.PutUint16(buf[i*2:], v)
	}
	if err := os.WriteFile(path, buf, 0o644); err != nil {
		t.Fatal(err)
	}

	got, err := loadUint16LUT(path, 4)
	if err != nil {
		t.Fatalf("loadUint16LUT: %v", err)
	}
	if len(got) != 4 {
		t.Fatalf("expected 4 elements, got %d", len(got))
	}
	for i, v := range vals {
		if got[i] != v {
			t.Errorf("got[%d]=%d, want %d", i, got[i], v)
		}
	}
}

// TestLoadUint16LUT_TooShort verifies error when LUT has fewer entries than expected.
func TestLoadUint16LUT_TooShort(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "short_u16.bin")

	// Write 2 uint16 values but expect 4.
	buf := make([]byte, 4) // 2 uint16s
	binary.LittleEndian.PutUint16(buf[0:], 1)
	binary.LittleEndian.PutUint16(buf[2:], 2)
	if err := os.WriteFile(path, buf, 0o644); err != nil {
		t.Fatal(err)
	}

	_, err := loadUint16LUT(path, 4)
	if err == nil {
		t.Error("expected error for LUT too short")
	}
}

// TestLoadUint16LUT_OddSize verifies error when file has odd byte count.
func TestLoadUint16LUT_OddSize(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "odd_u16.bin")

	// Write 3 bytes (not divisible by 2).
	if err := os.WriteFile(path, []byte{1, 2, 3}, 0o644); err != nil {
		t.Fatal(err)
	}

	_, err := loadUint16LUT(path, 1)
	if err == nil {
		t.Error("expected error for odd-sized LUT file")
	}
}

// TestLoadUint16LUT_FileNotFound verifies error for missing file.
func TestLoadUint16LUT_FileNotFound(t *testing.T) {
	_, err := loadUint16LUT("/nonexistent/path/lut.bin", 1)
	if err == nil {
		t.Error("expected error for non-existent file")
	}
}

// TestLoadUint16LUT_MoreThanExpected verifies success when file has more entries than expected.
func TestLoadUint16LUT_MoreThanExpected(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "extra_u16.bin")

	vals := []uint16{100, 200, 300}
	buf := make([]byte, len(vals)*2)
	for i, v := range vals {
		binary.LittleEndian.PutUint16(buf[i*2:], v)
	}
	if err := os.WriteFile(path, buf, 0o644); err != nil {
		t.Fatal(err)
	}

	got, err := loadUint16LUT(path, 2) // expect 2, have 3
	if err != nil {
		t.Fatalf("loadUint16LUT: %v", err)
	}
	if len(got) != 3 {
		t.Fatalf("expected 3 elements, got %d", len(got))
	}
}

// TestLoadBoolLUT_Valid creates a small bool LUT file and verifies loading.
func TestLoadBoolLUT_Valid(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test_bool.bin")

	// Write 5 bytes: true, false, true, true, false
	blob := []byte{1, 0, 1, 1, 0}
	if err := os.WriteFile(path, blob, 0o644); err != nil {
		t.Fatal(err)
	}

	got, err := loadBoolLUT(path, 5)
	if err != nil {
		t.Fatalf("loadBoolLUT: %v", err)
	}
	if len(got) != 5 {
		t.Fatalf("expected 5 elements, got %d", len(got))
	}
	expected := []bool{true, false, true, true, false}
	for i, v := range expected {
		if got[i] != v {
			t.Errorf("got[%d]=%v, want %v", i, got[i], v)
		}
	}
}

// TestLoadBoolLUT_TooShort verifies error when bool LUT is shorter than expected.
func TestLoadBoolLUT_TooShort(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "short_bool.bin")

	if err := os.WriteFile(path, []byte{1, 0}, 0o644); err != nil {
		t.Fatal(err)
	}

	_, err := loadBoolLUT(path, 5)
	if err == nil {
		t.Error("expected error for bool LUT too short")
	}
}

// TestLoadBoolLUT_FileNotFound verifies error for missing file.
func TestLoadBoolLUT_FileNotFound(t *testing.T) {
	_, err := loadBoolLUT("/nonexistent/path/bool.bin", 1)
	if err == nil {
		t.Error("expected error for non-existent file")
	}
}

// TestLoadBoolLUT_NonzeroValues verifies that any nonzero byte maps to true.
func TestLoadBoolLUT_NonzeroValues(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "nonzero_bool.bin")

	blob := []byte{0, 1, 2, 127, 255}
	if err := os.WriteFile(path, blob, 0o644); err != nil {
		t.Fatal(err)
	}

	got, err := loadBoolLUT(path, 5)
	if err != nil {
		t.Fatalf("loadBoolLUT: %v", err)
	}
	expected := []bool{false, true, true, true, true}
	for i, v := range expected {
		if got[i] != v {
			t.Errorf("got[%d]=%v, want %v", i, got[i], v)
		}
	}
}
