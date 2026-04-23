package train

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
)

// TestQuantizePerRow_RoundTrip verifies that int8 per-row quantization produces
// bounded reconstruction error for typical weight distributions.
func TestQuantizePerRow_RoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	// Use realistic model-sized weights (256x512) to get meaningful statistics.
	rows, cols := 256, 512
	data := make([]float32, rows*cols)
	for i := range data {
		data[i] = float32(rng.NormFloat64()) * 0.1 // typical weight scale
	}

	qData, scales := quantizeTensorPerRow(data, rows, cols, "int8")
	recon := DequantizePerRow(qData, scales, rows, cols)

	sumSqErr := 0.0
	sumSqOrig := 0.0
	for i := range data {
		diff := float64(data[i] - recon[i])
		sumSqErr += diff * diff
		sumSqOrig += float64(data[i]) * float64(data[i])
	}

	relRMS := math.Sqrt(sumSqErr / sumSqOrig)
	t.Logf("int8 per-row: relRMS=%.6f", relRMS)

	// int8 per-row quantization: uniform quantization error is ~1/(127*sqrt(12)) per unit,
	// so for Gaussian weights relRMS should be well under 5%.
	if relRMS > 0.05 {
		t.Errorf("int8 relative RMS error %.4f exceeds 5%% threshold", relRMS)
	}
}

// TestQuantizePerRow_Int6_RoundTrip verifies int6 quantization.
func TestQuantizePerRow_Int6_RoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	rows, cols := 256, 512
	data := make([]float32, rows*cols)
	for i := range data {
		data[i] = float32(rng.NormFloat64()) * 0.1
	}

	qData, scales := quantizeTensorPerRow(data, rows, cols, "int6")
	recon := DequantizePerRow(qData, scales, rows, cols)

	sumSqErr := 0.0
	sumSqOrig := 0.0
	for i := range data {
		diff := float64(data[i] - recon[i])
		sumSqErr += diff * diff
		sumSqOrig += float64(data[i]) * float64(data[i])
	}

	relRMS := math.Sqrt(sumSqErr / sumSqOrig)
	t.Logf("int6 per-row: relRMS=%.6f", relRMS)

	// int6 rounds to multiples of 4 in int8 space, so ~4x coarser; allow up to 15%.
	if relRMS > 0.15 {
		t.Errorf("int6 relative RMS error %.4f exceeds 15%% threshold", relRMS)
	}

	// But it MUST be worse than int8 (sanity check).
	qData8, scales8 := quantizeTensorPerRow(data, rows, cols, "int8")
	recon8 := DequantizePerRow(qData8, scales8, rows, cols)
	sumSqErr8 := 0.0
	for i := range data {
		diff := float64(data[i] - recon8[i])
		sumSqErr8 += diff * diff
	}
	relRMS8 := math.Sqrt(sumSqErr8 / sumSqOrig)
	if relRMS <= relRMS8 {
		t.Errorf("int6 relRMS (%.6f) should be worse than int8 (%.6f)", relRMS, relRMS8)
	}
}

func TestQuantizePerRowSDClip_KControlsClipRange(t *testing.T) {
	data := []float32{
		-4, -2, 0, 2, 4,
		-8, -4, 0, 4, 8,
	}
	rows, cols := 2, 5

	_, lowScales := quantizeTensorPerRowSDClip(data, rows, cols, "int8", 1.5)
	_, highScales := quantizeTensorPerRowSDClip(data, rows, cols, "int8", 3.0)

	for r := 0; r < rows; r++ {
		if highScales[r] <= lowScales[r] {
			t.Fatalf("row %d high-k scale=%g should exceed low-k scale=%g", r, highScales[r], lowScales[r])
		}
	}
}

func TestQuantizePerRowSDClip_UniformDataUsesKStd(t *testing.T) {
	data := []float32{-3, -1, 1, 3}
	k := float32(2.0)

	_, scales := quantizeTensorPerRowSDClip(data, 1, len(data), "int8", k)

	// mean=0, variance=(9+1+1+9)/4=5, std=sqrt(5)
	wantScale := float32((float64(k) * math.Sqrt(5)) / 127.0)
	if math.Abs(float64(scales[0]-wantScale)) > 1e-6 {
		t.Fatalf("scale=%g, want %g", scales[0], wantScale)
	}
}

func TestQuantizePerRowSDClip_DiffersFromQuantile(t *testing.T) {
	data := []float32{-1, -0.5, 0, 0.5, 50}

	qQuantile, _ := quantizeTensorPerRow(data, 1, len(data), "int8")
	qSDClip, _ := quantizeTensorPerRowSDClip(data, 1, len(data), "int8", 1.0)

	if len(qQuantile) != len(qSDClip) {
		t.Fatalf("length mismatch: quantile=%d sdclip=%d", len(qQuantile), len(qSDClip))
	}
	for i := range qQuantile {
		if qQuantile[i] != qSDClip[i] {
			return
		}
	}
	t.Fatalf("SDClip quantized values should differ from quantile values: %v", qSDClip)
}

// TestQuantizeFlat_RoundTrip verifies flat (whole-tensor) quantization.
func TestQuantizeFlat_RoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(99))
	data := make([]float32, 512)
	for i := range data {
		data[i] = float32(rng.NormFloat64()) * 0.05
	}

	qData, scale := quantizeTensorFlat(data, "int8")
	recon := DequantizeFlat(qData, scale)

	sumSqErr := 0.0
	sumSqOrig := 0.0
	for i := range data {
		diff := float64(data[i] - recon[i])
		sumSqErr += diff * diff
		sumSqOrig += float64(data[i]) * float64(data[i])
	}

	relRMS := math.Sqrt(sumSqErr / sumSqOrig)
	t.Logf("int8 flat: relRMS=%.6f", relRMS)

	if relRMS > 0.01 {
		t.Errorf("int8 flat relative RMS error %.4f exceeds 1%% threshold", relRMS)
	}
}

// TestShouldQuantize verifies the skip logic for small/1-D tensors.
func TestShouldQuantize(t *testing.T) {
	tests := []struct {
		name   string
		shape  []int
		total  int
		expect bool
	}{
		{"2D large", []int{64, 128}, 64 * 128, true},
		{"2D small", []int{4, 4}, 16, false},
		{"1D norm", []int{256}, 256, false},
		{"1D tiny", []int{32}, 32, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := shouldQuantize(tt.shape, tt.total)
			if got != tt.expect {
				t.Errorf("shouldQuantize(%v, %d) = %v, want %v", tt.shape, tt.total, got, tt.expect)
			}
		})
	}
}

// TestExportSafetensorsQuantized_SmallerThanFloat32 writes both float32 and
// quantized safetensors files and verifies the quantized one is smaller.
func TestExportSafetensorsQuantized_SmallerThanFloat32(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	cfg := &ArchConfig{
		Name:      "test_quant",
		ModelDim:  64,
		VocabSize: 256,
		SeqLen:    32,
		Training:  TrainingSpec{Steps: 10},
	}

	shapes := []WeightShape{
		{Name: "embed", Shape: []int{256, 64}},
		{Name: "norm", Shape: []int{64}},
		{Name: "wq", Shape: []int{64, 64}},
		{Name: "wk", Shape: []int{64, 64}},
		{Name: "head", Shape: []int{64, 256}},
	}

	weights := make([][]float32, len(shapes))
	for i, ws := range shapes {
		total := 1
		for _, d := range ws.Shape {
			total *= d
		}
		w := make([]float32, total)
		for j := range w {
			w[j] = float32(rng.NormFloat64()) * 0.1
		}
		weights[i] = w
	}

	dir := t.TempDir()
	f32Path := filepath.Join(dir, "model_f32.safetensors")
	q8Path := filepath.Join(dir, "model_int8.safetensors")
	q6Path := filepath.Join(dir, "model_int6.safetensors")

	if err := exportSafetensors(f32Path, cfg, shapes, weights); err != nil {
		t.Fatalf("exportSafetensors: %v", err)
	}
	if err := exportSafetensorsQuantized(q8Path, cfg, shapes, weights, "int8", "", 0, 0); err != nil {
		t.Fatalf("exportSafetensorsQuantized(int8): %v", err)
	}
	if err := exportSafetensorsQuantized(q6Path, cfg, shapes, weights, "int6", "", 0, 0); err != nil {
		t.Fatalf("exportSafetensorsQuantized(int6): %v", err)
	}

	f32Info, _ := os.Stat(f32Path)
	q8Info, _ := os.Stat(q8Path)
	q6Info, _ := os.Stat(q6Path)

	t.Logf("f32=%d bytes, int8=%d bytes, int6=%d bytes",
		f32Info.Size(), q8Info.Size(), q6Info.Size())

	if q8Info.Size() >= f32Info.Size() {
		t.Errorf("int8 file (%d) should be smaller than f32 (%d)", q8Info.Size(), f32Info.Size())
	}
	if q6Info.Size() >= f32Info.Size() {
		t.Errorf("int6 file (%d) should be smaller than f32 (%d)", q6Info.Size(), f32Info.Size())
	}
}

// TestExportSafetensorsQuantized_HeaderDTypes verifies that the safetensors
// header marks quantized 2-D tensors as I8 and 1-D tensors as F32.
func TestExportSafetensorsQuantized_HeaderDTypes(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	cfg := &ArchConfig{
		Name:      "test_dtype",
		ModelDim:  32,
		VocabSize: 128,
		SeqLen:    16,
		Training:  TrainingSpec{Steps: 5},
	}

	shapes := []WeightShape{
		{Name: "embed", Shape: []int{128, 32}},
		{Name: "norm", Shape: []int{32}},
		{Name: "wq", Shape: []int{32, 32}},
	}

	weights := make([][]float32, len(shapes))
	for i, ws := range shapes {
		total := 1
		for _, d := range ws.Shape {
			total *= d
		}
		w := make([]float32, total)
		for j := range w {
			w[j] = float32(rng.NormFloat64()) * 0.1
		}
		weights[i] = w
	}

	dir := t.TempDir()
	path := filepath.Join(dir, "model_q8.safetensors")

	if err := exportSafetensorsQuantized(path, cfg, shapes, weights, "int8", "", 0, 0); err != nil {
		t.Fatalf("export: %v", err)
	}

	// Read back and parse header.
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer func() {
		if err := f.Close(); err != nil {
			t.Fatal(err)
		}
	}()

	var headerLen uint64
	if err := binary.Read(f, binary.LittleEndian, &headerLen); err != nil {
		t.Fatalf("read header length: %v", err)
	}
	headerBytes := make([]byte, headerLen)
	if _, err := f.Read(headerBytes); err != nil {
		t.Fatalf("read header: %v", err)
	}

	var header map[string]json.RawMessage
	if err := json.Unmarshal(headerBytes, &header); err != nil {
		t.Fatalf("parse header: %v", err)
	}

	// Check dtypes: embed (2D, >=256 elements) -> I8, norm (1D) -> F32, wq (2D, 1024 elements) -> I8.
	checkDType := func(name, expectedDType string) {
		t.Helper()
		raw, ok := header[name]
		if !ok {
			t.Errorf("tensor %q missing from header", name)
			return
		}
		var entry safetensorHeaderEntry
		if err := json.Unmarshal(raw, &entry); err != nil {
			t.Errorf("parse entry %q: %v", name, err)
			return
		}
		if entry.DType != expectedDType {
			t.Errorf("tensor %q dtype=%q, want %q", name, entry.DType, expectedDType)
		}
	}

	checkDType("w0_embed", "I8")
	checkDType("w1_norm", "F32") // 1-D, <256 elements -> float32
	checkDType("w2_wq", "I8")

	// Verify scale tensors exist for quantized weights.
	if _, ok := header["w0_embed.scale"]; !ok {
		t.Error("missing w0_embed.scale in header")
	}
	if _, ok := header["w2_wq.scale"]; !ok {
		t.Error("missing w2_wq.scale in header")
	}
	// norm should NOT have a scale.
	if _, ok := header["w1_norm.scale"]; ok {
		t.Error("w1_norm.scale should not exist (1-D tensor kept as F32)")
	}
}

// TestExportSafetensorsQuantized_InvalidMode returns error for bad mode.
func TestExportSafetensorsQuantized_InvalidMode(t *testing.T) {
	cfg := &ArchConfig{Name: "test", ModelDim: 32, VocabSize: 128, SeqLen: 16, Training: TrainingSpec{Steps: 1}}
	err := exportSafetensorsQuantized("/tmp/test.safetensors", cfg, nil, nil, "int4", "", 0, 0)
	if err == nil {
		t.Fatal("expected error for unsupported mode")
	}
}

// TestQuantizeClamp_Int8 verifies int8 mode clamping.
func TestQuantizeClamp_Int8(t *testing.T) {
	tests := []struct {
		name string
		raw  float32
		want int8
	}{
		{"zero", 0, 0},
		{"positive", 50, 50},
		{"negative", -50, -50},
		{"max_boundary", 127, 127},
		{"min_boundary", -127, -127},
		{"above_max", 200, 127},
		{"below_min", -200, -127},
		{"fractional_rounds", 50.6, 50}, // int8() truncates
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := quantizeClamp(tt.raw, "int8")
			if got != tt.want {
				t.Errorf("quantizeClamp(%v, int8)=%d, want %d", tt.raw, got, tt.want)
			}
		})
	}
}

// TestQuantizeClamp_Int6 verifies int6 mode clamping and rounding to multiples of 4.
func TestQuantizeClamp_Int6(t *testing.T) {
	tests := []struct {
		name string
		raw  float32
		want int8
	}{
		{"zero", 0, 0},
		{"exact_4", 4, 4},
		{"exact_neg4", -4, -4},
		{"round_up", 5, 4},   // round(5/4)*4 = round(1.25)*4 = 1*4 = 4
		{"round_up2", 6, 8},  // round(6/4)*4 = round(1.5)*4 = 2*4 = 8
		{"round_down", 3, 4}, // round(3/4)*4 = round(0.75)*4 = 1*4 = 4
		{"max_boundary", 124, 124},
		{"min_boundary", -128, -128},
		{"above_max", 200, 124},   // round(200/4)*4 = 200, clamped to 124
		{"below_min", -200, -128}, // round(-200/4)*4 = -200, clamped to -128
		{"near_max", 125, 124},    // round(125/4)*4 = round(31.25)*4 = 31*4 = 124
		{"near_max2", 126, 124},   // round(126/4)*4 = round(31.5)*4 = 32*4 = 128, clamped to 124
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := quantizeClamp(tt.raw, "int6")
			if got != tt.want {
				t.Errorf("quantizeClamp(%v, int6)=%d, want %d", tt.raw, got, tt.want)
			}
		})
	}
}

// TestQuantizeClamp_Int6_AllMultiplesOf4 verifies every int6 output is a multiple of 4.
func TestQuantizeClamp_Int6_AllMultiplesOf4(t *testing.T) {
	for raw := float32(-200); raw <= 200; raw += 0.5 {
		got := quantizeClamp(raw, "int6")
		if int(got)%4 != 0 {
			t.Errorf("quantizeClamp(%v, int6)=%d is not a multiple of 4", raw, got)
		}
	}
}

// TestClipQuantAbs_Empty verifies clipQuantAbs on empty input.
func TestClipQuantAbs_Empty(t *testing.T) {
	qz := &quantizer{}
	got := qz.clipQuantAbs(nil, 0.99)
	if got != 0 {
		t.Errorf("clipQuantAbs(nil)=%v, want 0", got)
	}
}

// TestClipQuantAbs_SingleElement verifies clipQuantAbs with one element.
func TestClipQuantAbs_SingleElement(t *testing.T) {
	qz := &quantizer{}
	got := qz.clipQuantAbs([]float32{-5.0}, 0.9999)
	if got != 5.0 {
		t.Errorf("clipQuantAbs([-5.0])=%v, want 5.0", got)
	}
}

// TestClipQuantAbs_AllZeros verifies clipQuantAbs with all-zero input.
func TestClipQuantAbs_AllZeros(t *testing.T) {
	qz := &quantizer{}
	got := qz.clipQuantAbs([]float32{0, 0, 0, 0}, 0.9999)
	if got != 0 {
		t.Errorf("clipQuantAbs(zeros)=%v, want 0", got)
	}
}

// TestClipQuantAbs_AllSameValue verifies clipQuantAbs when all values are identical.
func TestClipQuantAbs_AllSameValue(t *testing.T) {
	qz := &quantizer{}
	got := qz.clipQuantAbs([]float32{3.0, 3.0, 3.0, 3.0}, 0.9999)
	if got != 3.0 {
		t.Errorf("clipQuantAbs(all 3.0)=%v, want 3.0", got)
	}
}

// TestClipQuantAbs_AllNegative verifies clipQuantAbs returns absolute values.
func TestClipQuantAbs_AllNegative(t *testing.T) {
	qz := &quantizer{}
	got := qz.clipQuantAbs([]float32{-1.0, -2.0, -3.0, -4.0}, 0.9999)
	// q=0.9999 on 4 elements: idx = ceil(0.9999*4)-1 = ceil(3.9996)-1 = 4-1 = 3
	// Sorted abs: [1, 2, 3, 4], index 3 = 4.0
	if got != 4.0 {
		t.Errorf("clipQuantAbs(negatives)=%v, want 4.0", got)
	}
}

// TestClipQuantAbs_ScratchReuse verifies the scratch buffer is reused.
func TestClipQuantAbs_ScratchReuse(t *testing.T) {
	qz := &quantizer{}
	// First call with 4 elements.
	qz.clipQuantAbs([]float32{1, 2, 3, 4}, 0.5)
	// Second call with fewer elements should reuse scratch.
	got := qz.clipQuantAbs([]float32{10.0}, 0.9)
	if got != 10.0 {
		t.Errorf("clipQuantAbs after reuse=%v, want 10.0", got)
	}
}

// TestQuantizeZeroTensor verifies that an all-zero tensor doesn't panic.
func TestQuantizeZeroTensor(t *testing.T) {
	data := make([]float32, 256)
	qData, scale := quantizeTensorFlat(data, "int8")
	if scale != 1.0 {
		t.Errorf("expected scale=1.0 for zero tensor, got %f", scale)
	}
	for i, v := range qData {
		if v != 0 {
			t.Errorf("qData[%d]=%d, want 0", i, v)
		}
	}
}

// TestQuantizeEmptyTensor verifies that empty input doesn't panic.
func TestQuantizeEmptyTensor(t *testing.T) {
	qData, scale := quantizeTensorFlat(nil, "int8")
	if len(qData) != 0 {
		t.Errorf("expected empty qData, got %d elements", len(qData))
	}
	if scale != 1.0 {
		t.Errorf("expected scale=1.0, got %f", scale)
	}
}
