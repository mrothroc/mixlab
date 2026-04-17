package train

import (
	"math"
	"unsafe"
)

// Quantization constants.
const (
	quantClipQ          = 0.9999 // Quantile for clipping outliers before quantization.
	quantKeepFloat1DMax = 256    // 1-D tensors with fewer elements stay float32.
	minRowScale         = 1.0 / 127.0
)

// quantizer holds reusable scratch buffers for quantization operations.
type quantizer struct {
	scratch []float32
}

// clipQuantAbs returns the q-th quantile of absolute values in vals.
func (qz *quantizer) clipQuantAbs(vals []float32, q float64) float32 {
	if len(vals) == 0 {
		return 0
	}
	if cap(qz.scratch) < len(vals) {
		qz.scratch = make([]float32, len(vals))
	}
	absVals := qz.scratch[:len(vals)]
	for i, v := range vals {
		absVals[i] = float32(math.Abs(float64(v)))
	}
	idx := int(math.Ceil(q*float64(len(absVals)))) - 1
	if idx < 0 {
		idx = 0
	}
	if idx >= len(absVals) {
		idx = len(absVals) - 1
	}
	return selectKthFloat32(absVals, idx)
}

// selectKthFloat32 returns the k-th smallest element using quickselect.
func selectKthFloat32(a []float32, k int) float32 {
	lo, hi := 0, len(a)-1
	for lo < hi {
		pivot := (lo + hi) / 2
		pivot = partitionFloat32(a, lo, hi, pivot)
		if k == pivot {
			return a[k]
		}
		if k < pivot {
			hi = pivot - 1
		} else {
			lo = pivot + 1
		}
	}
	return a[k]
}

func partitionFloat32(a []float32, lo, hi, pivot int) int {
	pv := a[pivot]
	a[pivot], a[hi] = a[hi], a[pivot]
	store := lo
	for i := lo; i < hi; i++ {
		if a[i] < pv {
			a[store], a[i] = a[i], a[store]
			store++
		}
	}
	a[store], a[hi] = a[hi], a[store]
	return store
}

// quantizeClamp clamps a raw quantized value to the valid range for the given mode.
func quantizeClamp(qRaw float32, mode string) int8 {
	if mode == "int6" {
		// 6-bit: round to multiples of 4, range [-128, 124].
		qRaw = float32(math.Round(float64(qRaw/4.0)) * 4.0)
		if qRaw < -128 {
			qRaw = -128
		}
		if qRaw > 124 {
			qRaw = 124
		}
	} else {
		if qRaw < -127 {
			qRaw = -127
		}
		if qRaw > 127 {
			qRaw = 127
		}
	}
	return int8(qRaw)
}

// quantizeTensorPerRow performs row-wise quantization on a 2-D tensor.
// For each row, it finds the clipping threshold, computes a per-row scale,
// and stores int8 quantized values. Returns quantized data and per-row scales.
func quantizeTensorPerRow(data []float32, rows, cols int, mode string) ([]int8, []float32) {
	qz := &quantizer{}
	qData := make([]int8, len(data))
	scales := make([]float32, rows)
	for r := 0; r < rows; r++ {
		row := data[r*cols : (r+1)*cols]
		clipAbs := qz.clipQuantAbs(row, quantClipQ)
		scale := float32(clipAbs / 127.0)
		if scale < minRowScale {
			scale = minRowScale
		}
		scales[r] = scale
		for c, v := range row {
			clipped := float32(math.Max(float64(-clipAbs), math.Min(float64(clipAbs), float64(v))))
			qRaw := float32(math.Round(float64(clipped / scale)))
			qData[r*cols+c] = quantizeClamp(qRaw, mode)
		}
	}
	return qData, scales
}

// quantizeTensorFlat performs whole-tensor quantization for 1-D tensors.
// Returns quantized int8 values and a single scalar scale.
func quantizeTensorFlat(data []float32, mode string) ([]int8, float32) {
	if len(data) == 0 {
		return []int8{}, 1.0
	}
	qz := &quantizer{}
	clipAbs := qz.clipQuantAbs(data, quantClipQ)
	scale := float32(clipAbs / 127.0)
	if clipAbs <= 0 {
		scale = 1.0
	}
	qData := make([]int8, len(data))
	for i, v := range data {
		clipped := float32(math.Max(float64(-clipAbs), math.Min(float64(clipAbs), float64(v))))
		qRaw := float32(math.Round(float64(clipped / scale)))
		qData[i] = quantizeClamp(qRaw, mode)
	}
	return qData, scale
}

// int8ToBytes reinterprets []int8 as []byte without copying.
// Safe because int8 and byte have identical size and alignment.
func int8ToBytes(data []int8) []byte {
	if len(data) == 0 {
		return nil
	}
	n := len(data)
	return (*[1 << 30]byte)(unsafe.Pointer(&data[0]))[:n:n]
}

// shouldQuantize returns true if a tensor with the given shape and element count
// should be quantized rather than kept as float32.
func shouldQuantize(shape []int, totalElements int) bool {
	if totalElements < quantKeepFloat1DMax {
		return false
	}
	if len(shape) == 1 {
		return false // norms, biases: keep float32
	}
	return true
}

// DequantizePerRow reconstructs float32 values from per-row quantized int8 + scales.
// This is used for verification/testing.
func DequantizePerRow(qData []int8, scales []float32, rows, cols int) []float32 {
	out := make([]float32, rows*cols)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			out[r*cols+c] = float32(qData[r*cols+c]) * scales[r]
		}
	}
	return out
}

// DequantizeFlat reconstructs float32 values from flat quantized int8 + scalar scale.
func DequantizeFlat(qData []int8, scale float32) []float32 {
	out := make([]float32, len(qData))
	for i, v := range qData {
		out[i] = float32(v) * scale
	}
	return out
}
