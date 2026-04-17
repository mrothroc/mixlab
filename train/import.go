package train

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
)

// safeTensorBlob holds parsed tensor data from a safetensors file.
type safeTensorBlob struct {
	DType string
	Shape []int
	Data  []byte
}

// loadSafetensors parses a safetensors file and returns all tensors keyed by name.
// The safetensors format is: 8-byte LE header length + JSON header + binary payload.
func loadSafetensors(path string) (map[string]safeTensorBlob, error) {
	blob, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read safetensors %q: %w", path, err)
	}
	if len(blob) < 8 {
		return nil, fmt.Errorf("invalid safetensors %q: too small (%d bytes)", path, len(blob))
	}
	headerLen := binary.LittleEndian.Uint64(blob[:8])
	if headerLen == 0 {
		return nil, fmt.Errorf("invalid safetensors %q: empty header", path)
	}
	if headerLen > uint64(len(blob)-8) {
		return nil, fmt.Errorf("invalid safetensors %q: header length %d exceeds file size %d", path, headerLen, len(blob))
	}
	headerBlob := blob[8 : 8+headerLen]
	payload := blob[8+headerLen:]

	raw := map[string]json.RawMessage{}
	if err := json.Unmarshal(headerBlob, &raw); err != nil {
		return nil, fmt.Errorf("parse safetensors header %q: %w", path, err)
	}
	out := make(map[string]safeTensorBlob, len(raw))
	for name, msg := range raw {
		if name == "__metadata__" {
			continue
		}
		var ent safetensorHeaderEntry
		if err := json.Unmarshal(msg, &ent); err != nil {
			return nil, fmt.Errorf("parse entry %q in %q: %w", name, path, err)
		}
		if len(ent.DataOffsets) != 2 {
			return nil, fmt.Errorf("invalid offsets for %q in %q: expected 2, got %d", name, path, len(ent.DataOffsets))
		}
		start := ent.DataOffsets[0]
		end := ent.DataOffsets[1]
		if end < start || end > uint64(len(payload)) {
			return nil, fmt.Errorf("invalid offsets for %q in %q: start=%d end=%d payload=%d", name, path, start, end, len(payload))
		}
		out[name] = safeTensorBlob{
			DType: ent.DType,
			Shape: append([]int(nil), ent.Shape...),
			Data:  append([]byte(nil), payload[start:end]...),
		}
	}
	return out, nil
}

// loadSafetensorsWeights loads safetensors from a file and returns weight data
// in the same order as the given shapes. Tensor names must match the export naming
// convention: "w{index}_{name}".
func loadSafetensorsWeights(path string, shapes []WeightShape) ([][]float32, error) {
	tensors, err := loadSafetensors(path)
	if err != nil {
		return nil, err
	}

	weights := make([][]float32, len(shapes))
	for i, ws := range shapes {
		name := fmt.Sprintf("w%d_%s", i, ws.Name)
		data, err := decodeSafetensorFloat32(name, ws.Shape, tensors)
		if err != nil {
			return nil, fmt.Errorf("load weight %d (%s): %w", i, name, err)
		}
		weights[i] = data
	}
	return weights, nil
}

// decodeSafetensorFloat32 decodes a single tensor from the parsed safetensors map.
func decodeSafetensorFloat32(name string, wantShape []int, tensors map[string]safeTensorBlob) ([]float32, error) {
	ent, ok := tensors[name]
	if !ok {
		return nil, fmt.Errorf("missing tensor %q in safetensors file", name)
	}
	if !shapesEqual(ent.Shape, wantShape) {
		return nil, fmt.Errorf("shape mismatch for %q: file=%v expected=%v", name, ent.Shape, wantShape)
	}
	n := shapeProduct(wantShape)
	if n <= 0 {
		return nil, fmt.Errorf("invalid shape for %q: %v", name, wantShape)
	}
	switch ent.DType {
	case "F32":
		if len(ent.Data) != n*4 {
			return nil, fmt.Errorf("F32 payload size mismatch for %q: got=%d want=%d", name, len(ent.Data), n*4)
		}
		out := make([]float32, n)
		for j := 0; j < n; j++ {
			out[j] = math.Float32frombits(binary.LittleEndian.Uint32(ent.Data[j*4:]))
		}
		return out, nil
	default:
		return nil, fmt.Errorf("unsupported dtype %q for %q (only F32 supported for weight loading)", ent.DType, name)
	}
}

// shapesEqual compares two shape slices for equality.
func shapesEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// shapeProduct computes the total element count from a shape.
func shapeProduct(shape []int) int {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}
