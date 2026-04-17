package train

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
)

// safetensorHeaderEntry describes one tensor in the safetensors header.
type safetensorHeaderEntry struct {
	DType       string   `json:"dtype"`
	Shape       []int    `json:"shape"`
	DataOffsets []uint64 `json:"data_offsets"`
}

// exportSafetensors writes trained weights to the safetensors format.
// Weights are stored as F32 (little-endian float32).
func exportSafetensors(path string, cfg *ArchConfig, shapes []WeightShape, weights [][]float32) error {
	if path == "" {
		return nil
	}
	if cfg == nil {
		return fmt.Errorf("nil config")
	}
	if len(shapes) != len(weights) {
		return fmt.Errorf("shapes/weights count mismatch: %d vs %d", len(shapes), len(weights))
	}

	// Build tensor names from weight shapes
	names := make([]string, len(shapes))
	for i, ws := range shapes {
		names[i] = fmt.Sprintf("w%d_%s", i, ws.Name)
	}

	// Build header
	header := make(map[string]safetensorHeaderEntry, len(weights))
	var offset uint64
	for i, ws := range shapes {
		sizeBytes := uint64(len(weights[i]) * 4)
		header[names[i]] = safetensorHeaderEntry{
			DType:       "F32",
			Shape:       append([]int(nil), ws.Shape...),
			DataOffsets: []uint64{offset, offset + sizeBytes},
		}
		offset += sizeBytes
	}

	// Add metadata
	meta := map[string]string{
		"name":       cfg.Name,
		"model_dim":  fmt.Sprintf("%d", cfg.ModelDim),
		"vocab_size": fmt.Sprintf("%d", cfg.VocabSize),
		"seq_len":    fmt.Sprintf("%d", cfg.SeqLen),
		"steps":      fmt.Sprintf("%d", cfg.Training.Steps),
		"format":     "mixlab_v1",
	}
	metaBytes, _ := json.Marshal(meta)
	header["__metadata__"] = safetensorHeaderEntry{
		DType:       "U8",
		Shape:       nil,
		DataOffsets: []uint64{0, 0},
	}
	_ = metaBytes // Metadata goes into the header JSON directly

	// Re-build header with metadata as a raw field
	headerMap := make(map[string]json.RawMessage, len(header)+1)
	for k, v := range header {
		if k == "__metadata__" {
			continue
		}
		b, err := json.Marshal(v)
		if err != nil {
			return err
		}
		headerMap[k] = b
	}
	headerMap["__metadata__"] = metaBytes

	headerBytes, err := json.Marshal(headerMap)
	if err != nil {
		return err
	}

	// Write file
	if dir := filepath.Dir(path); dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer func() { _ = f.Close() }()

	if err := binary.Write(f, binary.LittleEndian, uint64(len(headerBytes))); err != nil {
		return err
	}
	if _, err := f.Write(headerBytes); err != nil {
		return err
	}
	for i, data := range weights {
		var buf bytes.Buffer
		for _, v := range data {
			if err := binary.Write(&buf, binary.LittleEndian, math.Float32bits(v)); err != nil {
				return fmt.Errorf("encode tensor %s: %w", names[i], err)
			}
		}
		if _, err := f.Write(buf.Bytes()); err != nil {
			return err
		}
	}

	fmt.Printf("exported safetensors to %s (%d tensors)\n", path, len(weights))
	return nil
}

// encodeFloat32Data encodes float32 values as little-endian bytes.
func encodeFloat32Data(data []float32) []byte {
	out := make([]byte, len(data)*4)
	for i, v := range data {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
	}
	return out
}

// encodeFloat32Scalar encodes a single float32 as 4 little-endian bytes.
func encodeFloat32Scalar(v float32) []byte {
	out := make([]byte, 4)
	binary.LittleEndian.PutUint32(out, math.Float32bits(v))
	return out
}

// addSafetensorPayload adds a tensor entry to the header and appends its payload.
func addSafetensorPayload(header map[string]safetensorHeaderEntry, payloads *[][]byte, offset *uint64, name, dtype string, shape []int, data []byte) {
	start := *offset
	end := start + uint64(len(data))
	header[name] = safetensorHeaderEntry{
		DType:       dtype,
		Shape:       append([]int(nil), shape...),
		DataOffsets: []uint64{start, end},
	}
	*payloads = append(*payloads, data)
	*offset = end
}

// exportSafetensorsQuantized writes trained weights in quantized safetensors format.
// Supported modes: "int8" (row-wise 8-bit) and "int6" (row-wise 6-bit packed as int8).
// 1-D weights (norms, biases) and small tensors (<256 elements) are kept as F32.
// Quantized 2-D tensors store int8 data + a per-row F32 scale vector.
func exportSafetensorsQuantized(path string, cfg *ArchConfig, shapes []WeightShape, weights [][]float32, mode string) error {
	if path == "" {
		return nil
	}
	if cfg == nil {
		return fmt.Errorf("nil config")
	}
	if mode != "int8" && mode != "int6" {
		return fmt.Errorf("unsupported quantize mode %q", mode)
	}
	if len(shapes) != len(weights) {
		return fmt.Errorf("shapes/weights count mismatch: %d vs %d", len(shapes), len(weights))
	}

	names := make([]string, len(shapes))
	for i, ws := range shapes {
		names[i] = fmt.Sprintf("w%d_%s", i, ws.Name)
	}

	header := make(map[string]safetensorHeaderEntry, len(weights)*2)
	payloads := make([][]byte, 0, len(weights)*2)
	var offset uint64

	for i, ws := range shapes {
		name := names[i]
		data := weights[i]

		if !shouldQuantize(ws.Shape, len(data)) {
			// Keep small / 1-D tensors as float32.
			addSafetensorPayload(header, &payloads, &offset, name, "F32", ws.Shape, encodeFloat32Data(data))
			continue
		}

		if len(ws.Shape) == 2 {
			rows := ws.Shape[0]
			cols := ws.Shape[1]
			if rows*cols != len(data) {
				return fmt.Errorf("tensor %s shape/data mismatch: shape=%v data=%d", name, ws.Shape, len(data))
			}
			qData, scales := quantizeTensorPerRow(data, rows, cols, mode)
			addSafetensorPayload(header, &payloads, &offset, name, "I8", ws.Shape, int8ToBytes(qData))
			addSafetensorPayload(header, &payloads, &offset, name+".scale", "F32", []int{rows}, encodeFloat32Data(scales))
			continue
		}

		// 1-D large tensor (unlikely but handled): flat quantization.
		qData, scale := quantizeTensorFlat(data, mode)
		addSafetensorPayload(header, &payloads, &offset, name, "I8", ws.Shape, int8ToBytes(qData))
		addSafetensorPayload(header, &payloads, &offset, name+".scale", "F32", []int{}, encodeFloat32Scalar(scale))
	}

	// Add metadata with quantization info.
	meta := map[string]string{
		"name":       cfg.Name,
		"model_dim":  fmt.Sprintf("%d", cfg.ModelDim),
		"vocab_size": fmt.Sprintf("%d", cfg.VocabSize),
		"seq_len":    fmt.Sprintf("%d", cfg.SeqLen),
		"steps":      fmt.Sprintf("%d", cfg.Training.Steps),
		"quantize":   mode,
		"format":     "mixlab_v1",
	}
	metaBytes, _ := json.Marshal(meta)

	headerMap := make(map[string]json.RawMessage, len(header)+1)
	for k, v := range header {
		b, err := json.Marshal(v)
		if err != nil {
			return err
		}
		headerMap[k] = b
	}
	headerMap["__metadata__"] = metaBytes

	headerBytes, err := json.Marshal(headerMap)
	if err != nil {
		return err
	}

	if dir := filepath.Dir(path); dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer func() { _ = f.Close() }()

	if err := binary.Write(f, binary.LittleEndian, uint64(len(headerBytes))); err != nil {
		return err
	}
	if _, err := f.Write(headerBytes); err != nil {
		return err
	}
	for _, p := range payloads {
		if _, err := f.Write(p); err != nil {
			return err
		}
	}

	fmt.Printf("exported quantized (%s) safetensors to %s (%d tensors)\n", mode, path, len(weights))
	return nil
}
