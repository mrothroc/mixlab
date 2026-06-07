package train

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
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
		addSafetensorHeaderEntry(header, &offset, names[i], "F32", ws.Shape, sizeBytes)
	}

	// Add metadata
	meta := map[string]string{
		"name":       cfg.Name,
		"model_dim":  fmt.Sprintf("%d", cfg.ModelDim),
		"vocab_size": fmt.Sprintf("%d", cfg.VocabSize),
		"seq_len":    fmt.Sprintf("%d", cfg.SeqLen),
		"steps":      fmt.Sprintf("%d", cfg.Training.TotalSteps()),
		"format":     "mixlab_v1",
	}
	metaBytes, _ := json.Marshal(meta)
	// Re-build header with metadata as a raw field
	headerMap := make(map[string]json.RawMessage, len(header)+1)
	for k, v := range header {
		b, err := json.Marshal(v)
		if err != nil {
			return err
		}
		headerMap[k] = b
	}
	headerMap["__metadata__"] = metaBytes

	headerBytes, err := marshalSafetensorHeader(headerMap)
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
		if err := writeSafetensorPayloadBytes(f, names[i], encodeFloat32Data(data)); err != nil {
			return err
		}
	}

	fmt.Printf("exported safetensors to %s (%d tensors)\n", path, len(weights))
	return nil
}

func marshalSafetensorHeader(headerMap map[string]json.RawMessage) ([]byte, error) {
	headerBytes, err := json.Marshal(headerMap)
	if err != nil {
		return nil, err
	}
	if rem := len(headerBytes) % 8; rem != 0 {
		headerBytes = append(headerBytes, bytesOfRepeatedSpace(8-rem)...)
	}
	return headerBytes, nil
}

func bytesOfRepeatedSpace(n int) []byte {
	if n <= 0 {
		return nil
	}
	out := make([]byte, n)
	for i := range out {
		out[i] = ' '
	}
	return out
}

func exportTrainingSafetensorsArtifacts(cfg *ArchConfig, trainer any, shapes []WeightShape, opts TrainOptions, swaEMA [][]float32) (safetensorsArtifacts, error) {
	if opts.SafetensorsPath == "" {
		return safetensorsArtifacts{}, nil
	}
	liveWeights, err := readTrainerWeights(trainer)
	if err != nil {
		return safetensorsArtifacts{}, fmt.Errorf("read trainer weights: %w", err)
	}
	if hasSWAWeights(swaEMA) {
		artifacts := suffixedSafetensorsPaths(opts.SafetensorsPath)
		if err := exportWeightsForMode(artifacts.FinalPath, cfg, shapes, liveWeights, opts); err != nil {
			return safetensorsArtifacts{}, fmt.Errorf("export final weights: %w", err)
		}
		if err := exportWeightsForMode(artifacts.SWAPath, cfg, shapes, swaEMA, opts); err != nil {
			return safetensorsArtifacts{}, fmt.Errorf("export swa weights: %w", err)
		}
		return artifacts, nil
	}
	artifacts := safetensorsArtifacts{FinalPath: opts.SafetensorsPath}
	if err := exportWeightsForMode(artifacts.FinalPath, cfg, shapes, liveWeights, opts); err != nil {
		return safetensorsArtifacts{}, err
	}
	return artifacts, nil
}

func exportWeightsForMode(path string, cfg *ArchConfig, shapes []WeightShape, weights [][]float32, opts TrainOptions) error {
	switch opts.Quantize {
	case "int8", "int6":
		return exportSafetensorsQuantized(path, cfg, shapes, weights, opts.Quantize, opts.QuantMethod, opts.QuantK, opts.QuantKEmbed)
	default:
		return exportSafetensors(path, cfg, shapes, weights)
	}
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

func addSafetensorHeaderEntry(header map[string]safetensorHeaderEntry, offset *uint64, name, dtype string, shape []int, sizeBytes uint64) {
	start := *offset
	end := start + sizeBytes
	header[name] = safetensorHeaderEntry{
		DType:       dtype,
		Shape:       append([]int(nil), shape...),
		DataOffsets: []uint64{start, end},
	}
	*offset = alignSafetensorOffset(end)
}

func alignSafetensorOffset(offset uint64) uint64 {
	if rem := offset % 8; rem != 0 {
		return offset + (8 - rem)
	}
	return offset
}

func safetensorPayloadPadding(sizeBytes uint64) int {
	aligned := alignSafetensorOffset(sizeBytes)
	return int(aligned - sizeBytes)
}

func writeSafetensorPayloadBytes(f *os.File, name string, data []byte) error {
	if _, err := f.Write(data); err != nil {
		return fmt.Errorf("write tensor %s: %w", name, err)
	}
	padding := safetensorPayloadPadding(uint64(len(data)))
	if padding == 0 {
		return nil
	}
	if _, err := f.Write(make([]byte, padding)); err != nil {
		return fmt.Errorf("write tensor %s padding: %w", name, err)
	}
	return nil
}

func addQuantizedTensorHeaders(header map[string]safetensorHeaderEntry, offset *uint64, name string, ws WeightShape, dataLen int) error {
	if !shouldQuantize(ws.Shape, dataLen) {
		addSafetensorHeaderEntry(header, offset, name, "F32", ws.Shape, uint64(dataLen*4))
		return nil
	}
	if len(ws.Shape) == 2 {
		rows := ws.Shape[0]
		cols := ws.Shape[1]
		if rows*cols != dataLen {
			return fmt.Errorf("tensor %s shape/data mismatch: shape=%v data=%d", name, ws.Shape, dataLen)
		}
		addSafetensorHeaderEntry(header, offset, name, "I8", ws.Shape, uint64(dataLen))
		addSafetensorHeaderEntry(header, offset, name+".scale", "F32", []int{rows}, uint64(rows*4))
		return nil
	}
	addSafetensorHeaderEntry(header, offset, name, "I8", ws.Shape, uint64(dataLen))
	addSafetensorHeaderEntry(header, offset, name+".scale", "F32", []int{}, 4)
	return nil
}

func writeQuantizedTensorPayload(f *os.File, name string, ws WeightShape, data []float32, mode, quantMethod string, kMatrix, kEmbed float32) error {
	if !shouldQuantize(ws.Shape, len(data)) {
		return writeSafetensorPayloadBytes(f, name, encodeFloat32Data(data))
	}
	if len(ws.Shape) == 2 {
		rows := ws.Shape[0]
		cols := ws.Shape[1]
		if rows*cols != len(data) {
			return fmt.Errorf("tensor %s shape/data mismatch: shape=%v data=%d", name, ws.Shape, len(data))
		}
		var qData []int8
		var scales []float32
		if quantMethod == "sdclip" {
			k := kMatrix
			if strings.Contains(strings.ToLower(ws.Name), "embed") {
				k = kEmbed
			}
			qData, scales = quantizeTensorPerRowSDClip(data, rows, cols, mode, k)
		} else {
			qData, scales = quantizeTensorPerRow(data, rows, cols, mode)
		}
		if err := writeSafetensorPayloadBytes(f, name, int8ToBytes(qData)); err != nil {
			return err
		}
		return writeSafetensorPayloadBytes(f, name+".scale", encodeFloat32Data(scales))
	}
	qData, scale := quantizeTensorFlat(data, mode)
	if err := writeSafetensorPayloadBytes(f, name, int8ToBytes(qData)); err != nil {
		return err
	}
	return writeSafetensorPayloadBytes(f, name+".scale", encodeFloat32Scalar(scale))
}

// exportSafetensorsQuantized writes trained weights in quantized safetensors format.
// Supported modes: "int8" (row-wise 8-bit) and "int6" (row-wise 6-bit packed as int8).
// 1-D weights (norms, biases) and small tensors (<256 elements) are kept as F32.
// Quantized 2-D tensors store int8 data + a per-row F32 scale vector.
func exportSafetensorsQuantized(path string, cfg *ArchConfig, shapes []WeightShape, weights [][]float32, mode string, quantMethod string, kMatrix, kEmbed float32) error {
	if path == "" {
		return nil
	}
	if cfg == nil {
		return fmt.Errorf("nil config")
	}
	if mode != "int8" && mode != "int6" {
		return fmt.Errorf("unsupported quantize mode %q", mode)
	}
	if quantMethod == "" {
		quantMethod = "quantile"
	}
	if quantMethod != "quantile" && quantMethod != "sdclip" {
		return fmt.Errorf("unsupported quantization method %q", quantMethod)
	}
	if len(shapes) != len(weights) {
		return fmt.Errorf("shapes/weights count mismatch: %d vs %d", len(shapes), len(weights))
	}

	names := make([]string, len(shapes))
	for i, ws := range shapes {
		names[i] = fmt.Sprintf("w%d_%s", i, ws.Name)
	}

	header := make(map[string]safetensorHeaderEntry, len(weights)*2)
	var offset uint64

	for i, ws := range shapes {
		name := names[i]
		if err := addQuantizedTensorHeaders(header, &offset, name, ws, len(weights[i])); err != nil {
			return err
		}
	}

	// Add metadata with quantization info.
	meta := map[string]string{
		"name":         cfg.Name,
		"model_dim":    fmt.Sprintf("%d", cfg.ModelDim),
		"vocab_size":   fmt.Sprintf("%d", cfg.VocabSize),
		"seq_len":      fmt.Sprintf("%d", cfg.SeqLen),
		"steps":        fmt.Sprintf("%d", cfg.Training.TotalSteps()),
		"quantize":     mode,
		"quant_method": quantMethod,
		"format":       "mixlab_v1",
	}
	if quantMethod == "sdclip" {
		meta["quant_k"] = fmt.Sprintf("%g", kMatrix)
		meta["quant_k_embed"] = fmt.Sprintf("%g", kEmbed)
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

	headerBytes, err := marshalSafetensorHeader(headerMap)
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
	for i, ws := range shapes {
		if err := writeQuantizedTensorPayload(f, names[i], ws, weights[i], mode, quantMethod, kMatrix, kEmbed); err != nil {
			return err
		}
	}

	fmt.Printf("exported quantized (%s) safetensors to %s (%d tensors)\n", mode, path, len(weights))
	return nil
}
