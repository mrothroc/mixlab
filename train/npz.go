//go:build mlx && cgo && (darwin || linux)

package train

import (
	"archive/zip"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"
)

type npzTensor struct {
	DType string
	Shape []int
	F32   []float32
	I64   []int64
}

func loadNPZ(path string) (map[string]npzTensor, error) {
	r, err := zip.OpenReader(path)
	if err != nil {
		return nil, fmt.Errorf("open npz %q: %w", path, err)
	}
	defer func() { _ = r.Close() }()

	out := make(map[string]npzTensor, len(r.File))
	for _, f := range r.File {
		rc, err := f.Open()
		if err != nil {
			return nil, fmt.Errorf("open npz entry %q: %w", f.Name, err)
		}
		data, err := io.ReadAll(rc)
		_ = rc.Close()
		if err != nil {
			return nil, fmt.Errorf("read npz entry %q: %w", f.Name, err)
		}
		name := strings.TrimSuffix(f.Name, ".npy")
		tensor, err := decodeNPY(data)
		if err != nil {
			return nil, fmt.Errorf("decode npy %q: %w", f.Name, err)
		}
		out[name] = tensor
	}
	return out, nil
}

func decodeNPY(data []byte) (npzTensor, error) {
	const magic = "\x93NUMPY"
	if len(data) < 10 || string(data[:6]) != magic {
		return npzTensor{}, fmt.Errorf("invalid npy header")
	}
	major := data[6]
	headerOffset := 10
	headerLen := 0
	switch major {
	case 1:
		headerLen = int(binary.LittleEndian.Uint16(data[8:10]))
	case 2, 3:
		if len(data) < 12 {
			return npzTensor{}, fmt.Errorf("truncated npy header")
		}
		headerLen = int(binary.LittleEndian.Uint32(data[8:12]))
		headerOffset = 12
	default:
		return npzTensor{}, fmt.Errorf("unsupported npy version %d", major)
	}
	if len(data) < headerOffset+headerLen {
		return npzTensor{}, fmt.Errorf("npy header exceeds file size")
	}
	header := string(data[headerOffset : headerOffset+headerLen])
	descr, shape, err := parseNPYHeader(header)
	if err != nil {
		return npzTensor{}, err
	}
	payload := data[headerOffset+headerLen:]
	n := shapeProduct(shape)
	switch descr {
	case "<f4":
		if len(payload) != n*4 {
			return npzTensor{}, fmt.Errorf("float payload mismatch: got=%d want=%d", len(payload), n*4)
		}
		out := make([]float32, n)
		for i := range out {
			out[i] = mathFloat32(payload[i*4:])
		}
		return npzTensor{DType: descr, Shape: shape, F32: out}, nil
	case "<i8":
		if len(payload) != n*8 {
			return npzTensor{}, fmt.Errorf("int64 payload mismatch: got=%d want=%d", len(payload), n*8)
		}
		out := make([]int64, n)
		for i := range out {
			out[i] = int64(binary.LittleEndian.Uint64(payload[i*8:]))
		}
		return npzTensor{DType: descr, Shape: shape, I64: out}, nil
	default:
		return npzTensor{}, fmt.Errorf("unsupported dtype %q", descr)
	}
}

func parseNPYHeader(header string) (string, []int, error) {
	if strings.Contains(header, "True") {
		return "", nil, fmt.Errorf("fortran_order=True is unsupported")
	}
	descr, err := parseNPYQuotedValue(header, "'descr':")
	if err != nil {
		return "", nil, err
	}
	shapeField, err := parseNPYTupleValue(header, "'shape':")
	if err != nil {
		return "", nil, err
	}
	shape, err := parseNPYShape(shapeField)
	if err != nil {
		return "", nil, err
	}
	return descr, shape, nil
}

func parseNPYQuotedValue(header, key string) (string, error) {
	idx := strings.Index(header, key)
	if idx < 0 {
		return "", fmt.Errorf("missing %s", key)
	}
	rest := header[idx+len(key):]
	start := strings.Index(rest, "'")
	if start < 0 {
		return "", fmt.Errorf("missing opening quote for %s", key)
	}
	rest = rest[start+1:]
	end := strings.Index(rest, "'")
	if end < 0 {
		return "", fmt.Errorf("missing closing quote for %s", key)
	}
	return rest[:end], nil
}

func parseNPYTupleValue(header, key string) (string, error) {
	idx := strings.Index(header, key)
	if idx < 0 {
		return "", fmt.Errorf("missing %s", key)
	}
	rest := header[idx+len(key):]
	start := strings.Index(rest, "(")
	end := strings.Index(rest, ")")
	if start < 0 || end < 0 || end <= start {
		return "", fmt.Errorf("invalid tuple for %s", key)
	}
	return rest[start+1 : end], nil
}

func parseNPYShape(s string) ([]int, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return []int{}, nil
	}
	parts := strings.Split(s, ",")
	shape := make([]int, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		n, err := strconv.Atoi(part)
		if err != nil {
			return nil, fmt.Errorf("parse shape dim %q: %w", part, err)
		}
		shape = append(shape, n)
	}
	return shape, nil
}

func mathFloat32(b []byte) float32 {
	return math.Float32frombits(binary.LittleEndian.Uint32(b[:4]))
}
