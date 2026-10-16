// Package tensors is a bounded F32 safetensors codec. It imports no model
// compiler, trainer or GPU runtime. Payload order is preserved, not key order.
package tensors

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"sort"

	"github.com/mrothroc/mixlab/artifact"
	"github.com/mrothroc/mixlab/diloco/optimizer"
)

const MaxHeader = 1 << 20

type entry struct {
	DType   string   `json:"dtype"`
	Shape   []int    `json:"shape"`
	Offsets []uint64 `json:"data_offsets"`
}

func Write(w io.Writer, ts []optimizer.Tensor) error {
	if err := optimizer.Validate(ts); err != nil {
		return err
	}
	header := map[string]entry{}
	var offset uint64
	for _, t := range ts {
		if t.Name == "__metadata__" {
			return fmt.Errorf("reserved tensor name")
		}
		n := uint64(len(t.Values)) * 4
		header[t.Name] = entry{"F32", t.Shape, []uint64{offset, offset + n}}
		offset += n
	}
	b, err := json.Marshal(header)
	if err != nil {
		return err
	}
	for len(b)%8 != 0 {
		b = append(b, ' ')
	}
	if len(b) > MaxHeader || uint64(len(b))+8+offset > artifact.MaxBytes {
		return fmt.Errorf("safetensors byte bound exceeded")
	}
	if err = binary.Write(w, binary.LittleEndian, uint64(len(b))); err != nil {
		return err
	}
	if n, e := w.Write(b); e != nil {
		return e
	} else if n != len(b) {
		return io.ErrShortWrite
	}
	buffer := make([]byte, 64<<10)
	for _, t := range ts {
		for start := 0; start < len(t.Values); {
			n := min(len(buffer)/4, len(t.Values)-start)
			for i := 0; i < n; i++ {
				binary.LittleEndian.PutUint32(buffer[i*4:], math.Float32bits(t.Values[start+i]))
			}
			if count, e := w.Write(buffer[:n*4]); e != nil {
				return e
			} else if count != n*4 {
				return io.ErrShortWrite
			}
			start += n
		}
	}
	return nil
}

// Read accepts contiguous F32 payloads and native mixlab_v1 checkpoints whose
// tensors (including the last) are zero-padded to eight-byte boundaries. Native
// metadata selects a checked layout, not a trusted source. All other gaps,
// overlaps, trailing bytes, malformed layouts and non-finite values are rejected.
func Read(r io.Reader, size uint64) ([]optimizer.Tensor, error) {
	if size < 10 || size > artifact.MaxBytes {
		return nil, fmt.Errorf("invalid safetensors size")
	}
	var length uint64
	if err := binary.Read(r, binary.LittleEndian, &length); err != nil {
		return nil, err
	}
	if length < 2 || length > MaxHeader || length > size-8 {
		return nil, fmt.Errorf("invalid safetensors header length")
	}
	b := make([]byte, int(length))
	if _, err := io.ReadFull(r, b); err != nil {
		return nil, err
	}
	fields, err := object(b)
	if err != nil {
		return nil, err
	}
	type named struct {
		name string
		e    entry
	}
	entries := make([]named, 0, len(fields))
	native := false
	for name, raw := range fields {
		if name == "__metadata__" {
			m, err := object(raw)
			if err != nil {
				return nil, err
			}
			for key, v := range m {
				var s string
				if err = json.Unmarshal(v, &s); err != nil || bytes.Equal(v, []byte("null")) {
					return nil, fmt.Errorf("invalid metadata")
				}
				if key == "format" && s == "mixlab_v1" {
					native = true
				}
			}
			continue
		}
		f, err := object(raw)
		if err != nil {
			return nil, err
		}
		if len(f) != 3 || f["dtype"] == nil || f["shape"] == nil || f["data_offsets"] == nil {
			return nil, fmt.Errorf("invalid tensor fields")
		}
		var e entry
		if err = json.Unmarshal(raw, &e); err != nil {
			return nil, err
		}
		if e.DType != "F32" || len(e.Offsets) != 2 || len(e.Shape) == 0 || len(e.Shape) > 8 {
			return nil, fmt.Errorf("unsupported tensor layout")
		}
		n := uint64(1)
		for _, d := range e.Shape {
			if d <= 0 || uint64(d) > artifact.MaxBytes/4/n {
				return nil, fmt.Errorf("oversized tensor shape")
			}
			n *= uint64(d)
		}
		if e.Offsets[0] > e.Offsets[1] || e.Offsets[1] > size-8-length || e.Offsets[1]-e.Offsets[0] != n*4 {
			return nil, fmt.Errorf("invalid tensor offsets")
		}
		entries = append(entries, named{name, e})
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].e.Offsets[0] < entries[j].e.Offsets[0] })
	var end uint64
	contiguous, aligned := true, native
	for _, t := range entries {
		contiguous = contiguous && t.e.Offsets[0] == end
		aligned = aligned && t.e.Offsets[0] == alignNativeOffset(end)
		end = t.e.Offsets[1]
	}
	payloadSize := size - 8 - length
	contiguous = contiguous && end == payloadSize
	aligned = aligned && alignNativeOffset(end) == payloadSize
	if len(entries) == 0 || (!contiguous && !aligned) {
		return nil, fmt.Errorf("payload coverage mismatch")
	}
	out := make([]optimizer.Tensor, len(entries))
	buffer := make([]byte, 64<<10)
	var position uint64
	for i, t := range entries {
		if err := readNativePadding(r, t.e.Offsets[0]-position); err != nil {
			return nil, err
		}
		n := int((t.e.Offsets[1] - t.e.Offsets[0]) / 4)
		v := make([]float32, n)
		for start := 0; start < n; {
			count := min(len(buffer)/4, n-start)
			if _, err := io.ReadFull(r, buffer[:count*4]); err != nil {
				return nil, err
			}
			for j := 0; j < count; j++ {
				v[start+j] = math.Float32frombits(binary.LittleEndian.Uint32(buffer[j*4:]))
			}
			start += count
		}
		out[i] = optimizer.Tensor{Name: t.name, Shape: t.e.Shape, Values: v}
		position = t.e.Offsets[1]
	}
	if err := readNativePadding(r, payloadSize-position); err != nil {
		return nil, err
	}
	var tail [1]byte
	if _, err := io.ReadFull(r, tail[:]); err != io.EOF {
		return nil, fmt.Errorf("trailing safetensors bytes")
	}
	if err := optimizer.Validate(out); err != nil {
		return nil, err
	}
	return out, nil
}

// Offsets are already bounded by artifact.MaxBytes before alignment. F32
// payloads need at most four bytes; never skip unchecked attacker-chosen gaps.
func alignNativeOffset(offset uint64) uint64 { return (offset + 7) &^ 7 }

func readNativePadding(r io.Reader, n uint64) error {
	if n == 0 {
		return nil
	}
	if n != 4 {
		return fmt.Errorf("invalid native tensor padding length")
	}
	var padding [4]byte
	if _, err := io.ReadFull(r, padding[:]); err != nil {
		return err
	}
	if padding != [4]byte{} {
		return fmt.Errorf("nonzero native tensor padding")
	}
	return nil
}

func object(b []byte) (map[string]json.RawMessage, error) {
	d := json.NewDecoder(bytes.NewReader(b))
	tok, err := d.Token()
	if err != nil || tok != json.Delim('{') {
		return nil, fmt.Errorf("JSON object required")
	}
	out := map[string]json.RawMessage{}
	for d.More() {
		k, err := d.Token()
		if err != nil {
			return nil, err
		}
		name, ok := k.(string)
		if !ok || out[name] != nil {
			return nil, fmt.Errorf("duplicate JSON key")
		}
		var v json.RawMessage
		if err = d.Decode(&v); err != nil {
			return nil, err
		}
		out[name] = v
	}
	if _, err = d.Token(); err != nil {
		return nil, err
	}
	if _, err = d.Token(); err != io.EOF {
		return nil, fmt.Errorf("trailing JSON")
	}
	return out, nil
}
