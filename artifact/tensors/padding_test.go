package tensors

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"testing"

	"github.com/mrothroc/mixlab/diloco/optimizer"
)

func paddingFixture(t *testing.T, format string, offsets [][2]uint64, payload []byte) []byte {
	t.Helper()
	header := make(map[string]any)
	if format != "" {
		header["__metadata__"] = map[string]string{"format": format}
	}
	for i, offset := range offsets {
		header[fmt.Sprintf("w%d_tensor", i)] = entry{"F32", []int{int((offset[1] - offset[0]) / 4)}, offset[:]}
	}
	b, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	for len(b)%8 != 0 {
		b = append(b, ' ')
	}
	var out bytes.Buffer
	if err := binary.Write(&out, binary.LittleEndian, uint64(len(b))); err != nil {
		t.Fatal(err)
	}
	out.Write(b)
	out.Write(payload)
	return out.Bytes()
}

func TestReadNativeF32PaddingAndCanonicalRewrite(t *testing.T) {
	payload := make([]byte, 24)
	for i, offset := range []int{0, 8, 12, 16} {
		binary.LittleEndian.PutUint32(payload[offset:], math.Float32bits(float32(i+1)))
	}
	native := paddingFixture(t, "mixlab_v1", [][2]uint64{{0, 4}, {8, 20}}, payload)
	want := []optimizer.Tensor{{Name: "w0_tensor", Shape: []int{1}, Values: []float32{1}}, {Name: "w1_tensor", Shape: []int{3}, Values: []float32{2, 3, 4}}}
	got, err := Read(bytes.NewReader(native), uint64(len(native)))
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatal("native padding changed data or ordering", got, err)
	}
	var canonical bytes.Buffer
	if err := Write(&canonical, got); err != nil {
		t.Fatal(err)
	}
	headerLen := binary.LittleEndian.Uint64(canonical.Bytes()[:8])
	var header map[string]entry
	if err := json.Unmarshal(canonical.Bytes()[8:8+headerLen], &header); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(header["w0_tensor"].Offsets, []uint64{0, 4}) || !reflect.DeepEqual(header["w1_tensor"].Offsets, []uint64{4, 16}) || uint64(canonical.Len()) != 8+headerLen+16 {
		t.Fatal("canonical writer introduced payload padding")
	}
	again, err := Read(bytes.NewReader(canonical.Bytes()), uint64(canonical.Len()))
	if err != nil || !reflect.DeepEqual(again, want) {
		t.Fatal("canonical rewrite changed data", err)
	}
}

func TestReadF32PaddingLayoutPolicy(t *testing.T) {
	for _, tc := range []struct {
		name, format string
		offsets      [][2]uint64
		bytes        int
		valid        bool
	}{
		{"canonical", "", [][2]uint64{{0, 4}, {4, 16}}, 16, true},
		{"native_metadata_contiguous", "mixlab_v1", [][2]uint64{{0, 4}, {4, 16}}, 16, true},
		{"native_single_odd", "mixlab_v1", [][2]uint64{{0, 4}}, 8, true},
		{"native_odd_then_even", "mixlab_v1", [][2]uint64{{0, 4}, {8, 16}}, 16, true},
		{"native_even_then_odd", "mixlab_v1", [][2]uint64{{0, 8}, {8, 12}}, 16, true},
		{"native_even", "mixlab_v1", [][2]uint64{{0, 8}, {8, 16}}, 16, true},
		{"unmarked_padding", "", [][2]uint64{{0, 4}, {8, 20}}, 24, false},
		{"unknown_format", "other", [][2]uint64{{0, 4}, {8, 20}}, 24, false},
		{"leading_gap", "mixlab_v1", [][2]uint64{{4, 8}, {8, 20}}, 24, false},
		{"oversized_gap", "mixlab_v1", [][2]uint64{{0, 4}, {12, 24}}, 24, false},
		{"overlap", "mixlab_v1", [][2]uint64{{0, 8}, {4, 16}}, 16, false},
		{"mixed_layout", "mixlab_v1", [][2]uint64{{0, 4}, {4, 8}, {16, 20}}, 24, false},
		{"packed_body_padded_tail", "mixlab_v1", [][2]uint64{{0, 4}, {4, 12}}, 16, false},
		{"missing_native_tail", "mixlab_v1", [][2]uint64{{0, 4}, {8, 20}}, 20, false},
		{"short_native_tail", "mixlab_v1", [][2]uint64{{0, 4}, {8, 20}}, 22, false},
		{"excess_native_tail", "mixlab_v1", [][2]uint64{{0, 4}, {8, 20}}, 28, false},
		{"already_aligned_tail", "mixlab_v1", [][2]uint64{{0, 8}}, 12, false},
		{"unmarked_tail", "", [][2]uint64{{0, 4}}, 8, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b := paddingFixture(t, tc.format, tc.offsets, make([]byte, tc.bytes))
			got, err := Read(bytes.NewReader(b), uint64(len(b)))
			if tc.valid && err != nil {
				t.Fatal(err)
			}
			if !tc.valid && (err == nil || got != nil) {
				t.Fatal("invalid layout produced tensors", err)
			}
		})
	}
}

func TestReadNativePaddingRejectsHiddenAndTrailingBytes(t *testing.T) {
	base := paddingFixture(t, "mixlab_v1", [][2]uint64{{0, 4}, {8, 20}}, make([]byte, 24))
	payload := int(binary.LittleEndian.Uint64(base[:8])) + 8
	for _, offset := range []int{4, 5, 6, 7, 20, 21, 22, 23} {
		t.Run(fmt.Sprintf("nonzero_padding_%d", offset), func(t *testing.T) {
			b := bytes.Clone(base)
			b[payload+offset] = 0x80
			if got, err := Read(bytes.NewReader(b), uint64(len(b))); err == nil || got != nil {
				t.Fatal("unchecked padding returned tensors", err)
			}
		})
	}
	for name, mutate := range map[string]func([]byte) []byte{
		"truncated":     func(b []byte) []byte { return b[:len(b)-1] },
		"extra_zero":    func(b []byte) []byte { return append(b, 0) },
		"extra_nonzero": func(b []byte) []byte { return append(b, 1) },
		"nonfinite_value": func(b []byte) []byte {
			binary.LittleEndian.PutUint32(b[payload+8:], math.Float32bits(float32(math.NaN())))
			return b
		},
	} {
		t.Run(name, func(t *testing.T) {
			b := mutate(bytes.Clone(base))
			// Exercise both honest length and a reader that violates its declared size.
			for _, size := range []uint64{uint64(len(b)), uint64(len(base))} {
				if got, err := Read(bytes.NewReader(b), size); err == nil || got != nil {
					t.Fatal("malformed native file returned tensors", err)
				}
			}
		})
	}
}
