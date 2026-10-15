package tensors

import (
	"bytes"
	"encoding/binary"
	"math"
	"reflect"
	"testing"

	"github.com/mrothroc/mixlab/diloco/optimizer"
)

func TestF32RoundTripPayloadOrder(t *testing.T) {
	want := []optimizer.Tensor{{Name: "w2_z", Shape: []int{2}, Values: []float32{1, -3}}, {Name: "w10_a", Shape: []int{1, 1}, Values: []float32{.25}}}
	var b bytes.Buffer
	if err := Write(&b, want); err != nil {
		t.Fatal(err)
	}
	got, err := Read(bytes.NewReader(b.Bytes()), uint64(b.Len()))
	if err != nil || !reflect.DeepEqual(want, got) {
		t.Fatal(got, err)
	}
	var again bytes.Buffer
	if err := Write(&again, got); err != nil || !bytes.Equal(b.Bytes(), again.Bytes()) {
		t.Fatal("non deterministic write", err)
	}
}
func TestRejectMalformedSafetensors(t *testing.T) {
	for name, header := range map[string]string{
		"duplicate":       `{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`,
		"duplicate_field": `{"w":{"dtype":"F32","dtype":"F32","shape":[1],"data_offsets":[0,4]}}`,
		"gap":             `{"w":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}`,
		"overflow":        `{"w":{"dtype":"F32","shape":[1073741824,1073741824],"data_offsets":[0,4]}}`,
		"dtype":           `{"w":{"dtype":"F16","shape":[2],"data_offsets":[0,4]}}`,
		"unknown":         `{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4],"extra":1}}`,
		"empty":           `{}`,
	} {
		t.Run(name, func(t *testing.T) {
			var b bytes.Buffer
			_ = binary.Write(&b, binary.LittleEndian, uint64(len(header)))
			b.WriteString(header)
			b.Write(make([]byte, 4))
			if _, err := Read(bytes.NewReader(b.Bytes()), uint64(b.Len())); err == nil {
				t.Fatal("accepted malformed header")
			}
		})
	}
	var b bytes.Buffer
	_ = Write(&b, []optimizer.Tensor{{Name: "w", Shape: []int{1}, Values: []float32{1}}})
	for _, mutate := range []func([]byte) []byte{
		func(v []byte) []byte { return v[:len(v)-1] },
		func(v []byte) []byte { return append(v, 0) },
		func(v []byte) []byte {
			binary.LittleEndian.PutUint32(v[len(v)-4:], math.Float32bits(float32(math.NaN())))
			return v
		},
		func(v []byte) []byte { binary.LittleEndian.PutUint64(v, MaxHeader+1); return v },
	} {
		v := mutate(bytes.Clone(b.Bytes()))
		if _, err := Read(bytes.NewReader(v), uint64(len(v))); err == nil {
			t.Fatal("accepted malformed payload")
		}
	}
}
