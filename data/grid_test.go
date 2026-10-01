package data

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func gridFixture(dtype int) []byte {
	ids, _ := json.Marshal([]string{"record"})
	h := make([]byte, 1024)
	for j, n := range []int{GridShardMagic, 1, dtype, 2, 1, 3, 1, 1, len(ids)} {
		binary.LittleEndian.PutUint32(h[j*4:], uint32(n))
	}
	h = append(h, ids...)
	b := h
	values := []float32{math.Float32frombits(0x80000000), 1, 2, 3, 4, 5, 6, 7, 8}
	if dtype == 1 {
		for _, v := range values {
			b = binary.LittleEndian.AppendUint32(b, math.Float32bits(v))
		}
	} else {
		for _, v := range []uint16{0x8000, 0x3c00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700, 0x4800} {
			b = binary.LittleEndian.AppendUint16(b, v)
		}
	}
	return append(b, 5)
}
func TestGridShardRoundTripAndCorruption(t *testing.T) {
	for _, dtype := range []int{1, 2} {
		path := filepath.Join(t.TempDir(), "data.grid")
		b := gridFixture(dtype)
		if err := os.WriteFile(path, b, 0600); err != nil {
			t.Fatal(err)
		}
		s, err := OpenGridShard(path)
		if err != nil {
			t.Fatal(err)
		}
		x, y, m := make([]float32, 6), make([]float32, 3), make([]float32, 3)
		if err = s.ReadNHWC(0, x, y, m); err != nil {
			t.Fatal(err)
		}
		_ = s.Close()
		if math.Float32bits(x[0]) != 0x80000000 || x[1] != 3 || x[2] != 1 || y[2] != 8 || m[0] != 1 || m[1] != 0 || m[2] != 1 {
			t.Fatal(x, y, m)
		}
	}
	for _, kind := range []string{"truncated", "trailing", "reserved", "dimensions", "nan", "mask_padding", "ids"} {
		t.Run(kind, func(t *testing.T) {
			b := gridFixture(1)
			runtimeFailure := false
			switch kind {
			case "truncated":
				b = b[:len(b)-1]
			case "trailing":
				b = append(b, 0)
			case "reserved":
				b[36] = 1
			case "dimensions":
				binary.LittleEndian.PutUint32(b[12:], 0xffffffff)
			case "nan":
				binary.LittleEndian.PutUint32(b[len(b)-37:], 0x7fc00000)
				runtimeFailure = true
			case "mask_padding":
				b[len(b)-1] = 128
				runtimeFailure = true
			case "ids":
				b[1026] = 0
			}
			path := filepath.Join(t.TempDir(), "bad.grid")
			if err := os.WriteFile(path, b, 0600); err != nil {
				t.Fatal(err)
			}
			s, err := OpenGridShard(path)
			if runtimeFailure {
				if err != nil {
					t.Fatal(err)
				}
				defer func() { _ = s.Close() }()
				err = s.ReadNHWC(0, make([]float32, 6), make([]float32, 3), make([]float32, 3))
			} else if err == nil {
				_ = s.Close()
			}
			if err == nil {
				t.Fatal("corrupt shard accepted")
			}
		})
	}
}
func TestGridHalfDecoder(t *testing.T) {
	for _, h := range []uint16{0, 1, 1023, 1024, 0x3c00, 0x7bff, 0x8000, 0x8001, 0xfbff} {
		sign := 1.
		if h&0x8000 != 0 {
			sign = -1
		}
		e := int((h >> 10) & 31)
		f := int(h & 1023)
		want := sign * math.Ldexp(float64(f), -24)
		if e != 0 {
			want = sign * math.Ldexp(1+float64(f)/1024, e-15)
		}
		if math.Float32bits(decodeGridHalf(h)) != math.Float32bits(float32(want)) {
			t.Fatalf("half %x", h)
		}
	}
}

func FuzzGridShard(f *testing.F) {
	f.Add(gridFixture(1))
	f.Fuzz(func(t *testing.T, b []byte) {
		if len(b) > 1<<16 {
			return
		}
		path := filepath.Join(t.TempDir(), "f.grid")
		if err := os.WriteFile(path, b, 0600); err != nil {
			t.Fatal(err)
		}
		s, err := OpenGridShard(path)
		if err == nil {
			_ = s.Close()
		}
	})
}
