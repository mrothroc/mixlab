package data

import (
	"encoding/binary"
	"math"
	"os"
	"reflect"
	"slices"
	"strings"
	"testing"
)

func TestGridVectorsReadPrefetchResume(t *testing.T) {
	for _, dtype := range []int{1, 2} {
		g := GridGeometry{Channels: 2, Height: 2, Width: 2, TargetChannels: 1, VectorDim: 3}
		d := gridReaderFixtureGeometry(t, dtype, g, 5)
		const steps, size, seed = 12, 4, 42
		s, _ := NewGridSampler(d.Len(), seed)
		var want []GridBatch
		var snapshots []GridSamplerState
		for step := 0; step < steps; step++ {
			indices, epoch, occurrence, _ := s.Next(size)
			b, err := d.ReadBatch(indices, size)
			if err != nil {
				t.Fatal(err)
			}
			for row, index := range indices {
				ref := d.records[index]
				if b.IDs[row] != ref.id {
					t.Fatal("ID misalignment")
				}
				for j := 0; j < g.VectorDim; j++ {
					if b.Vectors[row*g.VectorDim+j] != float32(100*ref.shard+12*ref.index+j)+.123456 {
						t.Fatal("vector misalignment or fp16 truncation")
					}
				}
			}
			for row := b.Count; row < size; row++ {
				if !slices.Equal(b.Vectors[row*3:(row+1)*3], b.Vectors[:3]) {
					t.Fatal("padding must duplicate a real vector")
				}
				for _, v := range b.LossMask[row*4 : (row+1)*4] {
					if v != 0 {
						t.Fatal("padded loss")
					}
				}
			}
			vectors := slices.Clone(b.Vectors)
			var a GridAugmenter
			if err = a.Apply(b, seed, epoch, occurrence); err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(vectors, b.Vectors) {
				t.Fatal("generic vectors augmented")
			}
			want = append(want, cloneGridBatch(b))
			snapshots = append(snapshots, s.Snapshot())
		}
		for _, depth := range []int{0, 2} {
			for _, start := range []int{0, 3, 4, 7} {
				s, _ := NewGridSampler(d.Len(), seed)
				if start > 0 {
					if err := s.Restore(snapshots[start-1], start, size); err != nil {
						t.Fatal(err)
					}
				}
				p, err := NewGridPrefetch(d, s, GridPrefetchOptions{BatchSize: size, Depth: depth, Workers: 3, Steps: steps - start, Seed: seed, Dihedral: true})
				if err != nil {
					t.Fatal(err)
				}
				for step := start; step < steps; step++ {
					b, _, err := p.Next()
					if err != nil || !reflect.DeepEqual(b, want[step]) {
						t.Fatalf("dtype=%d depth=%d start=%d step=%d: %v", dtype, depth, start, step, err)
					}
				}
				if err = p.Close(); err != nil {
					t.Fatal(err)
				}
			}
		}
	}
}

func TestGridVectorShardCorruption(t *testing.T) {
	d := gridReaderFixtureGeometry(t, 2, GridGeometry{Channels: 1, Height: 1, Width: 1, TargetChannels: 1, VectorDim: 2}, 1)
	path := d.paths[0]
	original, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"version", "zero", "overflow", "reserved", "truncated", "trailing", "nan", "inf"} {
		t.Run(kind, func(t *testing.T) {
			raw := slices.Clone(original)
			switch kind {
			case "version":
				binary.LittleEndian.PutUint32(raw[4:], 3)
			case "zero":
				binary.LittleEndian.PutUint32(raw[36:], 0)
			case "overflow":
				binary.LittleEndian.PutUint32(raw[36:], math.MaxUint32)
			case "reserved":
				raw[40] = 1
			case "truncated":
				raw = raw[:len(raw)-1]
			case "trailing":
				raw = append(raw, 0)
			case "nan":
				binary.LittleEndian.PutUint32(raw[len(raw)-4:], 0x7fc00000)
			case "inf":
				binary.LittleEndian.PutUint32(raw[len(raw)-4:], 0x7f800000)
			}
			if err := os.WriteFile(path, raw, 0600); err != nil {
				t.Fatal(err)
			}
			shard, err := OpenGridShard(path)
			if kind == "nan" || kind == "inf" {
				if err != nil {
					t.Fatal(err)
				}
				err = shard.ReadNHWCWithVectors(0, make([]float32, 1), make([]float32, 1), make([]float32, 1), make([]float32, 2))
				_ = shard.Close()
				if err == nil || !strings.Contains(err.Error(), "s0-r0") {
					t.Fatal("nonfinite vector must identify record", err)
				}
			} else if err == nil {
				_ = shard.Close()
				t.Fatal("accepted corrupt shard")
			}
		})
	}
}

func TestGridVectorLoaderMemoryBounds(t *testing.T) {
	g := GridGeometry{Channels: 1, Height: 2, Width: 2, TargetChannels: 1}
	o := GridPrefetchOptions{BatchSize: 4, Depth: 2, Workers: 3}
	d0, s0 := GridLoaderMemoryBounds(g, "float16", o)
	g.VectorDim = 7
	d1, s1 := GridLoaderMemoryBounds(g, "float16", o)
	if d1-d0 != 3*4*7*4 || s1-s0 != 3*7*4 {
		t.Fatal(d1-d0, s1-s0)
	}
	g.VectorDim = math.MaxInt32
	if _, err := NewGridBatch(g, 2); err == nil {
		t.Fatal("unbounded vector allocation")
	}
}
