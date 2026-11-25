package data

import (
	"fmt"
	"math"
	"reflect"
	"slices"
	"testing"

	"github.com/mrothroc/mixlab/internal/gridpoints"
)

func TestGridPointsPrefetchRejectsMissingDataset(t *testing.T) {
	if _, err := NewGridPrefetch(nil, &GridSampler{}, GridPrefetchOptions{}); err == nil {
		t.Fatal("missing dataset accepted")
	}
}

func gridPointID(index int) string {
	for i := 0; ; i++ {
		id := fmt.Sprint(i)
		if GridDihedralIndex(42, 0, 0, id) == index {
			return id
		}
	}
}

func TestGridPointsFollowPixels(t *testing.T) {
	for _, n := range []int{8, 256} {
		points := [][2]int{{0, 0}, {0, n - 1}, {n - 1, 0}, {n - 1, n - 1}, {0, n / 2}, {n / 2, 0}, {n - 1, n / 2}, {n / 2, n - 1}, {2, 3}}
		for index := 0; index < 8; index++ {
			for k, point := range points {
				other := points[(k+1)%len(points)]
				b, err := NewGridBatch(GridGeometry{Height: n, Width: n, Channels: 2, TargetChannels: 2, VectorDim: 5}, 2)
				if err != nil {
					t.Fatal(err)
				}
				b.Count, b.IDs[0] = 1, gridPointID(index)
				b.Vectors = []float32{float32(point[1]), float32(other[1]), float32(point[0]), 123.25, float32(other[0]), 1, 2, 3, 4, 5}
				for channel, p := range [][2]int{point, other} {
					for _, values := range [][]float32{b.Inputs, b.Targets, b.LossMask} {
						values[(p[0]*n+p[1])*2+channel] = 1
					}
				}
				a := GridAugmenter{Points: gridpoints.Columns{{2, 0}, {4, 1}}}
				if err := a.Apply(b, 42, 0, 0); err != nil {
					t.Fatal(err)
				}
				for channel, pair := range a.Points {
					pixel := int(b.Vectors[pair[0]])*n + int(b.Vectors[pair[1]])
					for _, values := range [][]float32{b.Inputs, b.Targets, b.LossMask} {
						for i := 0; i < n*n; i++ {
							want := float32(0)
							if i == pixel {
								want = 1
							}
							if values[i*2+channel] != want {
								t.Fatalf("n=%d index=%d point=%v pixel=%d", n, index, point, i)
							}
						}
						for _, v := range values[n*n*2:] {
							if v != 0 {
								t.Fatal("padding map changed")
							}
						}
					}
				}
				if b.Vectors[3] != 123.25 || !slices.Equal(b.Vectors[5:], []float32{1, 2, 3, 4, 5}) {
					t.Fatal("sentinel/padding changed")
				}
			}
		}
	}
}

func TestGridPointsFloat32Order(t *testing.T) {
	// Frozen IEEE-754 results from an independent float32 step-by-step oracle.
	gold := [2][8][2]uint32{
		{{0x42c8999a, 0x4266cccd}, {0x43454ccd, 0x42c8999a}, {0x431ab333, 0x43454ccd}, {0x4266cccc, 0x431ab333}, {0x42c8999a, 0x43454ccd}, {0x4266cccc, 0x42c8999a}, {0x431ab333, 0x4266cccc}, {0x43454ccd, 0x431ab333}},
		{{0x3dcccccd, 0x437ee666}, {0x3dccd000, 0x3dcccccd}, {0x437ee666, 0x3dccd000}, {0x437ee666, 0x437ee666}, {0x3dcccccd, 0x3dccd000}, {0x437ee666, 0x3dcccccd}, {0x437ee666, 0x437ee666}, {0x3dccd000, 0x437ee666}},
	}
	for p, point := range [][2]float32{{100.3, 57.7}, {.1, 254.9}} {
		for index := 0; index < 8; index++ {
			b, _ := NewGridBatch(GridGeometry{Height: 256, Width: 256, Channels: 1, TargetChannels: 1, VectorDim: 2}, 1)
			b.Count, b.IDs[0] = 1, gridPointID(index)
			copy(b.Vectors, point[:])
			a := GridAugmenter{Points: gridpoints.Columns{{0, 1}}}
			if err := a.Apply(b, 42, 0, 0); err != nil {
				t.Fatal(err)
			}
			got := [2]uint32{math.Float32bits(b.Vectors[0]), math.Float32bits(b.Vectors[1])}
			if got != gold[p][index] {
				t.Fatalf("p=%d index=%d: %x != %x", p, index, got, gold[p][index])
			}
		}
	}
}

func TestGridPointsPrefetchReplay(t *testing.T) {
	g := GridGeometry{Height: 256, Width: 256, Channels: 1, TargetChannels: 1, VectorDim: 5}
	d := gridReaderFixtureGeometry(t, 1, g, 5)
	const steps, size, seed = 6, 4, 42
	points := gridpoints.Columns{{2, 0}, {4, 1}}
	s, _ := NewGridSampler(d.Len(), seed)
	var want []GridBatch
	var snapshots []GridSamplerState
	for step := 0; step < steps; step++ {
		ids, epoch, occurrence, _ := s.Next(size)
		b, err := d.ReadBatch(ids, size)
		if err != nil {
			t.Fatal(err)
		}
		original := slices.Clone(b.Vectors)
		var a GridAugmenter
		if err = a.Apply(b, seed, epoch, occurrence); err != nil {
			t.Fatal(err)
		}
		for row := 0; row < b.Count; row++ {
			index := GridDihedralIndex(seed, epoch, occurrence+row, b.IDs[row])
			for _, pair := range points {
				y, x := original[row*5+pair[0]], original[row*5+pair[1]]
				if index >= 4 {
					x = 255 - x
				}
				for r := 0; r < index%4; r++ {
					y, x = 255-x, y
				}
				b.Vectors[row*5+pair[0]], b.Vectors[row*5+pair[1]] = y, x
			}
		}
		want = append(want, cloneGridBatch(b))
		snapshots = append(snapshots, s.Snapshot())
	}
	for _, depth := range []int{0, 1, 3} {
		for _, start := range []int{0, 2, 3} {
			s, _ := NewGridSampler(d.Len(), seed)
			if start > 0 {
				if err := s.Restore(snapshots[start-1], start, size); err != nil {
					t.Fatal(err)
				}
			}
			owned := slices.Clone(points)
			p, err := NewGridPrefetch(d, s, GridPrefetchOptions{BatchSize: size, Depth: depth, Workers: 3, Steps: steps - start, Seed: seed, Dihedral: true, VectorPoints: owned})
			if err != nil {
				t.Fatal(err)
			}
			owned[0] = [2]int{0, 1} // producer owns an immutable copy
			for step := start; step < steps; step++ {
				b, _, err := p.Next()
				if err != nil || !reflect.DeepEqual(b, want[step]) {
					t.Fatalf("depth=%d start=%d step=%d: %v", depth, start, step, err)
				}
			}
			if err = p.Close(); err != nil {
				t.Fatal(err)
			}
		}
	}
	// A declaration alone must never enable augmentation.
	for _, depth := range []int{0, 2} {
		s, _ := NewGridSampler(d.Len(), seed)
		reference, _ := NewGridSampler(d.Len(), seed)
		ids, _, _, _ := reference.Next(size)
		want, err := d.ReadBatch(ids, size)
		if err != nil {
			t.Fatal(err)
		}
		p, err := NewGridPrefetch(d, s, GridPrefetchOptions{BatchSize: size, Depth: depth, Workers: 2, Steps: 1, Seed: seed, VectorPoints: points})
		if err != nil {
			t.Fatal(err)
		}
		got, _, err := p.Next()
		if err != nil || !reflect.DeepEqual(got, want) {
			t.Fatal("declaration enabled augmentation", err)
		}
		if err = p.Close(); err != nil {
			t.Fatal(err)
		}
	}
}
