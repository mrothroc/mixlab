package data

import (
	"math/rand"
	"reflect"
	"slices"
	"sync"
	"testing"
)

func TestGridD4EveryTransform(t *testing.T) {
	want := [][]float32{
		{1, 2, 3, 4, 5, 6, 7, 8, 9}, {3, 6, 9, 2, 5, 8, 1, 4, 7}, {9, 8, 7, 6, 5, 4, 3, 2, 1}, {7, 4, 1, 8, 5, 2, 9, 6, 3},
		{3, 2, 1, 6, 5, 4, 9, 8, 7}, {1, 4, 7, 2, 5, 8, 3, 6, 9}, {7, 8, 9, 4, 5, 6, 1, 2, 3}, {9, 6, 3, 8, 5, 2, 7, 4, 1},
	}
	for index := 0; index < 8; index++ {
		x, y, mask := make([]float32, 18), make([]float32, 9), make([]float32, 9)
		for p := 0; p < 9; p++ {
			x[p*2] = float32(p + 1)
			if p == 2 {
				x[p*2+1] = 1
			}
			y[p] = float32((p + 1) * 10)
			mask[p] = float32(p % 2)
		}
		var a GridAugmenter
		for k, values := range [][]float32{x, y, mask} {
			c := 1
			if k == 0 {
				c = 2
			}
			if err := a.Transform(values, 3, c, index); err != nil {
				t.Fatal(err)
			}
		}
		for p, v := range want[index] {
			if x[p*2] != v || y[p] != 10*v || mask[p] != float32((int(v)-1)%2) || (x[p*2+1] == 1) != (v == 3) {
				t.Fatalf("transform=%d pixel=%d x=%v y=%v mask=%v", index, p, x, y, mask)
			}
		}
	}
}

func TestGridD4OccurrenceAndPrefetchDeterminism(t *testing.T) {
	serial := make([]int, 128)
	parallel := make([]int, 128)
	var wg sync.WaitGroup
	for j := range serial {
		serial[j] = GridDihedralIndex(42, j/8, j%8, "repeated-record")
		wg.Add(1)
		go func(j int) { defer wg.Done(); parallel[j] = GridDihedralIndex(42, j/8, j%8, "repeated-record") }(j)
	}
	wg.Wait()
	if !slices.Equal(serial, parallel) {
		t.Fatal("worker ordering changed transformations")
	}
	seen := map[int]bool{}
	for _, v := range serial {
		seen[v] = true
	}
	if len(seen) != 8 {
		t.Fatal("occurrences did not cover D4", seen)
	}
	b := GridBatch{Inputs: []float32{1, 2, 3, 4, 0, 0, 0, 0}, Targets: []float32{1, 2, 3, 4, 0, 0, 0, 0}, LossMask: []float32{1, 0, 0, 1, 0, 0, 0, 0}, IDs: []string{"a", ""}, Count: 1, BatchSize: 2, Geometry: GridGeometry{Channels: 1, TargetChannels: 1, Height: 2, Width: 2}}
	var a GridAugmenter
	if err := a.Apply(b, 42, 1, 0); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(b.Inputs, b.Targets) {
		t.Fatal("targets diverged from inputs")
	}
	for j, v := range b.Inputs {
		if j >= 4 && (v != 0 || b.LossMask[j] != 0) {
			t.Fatal("augmented padding")
		}
		if j < 4 && (b.LossMask[j] == 1) != (v == 1 || v == 4) {
			t.Fatal("mask transform differs")
		}
	}
}

func TestGridSamplerLegacyOrderAndResume(t *testing.T) {
	const seed = 42
	s, err := NewGridSampler(5, seed)
	if err != nil {
		t.Fatal(err)
	}
	legacy := rand.New(rand.NewSource(seed))
	for epoch := 0; epoch < 4; epoch++ {
		want := legacy.Perm(5)
		for cursor := 0; cursor < 5; cursor += 2 {
			indices, e, o, err := s.Next(2)
			if err != nil {
				t.Fatal(err)
			}
			if e != epoch || o != cursor || !slices.Equal(indices, want[cursor:min(cursor+2, 5)]) {
				t.Fatal("changed legacy order", indices, e, o)
			}
		}
	}
	for _, step := range []int{1, 3, 4, 11} {
		original, _ := NewGridSampler(5, seed)
		for j := 0; j < step; j++ {
			if _, _, _, err = original.Next(2); err != nil {
				t.Fatal(err)
			}
		}
		saved := original.Snapshot()
		restored, _ := NewGridSampler(5, seed)
		if err = restored.Restore(saved, step, 2); err != nil {
			t.Fatal(err)
		}
		for j := 0; j < 10; j++ {
			x, xe, xo, _ := original.Next(2)
			y, ye, yo, _ := restored.Next(2)
			if !slices.Equal(x, y) || xe != ye || xo != yo {
				t.Fatal("resume diverged")
			}
		}
		bad := saved
		bad.Cursor++
		restored, _ = NewGridSampler(5, seed)
		if err = restored.Restore(bad, step, 2); err == nil {
			t.Fatal("accepted corrupted cursor")
		}
	}
	before := s.Snapshot()
	before.Order[0] = -1
	if reflect.DeepEqual(before, s.Snapshot()) {
		t.Fatal("snapshot aliases sampler")
	}
}
