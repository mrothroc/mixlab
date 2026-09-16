package data

import (
	"path/filepath"
	"reflect"
	"testing"
)

// Independent raster-image fixture, packing patches by nested loops rather
// than sharing the transform's addressing formula.
func packTestPixels(p PatchTransform, image []float32) []float32 {
	var out []float32
	for y := 0; y < p.Height; y += p.Patch {
		for x := 0; x < p.Width; x += p.Patch {
			for dy := 0; dy < p.Patch; dy++ {
				for dx := 0; dx < p.Patch; dx++ {
					for c := 0; c < p.Channels; c++ {
						out = append(out, image[((y+dy)*p.Width+x+dx)*p.Channels+c])
					}
				}
			}
		}
	}
	return out
}

func TestPatchTransformPixelOracle(t *testing.T) {
	p := PatchTransform{Height: 4, Width: 6, Channels: 2, Patch: 2, CropPad: 2, PadValue: []float32{-2, -3}}
	pixels := make([]float32, 48)
	for i := range pixels {
		pixels[i] = float32(i + 1)
	}
	src := packTestPixels(p, pixels)
	for _, flip := range []bool{false, true} {
		for _, offset := range [][2]int{{2, 2}, {0, 0}, {4, 1}} {
			want := make([]float32, len(pixels))
			for y := 0; y < p.Height; y++ {
				for x := 0; x < p.Width; x++ {
					for c := 0; c < p.Channels; c++ {
						cy, cx := y+offset[0]-2, x
						if flip {
							cx = p.Width - 1 - x
						}
						cx += offset[1] - 2
						v := p.PadValue[c]
						if cy >= 0 && cy < p.Height && cx >= 0 && cx < p.Width {
							v = pixels[(cy*p.Width+cx)*p.Channels+c]
						}
						want[(y*p.Width+x)*p.Channels+c] = v
					}
				}
			}
			got := make([]float32, len(src))
			p.transform(got, src, offset[0], offset[1], flip)
			if !reflect.DeepEqual(got, packTestPixels(p, want)) {
				t.Fatalf("flip=%v offset=%v got=%v", flip, offset, got)
			}
			if !flip && offset == [2]int{2, 2} && !reflect.DeepEqual(got, src) {
				t.Fatal("center crop not identity")
			}
		}
	}
	s, err := newPatchTransformState(&p, 13, 6, 8)
	if err != nil {
		t.Fatal(err)
	}
	if s.rng != nil || s.scratch != nil {
		t.Fatal("eval allocated augmentation state")
	}
	p.Training = true
	p.HFlip = true
	s, err = newPatchTransformState(&p, 13, 6, 8)
	if err != nil {
		t.Fatal(err)
	}
	if n := testing.AllocsPerRun(20, func() { s.apply(src) }); n != 0 {
		t.Fatalf("per-record allocations=%g", n)
	}
}

func TestPatchLoaderReplayDisabledAndEval(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "train_000.bin")
	frames := make([]float32, 3*48)
	for i := range frames {
		frames[i] = float32(i + 1)
	}
	writeContinuousTestShard(t, path, 3, 8, []int32{0, 1, 0}, frames)
	writeContinuousTestManifest(t, dir, 3, 6, 8)
	p := &PatchTransform{Height: 4, Width: 6, Channels: 2, Patch: 2}
	load := func(p *PatchTransform) *Loader {
		l, e := NewLoaderWithOptions(path, 71, LoaderOptions{PatchTransform: p})
		if e != nil {
			t.Fatal(e)
		}
		return l
	}
	next := func(l *Loader) Batch {
		b, e := l.NextBatchDetailed(12, 6)
		if e != nil {
			t.Fatal(e)
		}
		return b
	}
	a, b := load(nil), load(p)
	for i := 0; i < 5; i++ {
		if !reflect.DeepEqual(next(a), next(b)) {
			t.Fatal("disabled adapter changed batches")
		}
	}
	p.Training = true
	p.HFlip = true
	p.CropPad = 2
	a, b = load(p), load(p)
	baseline := load(nil)
	changed := false
	for i := 0; i < 10; i++ {
		x, y, z := next(a), next(b), next(baseline)
		if !reflect.DeepEqual(x, y) {
			t.Fatal("replay differs")
		}
		if !reflect.DeepEqual(x.Labels, z.Labels) {
			t.Fatal("augmentation changed shuffle")
		}
		changed = changed || !reflect.DeepEqual(x.Frames, z.Frames)
	}
	if !changed {
		t.Fatal("augmentation inactive")
	}
	// Reconstruct resume by replaying consumed batches from the same seed.
	restarted := load(p)
	for i := 0; i < 10; i++ {
		next(restarted)
	}
	if !reflect.DeepEqual(next(a), next(restarted)) {
		t.Fatal("resume replay differs")
	}
	for _, full := range []bool{false, true} {
		var v1, v2 *ValSet
		var err error
		if full {
			v1, err = NewClassificationValSetWithOptions(path, 0, 12, 6, LoaderOptions{})
			if err != nil {
				t.Fatal(err)
			}
			v2, err = NewClassificationValSetWithOptions(path, 0, 12, 6, LoaderOptions{PatchTransform: p})
		} else {
			v1, err = NewValSetWithOptions(path, 71, 3, 12, 6, LoaderOptions{})
			if err != nil {
				t.Fatal(err)
			}
			v2, err = NewValSetWithOptions(path, 71, 3, 12, 6, LoaderOptions{PatchTransform: p})
		}
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(v1, v2) {
			t.Fatal("eval augmented")
		}
	}
	shard, e := LoadContinuousSequenceShard(path)
	if e != nil {
		t.Fatal(e)
	}
	if !reflect.DeepEqual(shard.Frames, frames) {
		t.Fatal("mutated shard")
	}
	writeContinuousV2TestShard(t, path, 3, 6, 8, []int32{0, 1, 0}, []int32{6, 5, 6}, frames)
	if _, e = NewLoaderWithOptions(path, 71, LoaderOptions{PatchTransform: p}); e == nil {
		t.Fatal("accepted partial image")
	}
	if _, e = NewClassificationValSetWithOptions(path, 0, 12, 6, LoaderOptions{PatchTransform: p}); e == nil {
		t.Fatal("eval accepted partial image")
	}
}
