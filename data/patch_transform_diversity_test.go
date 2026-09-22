package data

import (
	"fmt"
	"math"
	"path/filepath"
	"reflect"
	"testing"
)

// Exercise actual shard rollover, not just repeated calls to the transform.
// Record identity is encoded in pixels so shuffle cannot hide a frozen draw.
func TestPatchAugmentationAcrossEpochs(t *testing.T) {
	const records = 16
	const epochs = 200
	p := PatchTransform{Height: 12, Width: 12, Channels: 1, Patch: 2, CropPad: 4, HFlip: true, Training: true}
	const seqLen, featureDim = 36, 4
	dir := t.TempDir()
	for shard := 0; shard < 2; shard++ {
		var frames []float32
		var labels []int32
		for id := shard * 8; id < (shard+1)*8; id++ {
			pixels := make([]float32, 144)
			for i := range pixels {
				pixels[i] = float32((id+1)*10000 + i + 1)
			}
			frames = append(frames, packTestPixels(p, pixels)...)
			labels = append(labels, int32(id%2))
		}
		writeContinuousTestShard(t, filepath.Join(dir, fmt.Sprintf("train_%d.bin", shard)), 8, featureDim, labels, frames)
	}
	writeContinuousTestManifest(t, dir, records, seqLen, featureDim)
	// Two always-in-bounds central pixels uniquely identify all 162 draws.
	draws := make(map[[2]int][3]int)
	for y := 0; y <= 8; y++ {
		for x := 0; x <= 8; x++ {
			for flip := 0; flip < 2; flip++ {
				sx, dx := x, 1
				if flip == 1 {
					sx, dx = x+3, -1
				}
				draws[[2]int{y*12 + sx + 1, y*12 + sx + dx + 1}] = [3]int{y, x, flip}
			}
		}
	}
	for _, shuffle := range []bool{false, true} {
		t.Run(fmt.Sprintf("shuffle=%v", shuffle), func(t *testing.T) {
			loader, err := NewLoaderWithOptions(filepath.Join(dir, "train_*.bin"), 42, LoaderOptions{PatchTransform: &p, NoShardShuffle: !shuffle})
			if err != nil {
				t.Fatal(err)
			}
			seen := make([]map[[3]int]int, records)
			var firstTen [][3]int
			var histY, histX [9]int
			flips := 0
			for epoch := 0; epoch < epochs; epoch++ {
				counts := make([]int, records)
				// A single batch crosses both shard boundaries on later epochs.
				batch, err := loader.NextBatchDetailed(records*seqLen, seqLen)
				if err != nil {
					t.Fatal(err)
				}
				for row := 0; row < records; row++ {
					frame := batch.Frames[row*144 : (row+1)*144]
					a, b := int(frame[p.pixelIndex(4, 4, 0)]), int(frame[p.pixelIndex(4, 5, 0)])
					id := a/10000 - 1
					draw, ok := draws[[2]int{a % 10000, b % 10000}]
					if !ok || id < 0 || id >= records {
						t.Fatalf("invalid transformed record: %d %d", a, b)
					}
					if batch.Labels[row] != int32(id%2) {
						t.Fatal("image/label alignment changed")
					}
					counts[id]++
					if seen[id] == nil {
						seen[id] = make(map[[3]int]int)
					}
					seen[id][draw]++
					histY[draw[0]]++
					histX[draw[1]]++
					flips += draw[2]
					if id == 0 && epoch < 10 {
						firstTen = append(firstTen, draw)
					}
				}
				for id, count := range counts {
					if count != 1 {
						t.Fatalf("epoch %d record %d seen %d times", epoch, id, count)
					}
				}
				if loader.patches.occurrence != uint64((epoch+1)*records) {
					t.Fatal("augmentation counter reset across shards/epochs")
				}
			}
			minUnique, maxUnique := 162, 0
			for id, variants := range seen {
				minUnique, maxUnique = min(minUnique, len(variants)), max(maxUnique, len(variants))
				// About 115 distinct draws are expected after 200 visits. This
				// deliberately loose bound catches frozen or short-cycle streams.
				if len(variants) < 80 {
					t.Fatalf("record %d has only %d distinct draws", id, len(variants))
				}
			}
			for _, hist := range [][9]int{histY, histX} {
				for _, n := range hist {
					if n < 250 || n > 470 {
						t.Fatalf("biased crop histogram: %v", hist)
					}
				}
			}
			if flips < 1400 || flips > 1800 {
				t.Fatalf("biased flip count: %d", flips)
			}
			t.Logf("record 0 first ten draws (y,x,flip): %v; distinct per record=%d..%d/200; flips=%d/%d", firstTen, minUnique, maxUnique, flips, records*epochs)
		})
	}
}

func TestPatchAugmentationNormalizedPixelOracle(t *testing.T) {
	mean, std := []float32{0.4914, 0.4822, 0.4465}, []float32{0.2023, 0.1994, 0.2010}
	p := PatchTransform{Height: 32, Width: 32, Channels: 3, Patch: 4, CropPad: 4, PadValue: make([]float32, 3)}
	raw, normalized := make([]float32, 32*32*3), make([]float32, 32*32*3)
	for c := range mean {
		p.PadValue[c] = -mean[c] / std[c]
	}
	for i := range raw {
		raw[i] = float32((i*37)%256) / 255
		normalized[i] = (raw[i] - mean[i%3]) / std[i%3]
	}
	source := packTestPixels(p, normalized)
	for y := 0; y <= 8; y++ {
		for x := 0; x <= 8; x++ {
			for _, flip := range []bool{false, true} {
				// Reference order: pad raw pixels with black, crop, then flip,
				// then normalize. Production operates on already-normalized patches.
				crop := make([]float32, len(raw))
				for dy := 0; dy < 32; dy++ {
					for dx := 0; dx < 32; dx++ {
						sy, sx := dy+y-4, dx+x-4
						for c := 0; c < 3; c++ {
							if sy >= 0 && sy < 32 && sx >= 0 && sx < 32 {
								crop[(dy*32+dx)*3+c] = raw[(sy*32+sx)*3+c]
							}
						}
					}
				}
				want := make([]float32, len(raw))
				for dy := 0; dy < 32; dy++ {
					for dx := 0; dx < 32; dx++ {
						sx := dx
						if flip {
							sx = 31 - dx
						}
						for c := 0; c < 3; c++ {
							want[(dy*32+dx)*3+c] = (crop[(dy*32+sx)*3+c] - mean[c]) / std[c]
						}
					}
				}
				got := make([]float32, len(raw))
				p.transform(got, source, y, x, flip)
				for i, expected := range packTestPixels(p, want) {
					if math.Abs(float64(got[i]-expected)) > 1e-6 {
						t.Fatalf("crop=(%d,%d) flip=%v index=%d: %g != %g", y, x, flip, i, got[i], expected)
					}
				}
			}
		}
	}
	if !reflect.DeepEqual(source, packTestPixels(p, normalized)) {
		t.Fatal("transform mutated source pixels")
	}
}
