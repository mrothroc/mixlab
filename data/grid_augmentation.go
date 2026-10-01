package data

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
)

// GridDihedralIndex has no mutable RNG state, including across prefetch workers.
func GridDihedralIndex(seed int64, epoch, occurrence int, id string) int {
	h := sha256.New()
	_, _ = h.Write([]byte("mixlab.grid.d4.v1\x00"))
	var key [24]byte
	binary.LittleEndian.PutUint64(key[:8], uint64(seed))
	binary.LittleEndian.PutUint64(key[8:16], uint64(epoch))
	binary.LittleEndian.PutUint64(key[16:], uint64(occurrence))
	_, _ = h.Write(key[:])
	_, _ = h.Write([]byte(id))
	return int(h.Sum(nil)[0] & 7)
}

// GridAugmenter owns scratch storage; use one per batch-preparation worker.
type GridAugmenter struct{ scratch []float32 }

func (a *GridAugmenter) Apply(b GridBatch, seed int64, epoch, occurrence int) error {
	if b.Geometry.Height != b.Geometry.Width || b.Geometry.Height <= 0 || b.Count < 0 || b.Count > b.BatchSize || len(b.IDs) < b.Count {
		return fmt.Errorf("D4 augmentation requires square, valid grid batches")
	}
	n := b.Geometry.Height
	for row := 0; row < b.Count; row++ {
		transform := GridDihedralIndex(seed, epoch, occurrence+row, b.IDs[row])
		for k, values := range [][]float32{b.Inputs, b.Targets, b.LossMask} {
			channels := b.Geometry.TargetChannels
			if k == 0 {
				channels = b.Geometry.Channels
			}
			size := n * n * channels
			if len(values) != b.BatchSize*size {
				return fmt.Errorf("invalid D4 batch buffer length")
			}
			if err := a.Transform(values[row*size:(row+1)*size], n, channels, transform); err != nil {
				return err
			}
		}
	}
	return nil
}

// Transform mirrors columns for indices 4..7, then rotates counterclockwise
// by (index % 4)*90 degrees. All channels travel with their pixel.
func (a *GridAugmenter) Transform(values []float32, n, channels, index int) error {
	if n <= 0 || channels < 0 || len(values) != n*n*channels || index < 0 || index > 7 {
		return fmt.Errorf("invalid D4 transform")
	}
	if index == 0 || channels == 0 {
		return nil
	}
	if cap(a.scratch) < len(values) {
		a.scratch = make([]float32, len(values))
	}
	scratch := a.scratch[:len(values)]
	copy(scratch, values)
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			dy, dx := y, x
			if index >= 4 {
				dx = n - 1 - dx
			}
			for r := 0; r < index%4; r++ {
				dy, dx = n-1-dx, dy
			}
			copy(values[(dy*n+dx)*channels:(dy*n+dx+1)*channels], scratch[(y*n+x)*channels:(y*n+x+1)*channels])
		}
	}
	return nil
}
