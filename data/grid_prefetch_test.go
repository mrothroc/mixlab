package data

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"
)

func gridReaderFixture(t testing.TB, dtype int) *GridDataset {
	return gridReaderFixtureGeometry(t, dtype, GridGeometry{Channels: 2, Height: 2, Width: 2, TargetChannels: 1}, 5)
}

func gridReaderFixtureGeometry(t testing.TB, dtype int, g GridGeometry, perShard int) *GridDataset {
	t.Helper()
	d := &GridDataset{Geometry: g, DType: "float32", currentIndex: -1}
	if dtype == 2 {
		d.DType = "float16"
	}
	dir := t.TempDir()
	for shard := 0; shard < 3; shard++ {
		ids := []string{}
		for i := 0; i < perShard; i++ {
			ids = append(ids, fmt.Sprintf("s%d-r%d", shard, i))
		}
		rawIDs, _ := json.Marshal(ids)
		raw := make([]byte, 1024)
		for i, v := range []int{GridShardMagic, 1, dtype, g.Channels, g.Height, g.Width, g.TargetChannels, perShard, len(rawIDs)} {
			binary.LittleEndian.PutUint32(raw[4*i:], uint32(v))
		}
		raw = append(raw, rawIDs...)
		for i, id := range ids {
			for j := 0; j < g.Height*g.Width*(g.Channels+g.TargetChannels); j++ {
				if dtype == 1 {
					raw = binary.LittleEndian.AppendUint32(raw, math.Float32bits(float32(100*shard+12*i+j)))
				} else {
					raw = binary.LittleEndian.AppendUint16(raw, uint16(0x3c00+shard*100+i*12+j%31))
				}
			}
			if g.TargetChannels > 0 {
				for p := 0; p < g.Height*g.Width; p += 8 {
					raw = append(raw, byte(1+i%15)&byte((1<<min(8, g.Height*g.Width-p))-1))
				}
			}
			d.records = append(d.records, gridRecordRef{shard, i, id})
		}
		path := filepath.Join(dir, fmt.Sprintf("%d.grid", shard))
		if err := os.WriteFile(path, raw, 0600); err != nil {
			t.Fatal(err)
		}
		d.paths = append(d.paths, path)
	}
	t.Cleanup(func() { _ = d.Close() })
	return d
}

func cloneGridBatch(b GridBatch) GridBatch {
	b.Inputs = slices.Clone(b.Inputs)
	b.Targets = slices.Clone(b.Targets)
	b.LossMask = slices.Clone(b.LossMask)
	b.IDs = slices.Clone(b.IDs)
	return b
}

func TestGridParallelReadParityAndPadding(t *testing.T) {
	for _, dtype := range []int{1, 2} {
		d := gridReaderFixture(t, dtype)
		for _, workers := range []int{1, 2, 8} {
			r, err := NewGridBatchReader(d, workers)
			if err != nil {
				t.Fatal(err)
			}
			b, err := NewGridBatch(d.Geometry, 8)
			if err != nil {
				t.Fatal(err)
			}
			for _, indices := range [][]int{{14, 0, 6, 1, 12, 5, 8, 9}, {4, 7}, {2, 2, 11}} {
				want, err := d.ReadBatch(indices, 8)
				if err != nil {
					t.Fatal(err)
				}
				if err = r.ReadInto(indices, &b); err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(b, want) {
					t.Fatalf("workers=%d dtype=%d: batch mismatch", workers, dtype)
				}
			}
			if err = r.ReadInto([]int{-1}, &b); err == nil {
				t.Fatal("accepted invalid index")
			}
			if err = r.Close(); err != nil {
				t.Fatal(err)
			}
		}
	}
}

func TestGridPrefetchOrderResumeAndBounds(t *testing.T) {
	d := gridReaderFixture(t, 2)
	const steps, batch, seed = 12, 4, 42
	want := []GridBatch{}
	snapshots := []GridSamplerState{}
	serial, _ := NewGridSampler(d.Len(), seed)
	var augmenter GridAugmenter
	for i := 0; i < steps; i++ {
		ids, epoch, occurrence, _ := serial.Next(batch)
		b, err := d.ReadBatch(ids, batch)
		if err != nil {
			t.Fatal(err)
		}
		if err = augmenter.Apply(b, seed, epoch, occurrence); err != nil {
			t.Fatal(err)
		}
		want = append(want, cloneGridBatch(b))
		snapshots = append(snapshots, serial.Snapshot())
	}
	for _, depth := range []int{0, 1, 2, 4} {
		for _, start := range []int{0, 3, 4, 5} {
			s, _ := NewGridSampler(d.Len(), seed)
			if start > 0 {
				if err := s.Restore(snapshots[start-1], start, batch); err != nil {
					t.Fatal(err)
				}
			}
			p, err := NewGridPrefetch(d, s, GridPrefetchOptions{BatchSize: batch, Depth: depth, Workers: 3, Steps: steps - start, Seed: seed, Dihedral: true})
			if err != nil {
				t.Fatal(err)
			}
			addresses := map[*float32]bool{}
			for i := start; i < steps; i++ {
				b, epoch, err := p.Next()
				if err != nil {
					t.Fatal(err)
				}
				addresses[&b.Inputs[0]] = true
				if !reflect.DeepEqual(b, want[i]) || epoch != snapshots[i].Epoch {
					t.Fatalf("depth=%d start=%d step=%d mismatch", depth, start, i)
				}
			}
			if len(addresses) > depth+1 {
				t.Fatal("unbounded batch buffers")
			}
			if _, _, err = p.Next(); err != io.EOF {
				t.Fatal(err)
			}
			if err = p.Close(); err != nil {
				t.Fatal(err)
			}
		}
	}
}

func TestGridPrefetchCloseAndReadErrors(t *testing.T) {
	d := gridReaderFixture(t, 1)
	for _, consumed := range []int{0, 1, 3} {
		s, _ := NewGridSampler(d.Len(), 1)
		p, err := NewGridPrefetch(d, s, GridPrefetchOptions{BatchSize: 4, Depth: 2, Workers: 2, Steps: 100})
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < consumed; i++ {
			if _, _, err = p.Next(); err != nil {
				t.Fatal(err)
			}
		}
		done := make(chan error, 1)
		go func() { done <- p.Close() }()
		select {
		case err = <-done:
			if err != nil {
				t.Fatal(err)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("Close blocked on full queue")
		}
		if err = p.Close(); err != nil {
			t.Fatal(err)
		}
		for _, w := range p.reader.workers {
			if w.shard != nil {
				t.Fatal("leaked shard handle")
			}
		}
	}
	// Corrupt a finite value without changing file length; all worker counts must
	// report the same first input-order failure and never suppress validation.
	raw, err := os.ReadFile(d.paths[0])
	if err != nil {
		t.Fatal(err)
	}
	start := 1024 + int(binary.LittleEndian.Uint32(raw[32:]))
	binary.LittleEndian.PutUint32(raw[start:], 0x7fc00000)
	if err = os.WriteFile(d.paths[0], raw, 0600); err != nil {
		t.Fatal(err)
	}
	for _, n := range []int{1, 4} {
		r, _ := NewGridBatchReader(d, n)
		b, _ := NewGridBatch(d.Geometry, 4)
		err = r.ReadInto([]int{0, 5, 0, 10}, &b)
		if err == nil || !strings.Contains(err.Error(), `grid record "s0-r0": grid input contains nonfinite`) {
			t.Fatal(err)
		}
		_ = r.Close()
	}
	for _, depth := range []int{0, 2} {
		s, _ := NewGridSampler(d.Len(), 1)
		p, err := NewGridPrefetch(d, s, GridPrefetchOptions{BatchSize: 15, Depth: depth, Workers: 4, Steps: 3})
		if err != nil {
			t.Fatal(err)
		}
		if _, _, err = p.Next(); err == nil || !strings.Contains(err.Error(), "nonfinite") {
			t.Fatal(err)
		}
		_ = p.Close()
	}
}

func TestGridLoaderMemoryBounds(t *testing.T) {
	g := GridGeometry{Channels: 7, TargetChannels: 1, Height: 256, Width: 256}
	o := GridPrefetchOptions{BatchSize: 16, Depth: 2, Workers: 8, Dihedral: true}
	decoded, scratch := GridLoaderMemoryBounds(g, "float16", o)
	if decoded != 3*16*256*256*9*4 || scratch != 8*(256*256*8*2+256*256/8)+256*256*7*4 {
		t.Fatal(decoded, scratch)
	}
}

func TestGridPrefetchConsumerBufferIsNotReusedEarly(t *testing.T) {
	d := gridReaderFixture(t, 1)
	s, _ := NewGridSampler(d.Len(), 42)
	p, err := NewGridPrefetch(d, s, GridPrefetchOptions{BatchSize: 4, Depth: 2, Workers: 3, Steps: 20})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = p.Close() }()
	b, _, err := p.Next()
	if err != nil {
		t.Fatal(err)
	}
	want := cloneGridBatch(b)
	deadline := time.Now().Add(5 * time.Second)
	for len(p.ready) < 2 {
		if time.Now().After(deadline) {
			t.Fatal("producer did not fill bounded read-ahead queue")
		}
		time.Sleep(time.Millisecond)
	}
	if !reflect.DeepEqual(b, want) {
		t.Fatal("producer overwrote live consumer batch")
	}
	if len(p.free) != 0 {
		t.Fatal("pool exceeds depth+1 while consumer and queue hold all buffers")
	}
}
