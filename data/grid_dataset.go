package data

import (
	"fmt"
	"path/filepath"
	"sort"
)

type gridRecordRef struct {
	shard, index int
	id           string
}

// GridDataset stores a small record index and at most one open shard/scratch.
type GridDataset struct {
	Geometry     GridGeometry
	DType        string
	paths        []string
	records      []gridRecordRef
	current      *GridShard
	currentIndex int
	batch        GridBatch
}

type GridBatch struct {
	Vectors                   []float32
	Inputs, Targets, LossMask []float32
	IDs                       []string
	Count, BatchSize          int
	Geometry                  GridGeometry
}

func OpenGridDataset(manifestPath, split string) (*GridDataset, error) {
	m, err := LoadDatasetManifest(manifestPath)
	if err != nil {
		return nil, err
	}
	if m.Representation != "grid" || m.Grid == nil {
		return nil, fmt.Errorf("expected grid dataset manifest")
	}
	s, ok := m.Splits[split]
	if !ok {
		return nil, fmt.Errorf("grid split %q not found", split)
	}
	paths, err := filepath.Glob(filepath.Join(filepath.Dir(manifestPath), s.Pattern))
	if err != nil {
		return nil, err
	}
	sort.Strings(paths)
	if len(paths) != s.Shards {
		return nil, fmt.Errorf("grid split shard count mismatch")
	}
	d := &GridDataset{Geometry: *m.Grid, DType: m.FeatureDType, paths: paths, currentIndex: -1}
	seen := map[string]bool{}
	for n, path := range paths {
		shard, err := OpenGridShard(path)
		if err != nil {
			return nil, fmt.Errorf("grid shard %s: %w", path, err)
		}
		if shard.Geometry != *m.Grid || shard.DType != m.FeatureDType {
			_ = shard.Close()
			return nil, fmt.Errorf("grid shard/manifest geometry or dtype mismatch")
		}
		for j, id := range shard.IDs {
			if seen[id] {
				_ = shard.Close()
				return nil, fmt.Errorf("duplicate grid ID %q across shards", id)
			}
			seen[id] = true
			d.records = append(d.records, gridRecordRef{n, j, id})
		}
		if err = shard.Close(); err != nil {
			return nil, err
		}
	}
	if int64(len(d.records)) != s.Sequences {
		return nil, fmt.Errorf("grid record count mismatch")
	}
	return d, nil
}

func (d *GridDataset) Len() int { return len(d.records) }
func (d *GridDataset) Close() error {
	if d.current != nil {
		err := d.current.Close()
		d.current = nil
		return err
	}
	return nil
}

// ReadBatch reuses batch storage; callers must finish consuming it before the
// next call. Short batches have zero maps/loss masks and duplicate a real vector.
func (d *GridDataset) ReadBatch(indices []int, batchSize int) (GridBatch, error) {
	if batchSize <= 0 || len(indices) > batchSize || len(indices) == 0 {
		return GridBatch{}, fmt.Errorf("invalid grid batch size")
	}
	g := d.Geometry
	p := g.Height * g.Width
	recordElements := int64(p)*int64(g.Channels+2*g.TargetChannels) + int64(g.VectorDim)
	if recordElements <= 0 || int64(batchSize) > mathMaxGridBatchElements/recordElements {
		return GridBatch{}, fmt.Errorf("grid batch exceeds element limit")
	}
	if d.batch.BatchSize != batchSize {
		var err error
		d.batch, err = NewGridBatch(g, batchSize)
		if err != nil {
			return GridBatch{}, err
		}
	}
	b := &d.batch
	clear(b.Inputs)
	clear(b.Targets)
	clear(b.LossMask)
	clear(b.IDs)
	clear(b.Vectors)
	b.Count = len(indices)
	for row, index := range indices {
		if index < 0 || index >= len(d.records) {
			return GridBatch{}, fmt.Errorf("grid index out of range")
		}
		r := d.records[index]
		if r.shard != d.currentIndex || d.current == nil {
			if err := d.Close(); err != nil {
				return GridBatch{}, err
			}
			s, err := OpenGridShard(d.paths[r.shard])
			if err != nil {
				return GridBatch{}, err
			}
			d.current = s
			d.currentIndex = r.shard
		}
		xN, yN := p*g.Channels, p*g.TargetChannels
		if d.current.Geometry != g || d.current.DType != d.DType {
			return GridBatch{}, fmt.Errorf("grid shard/manifest geometry or dtype mismatch")
		}
		if err := d.current.ReadNHWCWithVectors(r.index, b.Inputs[row*xN:(row+1)*xN], b.Targets[row*yN:(row+1)*yN], b.LossMask[row*yN:(row+1)*yN], b.Vectors[row*g.VectorDim:(row+1)*g.VectorDim]); err != nil {
			return GridBatch{}, fmt.Errorf("grid record %q: %w", r.id, err)
		}
		b.IDs[row] = r.id
	}
	padGridVectors(b)
	return *b, nil
}

// Includes input, target and broadcast mask buffers (at most 1 GiB total).
const mathMaxGridBatchElements = 1 << 28
