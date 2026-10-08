package data

import (
	"fmt"
	"sort"
	"sync"
)

// NewGridBatch allocates caller-owned decoded storage, including padded rows.
func NewGridBatch(g GridGeometry, batchSize int) (GridBatch, error) {
	if err := g.Validate(); err != nil {
		return GridBatch{}, err
	}
	p := int64(g.Height) * int64(g.Width)
	n := p * int64(g.Channels+2*g.TargetChannels)
	if batchSize <= 0 || n <= 0 || int64(batchSize) > mathMaxGridBatchElements/n {
		return GridBatch{}, fmt.Errorf("grid batch exceeds element limit or invalid batch size")
	}
	return GridBatch{BatchSize: batchSize, Geometry: g,
		Inputs:   make([]float32, batchSize*int(p)*g.Channels),
		Targets:  make([]float32, batchSize*int(p)*g.TargetChannels),
		LossMask: make([]float32, batchSize*int(p)*g.TargetChannels),
		IDs:      make([]string, batchSize)}, nil
}

type gridReadWorker struct {
	shard *GridShard
	index int
}

// GridBatchReader owns at most Workers shard handles and encoded-record scratch
// buffers. ReadInto and Close must not overlap; the dataset index is read-only.
// Unlike GridDataset.ReadBatch, decoded storage belongs to the caller.
type GridBatchReader struct {
	dataset *GridDataset
	workers []gridReadWorker
}

func NewGridBatchReader(d *GridDataset, workers int) (*GridBatchReader, error) {
	if d == nil || workers <= 0 {
		return nil, fmt.Errorf("invalid grid read worker count")
	}
	return &GridBatchReader{dataset: d, workers: make([]gridReadWorker, workers)}, nil
}

func (r *GridBatchReader) Close() error {
	var first error
	for i := range r.workers {
		w := &r.workers[i]
		if w.shard != nil {
			if err := w.shard.Close(); first == nil {
				first = err
			}
			w.shard = nil
		}
	}
	return first
}

func (r *GridBatchReader) ReadInto(indices []int, b *GridBatch) error {
	d := r.dataset
	if b == nil || b.Geometry != d.Geometry || b.BatchSize <= 0 || len(indices) == 0 || len(indices) > b.BatchSize {
		return fmt.Errorf("invalid grid batch")
	}
	g := d.Geometry
	xN, yN := g.Height*g.Width*g.Channels, g.Height*g.Width*g.TargetChannels
	if len(b.Inputs) != b.BatchSize*xN || len(b.Targets) != b.BatchSize*yN || len(b.LossMask) != b.BatchSize*yN || len(b.IDs) != b.BatchSize {
		return fmt.Errorf("invalid grid batch buffers")
	}
	rows := make([]int, len(indices))
	for row, index := range indices {
		if index < 0 || index >= d.Len() {
			return fmt.Errorf("grid index out of range")
		}
		rows[row] = row
	}
	// Group disk accesses by shard without changing their destination row order.
	sort.SliceStable(rows, func(i, j int) bool { return d.records[indices[rows[i]]].shard < d.records[indices[rows[j]]].shard })
	clear(b.Inputs)
	clear(b.Targets)
	clear(b.LossMask)
	clear(b.IDs)
	b.Count = len(indices)
	errs := make([]error, len(indices))
	n := min(len(r.workers), len(indices))
	read := func(worker, start, end int) {
		w := &r.workers[worker]
		for _, row := range rows[start:end] {
			ref := d.records[indices[row]]
			if w.shard == nil || w.index != ref.shard {
				if w.shard != nil {
					errs[row] = w.shard.Close()
					w.shard = nil
				}
				if errs[row] == nil {
					w.shard, errs[row] = OpenGridShard(d.paths[ref.shard])
				}
				w.index = ref.shard
			}
			if errs[row] == nil && (w.shard.Geometry != g || w.shard.DType != d.DType) {
				errs[row] = fmt.Errorf("grid shard/manifest geometry or dtype mismatch")
			}
			if errs[row] == nil {
				errs[row] = w.shard.ReadNHWC(ref.index, b.Inputs[row*xN:(row+1)*xN], b.Targets[row*yN:(row+1)*yN], b.LossMask[row*yN:(row+1)*yN])
			}
			if errs[row] != nil {
				errs[row] = fmt.Errorf("grid record %q: %w", ref.id, errs[row])
			}
			b.IDs[row] = ref.id
		}
	}
	if n == 1 {
		read(0, 0, len(rows))
	} else {
		var wg sync.WaitGroup
		for i := 0; i < n; i++ {
			wg.Add(1)
			go func(worker int) { defer wg.Done(); read(worker, worker*len(rows)/n, (worker+1)*len(rows)/n) }(i)
		}
		wg.Wait()
	}
	// Report the first input-order error, independent of worker scheduling.
	for _, err := range errs {
		if err != nil {
			return err
		}
	}
	return nil
}
