package data

import (
	"fmt"
	"io"
	"sync"
)

type GridPrefetchOptions struct {
	BatchSize, Depth, Workers, Steps int
	Seed                             int64
	Dihedral                         bool
}

type gridPreparedBatch struct {
	batch *GridBatch
	epoch int
	err   error
}

// GridPrefetch owns its sampler. Next releases the previously returned batch;
// the consumer must finish using it before calling Next again. Close cancels
// queue waits and joins outstanding file reads before closing worker handles.
type GridPrefetch struct {
	opts       GridPrefetchOptions
	reader     *GridBatchReader
	sampler    *GridSampler
	augmenter  GridAugmenter
	free       chan *GridBatch
	ready      chan gridPreparedBatch
	stop, done chan struct{}
	once       sync.Once
	current    *GridBatch
	remaining  int
}

func NewGridPrefetch(d *GridDataset, sampler *GridSampler, opts GridPrefetchOptions) (*GridPrefetch, error) {
	if sampler == nil || opts.Depth < 0 || opts.Depth > 64 || opts.Steps < 0 {
		return nil, fmt.Errorf("invalid grid prefetch options")
	}
	r, err := NewGridBatchReader(d, opts.Workers)
	if err != nil {
		return nil, err
	}
	p := &GridPrefetch{opts: opts, reader: r, sampler: sampler, remaining: opts.Steps,
		free: make(chan *GridBatch, opts.Depth+1), ready: make(chan gridPreparedBatch, opts.Depth),
		stop: make(chan struct{}), done: make(chan struct{})}
	for i := 0; i <= opts.Depth; i++ {
		b, err := NewGridBatch(d.Geometry, opts.BatchSize)
		if err != nil {
			return nil, err
		}
		p.free <- &b
	}
	if opts.Depth > 0 {
		go p.run()
	} else {
		close(p.done)
	}
	return p, nil
}

func (p *GridPrefetch) prepare(b *GridBatch) gridPreparedBatch {
	ids, epoch, occurrence, err := p.sampler.Next(p.opts.BatchSize)
	if err == nil {
		err = p.reader.ReadInto(ids, b)
	}
	if err == nil && p.opts.Dihedral {
		err = p.augmenter.Apply(*b, p.opts.Seed, epoch, occurrence)
	}
	return gridPreparedBatch{b, epoch, err}
}

func (p *GridPrefetch) run() {
	defer close(p.done)
	defer close(p.ready)
	for i := 0; i < p.opts.Steps; i++ {
		var b *GridBatch
		select {
		case <-p.stop:
			return
		case b = <-p.free:
		}
		select {
		case <-p.stop:
			return
		default:
		}
		item := p.prepare(b)
		select {
		case <-p.stop:
			return
		case p.ready <- item:
		}
		if item.err != nil {
			return
		}
	}
}

func (p *GridPrefetch) Next() (GridBatch, int, error) {
	select {
	case <-p.stop:
		return GridBatch{}, 0, io.EOF
	default:
	}
	if p.current != nil {
		p.free <- p.current
		p.current = nil
	}
	if p.remaining == 0 {
		return GridBatch{}, 0, io.EOF
	}
	var item gridPreparedBatch
	if p.opts.Depth == 0 {
		item = p.prepare(<-p.free)
	} else {
		var ok bool
		select {
		case <-p.stop:
			return GridBatch{}, 0, io.EOF
		case item, ok = <-p.ready:
			if !ok {
				return GridBatch{}, 0, io.EOF
			}
		}
	}
	p.remaining--
	p.current = item.batch
	return *item.batch, item.epoch, item.err
}

// Close is idempotent. Call it on the consumer goroutine, not concurrently with
// Next. Cancellation cannot interrupt a blocking kernel/filesystem ReadAt.
func (p *GridPrefetch) Close() error {
	p.once.Do(func() { close(p.stop) })
	<-p.done
	return p.reader.Close()
}

// GridLoaderMemoryBounds counts float32 decoded data and worst-case scratch.
// Index/ID metadata and OS page cache are not part of the payload bound.
func GridLoaderMemoryBounds(g GridGeometry, dtype string, opts GridPrefetchOptions) (decoded, scratch int64) {
	pixels := int64(g.Height) * int64(g.Width)
	decoded = int64(opts.Depth+1) * int64(opts.BatchSize) * pixels * int64(g.Channels+2*g.TargetChannels) * 4
	width := int64(4)
	if dtype == "float16" {
		width = 2
	}
	encoded := pixels * int64(g.Channels+g.TargetChannels) * width
	if g.TargetChannels > 0 {
		encoded += (pixels + 7) / 8
	}
	scratch = int64(opts.Workers) * encoded
	if opts.Dihedral {
		scratch += pixels * int64(max(g.Channels, g.TargetChannels)) * 4
	}
	return decoded, scratch
}
