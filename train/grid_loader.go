package train

import (
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/data"
)

func newTrainingGridPrefetch(cfg *ArchConfig, ds *data.GridDataset, consumed *data.GridSampler, startStep, steps int) (*data.GridPrefetch, error) {
	producer, err := data.NewGridSampler(ds.Len(), cfg.Training.Seed)
	if err != nil {
		return nil, err
	}
	if startStep > 0 {
		if err = producer.Restore(consumed.Snapshot(), startStep, cfg.Training.BatchSize); err != nil {
			return nil, err
		}
	}
	s := cfg.Training.GridLoader
	opts := data.GridPrefetchOptions{BatchSize: cfg.Training.BatchSize, Depth: s.EffectivePrefetchBatches(),
		Workers: s.EffectiveReadWorkers(cfg.Training.BatchSize), Steps: max(0, steps-startStep), Seed: cfg.Training.Seed,
		Dihedral: cfg.Training.GridAugmentation != nil && cfg.Training.GridAugmentation.Dihedral}
	p, err := data.NewGridPrefetch(ds, producer, opts)
	if err != nil {
		return nil, err
	}
	decoded, scratch := data.GridLoaderMemoryBounds(ds.Geometry, ds.DType, opts)
	fmt.Printf("  [%s] grid loader: prefetch_batches=%d read_workers=%d decoded_pool=%.1fMiB scratch_bound=%.1fMiB (excludes index metadata and page cache)\n", cfg.Name, opts.Depth, opts.Workers, float64(decoded)/(1<<20), float64(scratch)/(1<<20))
	return p, nil
}

type gridThroughput struct {
	compute, active, wait time.Duration
	records, allRecords   int
}

func (t *gridThroughput) observe(count int, wait, compute, active time.Duration, warmup bool) {
	t.allRecords += count
	if warmup {
		return
	}
	t.records += count
	t.compute += compute
	t.active += active
	t.wait += wait
}

func (t gridThroughput) rates(elapsed time.Duration) (compute, training, wall, waitPercent float64) {
	if t.compute > 0 {
		compute = float64(t.records) / t.compute.Seconds()
	}
	if t.active > 0 {
		training = float64(t.records) / t.active.Seconds()
		waitPercent = 100 * t.wait.Seconds() / t.active.Seconds()
	}
	if elapsed > 0 {
		wall = float64(t.allRecords) / elapsed.Seconds()
	}
	return
}
