package arch

import (
	"fmt"
	"runtime"
)

// GridLoaderSpec changes host scheduling only, never the sampled training task.
type GridLoaderSpec struct {
	PrefetchBatches *int `json:"prefetch_batches,omitempty"`
	ReadWorkers     int  `json:"read_workers,omitempty"`
}

func (s *GridLoaderSpec) EffectivePrefetchBatches() int {
	if s == nil || s.PrefetchBatches == nil {
		return 2
	}
	return *s.PrefetchBatches
}

func (s *GridLoaderSpec) EffectiveReadWorkers(batchSize int) int {
	n := runtime.GOMAXPROCS(0)
	if s != nil && s.ReadWorkers > 0 {
		n = s.ReadWorkers
	}
	return max(1, min(batchSize, n))
}

func (s *GridLoaderSpec) validate() error {
	if s.EffectivePrefetchBatches() < 0 || s.EffectivePrefetchBatches() > 64 {
		return fmt.Errorf("grid_loader.prefetch_batches must be in [0,64]")
	}
	if s != nil && (s.ReadWorkers < 0 || s.ReadWorkers > 256) {
		return fmt.Errorf("grid_loader.read_workers must be in [0,256] (0 is automatic)")
	}
	return nil
}
