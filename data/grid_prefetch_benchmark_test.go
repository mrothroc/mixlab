package data

import (
	"fmt"
	"runtime"
	"testing"
	"time"
)

// Local synthetic float16 reads/decodes/D4 with a sleeping consumer. The sleep
// models overlapping work, NOT measured GPU throughput or network latency.
func BenchmarkGridPrefetchOverlap(b *testing.B) {
	d := gridReaderFixtureGeometry(b, 2, GridGeometry{Channels: 7, TargetChannels: 1, Height: 256, Width: 256}, 17)
	for _, tc := range []struct{ depth, workers int }{{0, 1}, {0, min(16, runtime.GOMAXPROCS(0))}, {2, min(16, runtime.GOMAXPROCS(0))}} {
		b.Run(fmt.Sprintf("depth%d-workers%d", tc.depth, tc.workers), func(b *testing.B) {
			s, _ := NewGridSampler(d.Len(), 42)
			p, err := NewGridPrefetch(d, s, GridPrefetchOptions{BatchSize: 16, Depth: tc.depth, Workers: tc.workers, Steps: b.N, Seed: 42, Dihedral: true})
			if err != nil {
				b.Fatal(err)
			}
			defer func() { _ = p.Close() }()
			b.ResetTimer()
			start := time.Now()
			var wait, compute time.Duration
			records := 0
			for i := 0; i < b.N; i++ {
				tick := time.Now()
				batch, _, err := p.Next()
				if err != nil {
					b.Fatal(err)
				}
				wait += time.Since(tick)
				tick = time.Now()
				time.Sleep(20 * time.Millisecond)
				compute += time.Since(tick)
				records += batch.Count
			}
			elapsed := time.Since(start)
			b.StopTimer()
			b.ReportMetric(float64(records)/elapsed.Seconds(), "wall-records/s")
			b.ReportMetric(float64(records)/compute.Seconds(), "sim-compute-records/s")
			b.ReportMetric(100*wait.Seconds()/elapsed.Seconds(), "data-wait-%")
		})
	}
}
