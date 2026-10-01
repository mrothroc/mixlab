package data

import (
	"os"
	"testing"
)

// Measures real shard reads, finite checks and CHW->NHWC conversion, separately
// from GPU compute. Repeated access normally benefits from the OS page cache.
func BenchmarkGridShardIO(b *testing.B) {
	path := os.Getenv("GRID_BENCH_MANIFEST")
	if path == "" {
		b.Skip("set GRID_BENCH_MANIFEST to a prepared grid manifest")
	}
	d, err := OpenGridDataset(path, "train")
	if err != nil {
		b.Fatal(err)
	}
	defer func() { _ = d.Close() }()
	if _, err = d.ReadBatch([]int{0}, 1); err != nil {
		b.Fatal(err)
	}
	g := d.Geometry
	bytesPerValue := 4
	if d.DType == "float16" {
		bytesPerValue = 2
	}
	b.SetBytes(int64(g.Height*g.Width*(g.Channels+g.TargetChannels)*bytesPerValue + (g.Height*g.Width+7)/8))
	b.ReportAllocs()
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		if _, err = d.ReadBatch([]int{n % d.Len()}, 1); err != nil {
			b.Fatal(err)
		}
	}
}
