//go:build mlx && cgo && (darwin || linux)

package train

import (
	"math"
	"runtime"
	"testing"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/data"
	"github.com/mrothroc/mixlab/gpu"
)

// Run explicitly with -run '^$' -bench '^BenchmarkGridReferenceTraining$'
// -benchtime=12x. The shipped full-resolution graph needs several GiB at batch 15.
func BenchmarkGridReferenceTraining(b *testing.B) {
	if !mlxAvailable() {
		b.Skip("MLX required")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	cfg, err := LoadArchConfig("../examples/grid_unet_reference/two_channel_stage1.json")
	if err != nil {
		b.Fatal(err)
	}
	const batchSize, pixels = 15, 256 * 256
	cfg.Training.BatchSize = batchSize
	p, err := arch.BuildGridIRProgram(cfg, true)
	if err != nil {
		b.Fatal(err)
	}
	tr, err := initGPUTrainer(p, cfg, nil, nil)
	if err != nil {
		b.Fatal(err)
	}
	defer tr.CloseTrainer()
	batch := data.GridBatch{
		BatchSize: batchSize, Count: batchSize,
		Inputs:  make([]float32, batchSize*pixels*2),
		Targets: make([]float32, batchSize*pixels), LossMask: make([]float32, batchSize*pixels),
	}
	for i := range batch.Targets {
		batch.Inputs[2*i] = float32(i%251) / 250
		batch.Inputs[2*i+1] = float32((i/256)%127) / 126
		batch.Targets[i] = .3*batch.Inputs[2*i] + .2*batch.Inputs[2*i+1]
		batch.LossMask[i] = 1
	}
	step := func() {
		if err := submitPreparedStepGPU(tr, objectiveBatch{grid: &batch}, batchSize, 0, 1e-4); err != nil {
			b.Fatal(err)
		}
		loss, err := tr.CollectLossGPU()
		if err != nil || math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
			b.Fatalf("loss=%g error=%v", loss, err)
		}
	}
	for i := 0; i < 3; i++ {
		step()
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		step()
	}
	b.StopTimer()
	b.ReportMetric(float64(b.N*batchSize)/b.Elapsed().Seconds(), "records/s")
	b.ReportMetric(float64(gpu.MemoryStatsSnapshot().PeakBytes)/(1<<30), "peak-GiB")
}
