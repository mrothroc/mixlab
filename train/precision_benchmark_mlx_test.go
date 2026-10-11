//go:build mlx && cgo && (darwin || linux)

package train

import (
	"math"
	"os"
	"runtime"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/data"
	"github.com/mrothroc/mixlab/gpu"
)

// Separate invocations with MIXLAB_TEST_GRID_TF32=0 and =1 compare exactly the
// same stage-1 graph and seeded weights, without including loader IO or warmup.
func TestGridTF32Throughput(t *testing.T) {
	mode := os.Getenv("MIXLAB_TEST_GRID_TF32")
	if mode == "" {
		t.Skip("explicit full-resolution CUDA benchmark")
	}
	if gpu.PrecisionBackend() != "cuda" {
		t.Skip("CUDA benchmark")
	}
	if mode != "0" && mode != "1" {
		t.Fatal("MIXLAB_TEST_GRID_TF32 must be 0 or 1")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	cfg, err := LoadArchConfig("../examples/grid_unet_reference/two_channel_stage1.json")
	if err != nil {
		t.Fatal(err)
	}
	allow := mode == "1"
	cfg.Training.AllowTF32 = &allow
	cfg.Training.BatchSize = 16
	policy, err := trainingPrecision(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Log(policy.String())
	p, err := arch.BuildGridIRProgram(cfg, true)
	if err != nil {
		t.Fatal(err)
	}
	gpu.ApplyCUDAGraphLimits(gpu.TuneCUDAGraphLimits(p))
	if _, err = configureMLXMemoryLimits(cfg.Name); err != nil {
		t.Fatal(err)
	}
	tr, err := initGPUTrainer(p, cfg, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer tr.CloseTrainer()
	const batch, pixels, warmup, steps = 16, 256 * 256, 5, 30
	b := data.GridBatch{BatchSize: batch, Count: batch, Inputs: make([]float32, batch*pixels*2), Targets: make([]float32, batch*pixels), LossMask: make([]float32, batch*pixels)}
	for i := range b.Targets {
		b.Inputs[2*i] = float32(i%251) / 250
		b.Inputs[2*i+1] = float32((i/256)%127) / 126
		b.Targets[i] = .3*b.Inputs[2*i] + .2*b.Inputs[2*i+1]
		b.LossMask[i] = 1
	}
	var elapsed time.Duration
	for i := 0; i < steps; i++ {
		start := time.Now()
		if err = submitPreparedStepGPU(tr, objectiveBatch{grid: &b}, batch, 0, 1e-4); err != nil {
			t.Fatal(err)
		}
		loss, e := tr.CollectLossGPU()
		if e != nil || math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
			t.Fatalf("step%d loss=%g: %v", i, loss, e)
		}
		if i >= warmup {
			elapsed += time.Since(start)
		}
	}
	t.Logf("grid-tf32 allow=%t batch=%d measured_steps=%d records/s=%.4f elapsed=%s", allow, batch, steps-warmup, float64((steps-warmup)*batch)/elapsed.Seconds(), elapsed)
}
