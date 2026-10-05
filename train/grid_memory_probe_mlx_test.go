//go:build mlx && cgo && (darwin || linux)

package train

import (
	"fmt"
	"math"
	"os"
	"runtime"
	"strconv"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/data"
	"github.com/mrothroc/mixlab/gpu"
)

// Explicit full-size acceptance probe, not part of ordinary unit-test runs.
func TestGridMemoryProbe(t *testing.T) {
	if os.Getenv("GRID_MEMORY_PROBE") != "1" {
		t.Skip("set GRID_MEMORY_PROBE=1 for full-size GPU acceptance")
	}
	if !mlxAvailable() {
		t.Skip("MLX required")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	stage := os.Getenv("GRID_MEMORY_STAGE")
	if stage == "" {
		stage = "1"
	}
	if stage != "1" && stage != "2" {
		t.Fatal("GRID_MEMORY_STAGE must be 1 or 2")
	}
	steps := 200
	if s := os.Getenv("GRID_MEMORY_STEPS"); s != "" {
		var err error
		steps, err = strconv.Atoi(s)
		if err != nil || steps < 1 {
			t.Fatal("GRID_MEMORY_STEPS must be positive")
		}
	}
	cfg, err := LoadArchConfig(fmt.Sprintf("../examples/grid_unet_reference/two_channel_stage%s.json", stage))
	if err != nil {
		t.Fatal(err)
	}
	const pixels = 256 * 256
	batchSize := 16
	expectOOM := os.Getenv("GRID_MEMORY_EXPECT_OOM") == "1"
	if expectOOM {
		batchSize = 128
	}
	cfg.Training.BatchSize = batchSize
	prog, err := arch.BuildGridIRProgram(cfg, true)
	if err != nil {
		t.Fatal(err)
	}
	gpu.ApplyCUDAGraphLimits(gpu.TuneCUDAGraphLimits(prog))
	memoryPlan, err := configureMLXMemoryLimits(cfg.Name)
	if err != nil {
		t.Fatal(err)
	}
	if expectOOM && (!memoryPlan.DedicatedDevice || memoryPlan.DeviceMemoryBytes > 32<<30) {
		t.Skip("oversized OOM probe requires a dedicated CUDA GPU with at most 32 GiB")
	}
	logMemory := func(phase string) {
		m := gpu.MemoryStatsSnapshot()
		d, _ := gpu.DeviceMemoryInfo()
		t.Logf("grid-memory stage=%s phase=%s active=%d cache=%d peak=%d device_free=%d", stage, phase, m.ActiveBytes, m.CacheBytes, m.PeakBytes, d.FreeBytes)
		t.Log(gridMemoryDiagnostic(phase, memoryPlan))
	}
	tr, err := initGPUTrainer(prog, cfg, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer tr.CloseTrainer()
	logMemory("initialized")
	b := data.GridBatch{BatchSize: batchSize, Count: batchSize,
		Inputs: make([]float32, batchSize*pixels*2), Targets: make([]float32, batchSize*pixels), LossMask: make([]float32, batchSize*pixels)}
	for i := range b.Targets {
		b.Inputs[2*i] = float32(i%251) / 250
		b.Inputs[2*i+1] = float32((i/256)%127) / 126
		b.Targets[i] = .3*b.Inputs[2*i] + .2*b.Inputs[2*i+1]
		b.LossMask[i] = 1
	}
	preflight, err := gridMemoryPreflightEnabled(memoryPlan.DedicatedDevice)
	if err != nil {
		t.Fatal(err)
	}
	if expectOOM && !preflight {
		t.Fatal("oversized OOM probe requires preflight enabled")
	}
	if preflight {
		shapes, err := computeWeightShapes(cfg)
		if err != nil {
			t.Fatal(err)
		}
		if err := preflightGridTraining(tr, shapes, b, 0, 1e-4); err != nil {
			if expectOOM && isMLXOutOfMemoryError(err) {
				t.Log(annotateGridMemoryError(err, "preflight before full validation", cfg, memoryPlan))
				return
			}
			t.Fatal(annotateGridMemoryError(err, "preflight before full validation", cfg, memoryPlan))
		}
		if expectOOM {
			t.Fatal("oversized probe unexpectedly fit; choose a lower-memory test GPU")
		}
		logMemory("preflight-restored")
	}
	validate := func() {
		loss, err := tr.EvaluateObjectiveGPU(objectiveBatch{grid: &b}, batchSize, 0)
		if err != nil || math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
			t.Fatalf("validation loss=%g: %v", loss, err)
		}
		logMemory("validation")
	}
	validate()
	var elapsed time.Duration
	for step := 0; step < steps; step++ {
		start := time.Now()
		if err := submitPreparedStepGPU(tr, objectiveBatch{grid: &b}, batchSize, 0, 1e-4); err != nil {
			t.Fatalf("step %d: %v", step, err)
		}
		loss, err := tr.CollectLossGPU()
		if err != nil || math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
			t.Fatalf("step %d loss=%g: %v", step, loss, err)
		}
		if step > 2 {
			elapsed += time.Since(start)
		}
		if step < 10 || (step+1)%50 == 0 {
			t.Logf("grid-loss stage=%s step=%d loss=%.9g", stage, step+1, loss)
			logMemory("training")
		}
		if (step+1)%100 == 0 {
			validate()
		}
	}
	if steps > 3 {
		t.Logf("grid-throughput stage=%s records/s=%.4f", stage, float64((steps-3)*batchSize)/elapsed.Seconds())
	}
}
