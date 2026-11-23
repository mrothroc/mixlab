//go:build mlx && cgo && (darwin || linux)

package train

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"runtime"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/gpu"
)

// One config per process: allocator peaks include initialization and warmup.
// The runner, not this test, owns benchmark configs and resource admission.
func TestGridTrainingBenchmark(t *testing.T) {
	path := os.Getenv("MIXLAB_GRID_BENCH_CONFIG")
	if path == "" {
		t.Skip("explicit synthetic grid benchmark; set MIXLAB_GRID_BENCH_CONFIG")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := LoadArchConfig(path)
	if err != nil {
		t.Fatal(err)
	}
	if !cfg.GridEnabled() || cfg.Training.EffectiveComputeDType() != "float32" || cfg.Training.AllowTF32 == nil || *cfg.Training.AllowTF32 || cfg.Training.Optimizer != "adamw" {
		t.Fatal("benchmark requires an explicit strict-FP32 AdamW grid config")
	}
	p, err := arch.BuildGridIRProgram(cfg, true)
	if err != nil {
		t.Fatal(err)
	}
	gpu.ApplyCUDAGraphLimits(gpu.TuneCUDAGraphLimits(p))
	if _, err := gpu.ConfigurePrecision("float32", cfg.Training.AllowTF32); err != nil {
		t.Fatal(err)
	}
	if !mlxAvailable() {
		t.Fatal("MLX required for an explicitly requested benchmark")
	}
	hash := sha256.Sum256(raw)
	report := map[string]any{"format": "mixlab.grid_benchmark.v1", "config_sha256": hex.EncodeToString(hash[:]),
		"name": cfg.Name, "status": "error", "warmup": 5, "measured_steps": 20, "seed": cfg.Training.Seed,
		"synthetic_seed": 412, "batch": cfg.Training.BatchSize, "height": cfg.InputAdapter.Height, "width": cfg.InputAdapter.Width,
		"channels": cfg.InputAdapter.Channels, "device": gpu.DeviceName(), "backend": gpu.PrecisionBackend(),
		"memory_scope": "MLX process lifetime including initialization and warmup; excludes host batch buffers", "coordinate_cache": false}
	defer func() {
		if _, ok := report["memory"]; !ok {
			report["memory"] = gpu.MemoryStatsSnapshot()
		}
		encoded, e := json.Marshal(report)
		if e != nil {
			t.Errorf("benchmark report: %v", e)
			return
		}
		fmt.Printf("GRID_BENCHMARK %s\n", encoded)
	}()
	fail := func(e error) {
		report["error"] = e.Error()
		if isMLXOutOfMemoryError(e) {
			report["status"] = "oom"
		}
		t.Fatal(e)
	}
	limits, err := configureMLXMemoryLimits(cfg.Name)
	if err != nil {
		fail(err)
	}
	report["memory_limits"] = limits
	report["device"] = limits.DeviceName
	var baseline *ArchConfig
	if base := os.Getenv("MIXLAB_GRID_BENCH_BASELINE"); base != "" {
		baseline, err = LoadArchConfig(base)
		if err != nil {
			fail(err)
		}
		baseHash, e := torchStateFileHash(base)
		if e != nil {
			fail(e)
		}
		report["baseline_sha256"] = baseHash
	}
	weights, matched, err := gridBenchmarkWeights(cfg, baseline)
	if err != nil {
		fail(err)
	}
	report["matched_baseline_tensors"] = matched
	tr, err := initGPUTrainer(p, cfg, weights, nil)
	if err != nil {
		fail(err)
	}
	defer func() {
		report["memory"] = gpu.MemoryStatsSnapshot()
		tr.CloseTrainer()
	}()
	b := gridBenchmarkBatch(cfg)
	checks := arch.GridPolarInputChecks(p)
	var samples []float64
	for step := 0; step < 25; step++ {
		// Vary geometry outside the timed section, defeating record-plan reuse.
		for n := 0; n < b.BatchSize; n++ {
			for _, c := range checks {
				b.Vectors[n*cfg.InputAdapter.VectorDim+c.RowColumn] = float32((n*17+step)%97) / 97 * float32(c.Height-1)
				b.Vectors[n*cfg.InputAdapter.VectorDim+c.ColColumn] = float32((n*31+step)%89) / 89 * float32(c.Width-1)
			}
		}
		start := time.Now()
		if err = submitPreparedStepGPU(tr, objectiveBatch{grid: &b}, b.BatchSize, cfg.SeqLen, float32(cfg.Training.LR)); err != nil {
			fail(err)
		}
		loss, e := tr.CollectLossGPU() // synchronizes the complete optimizer step
		if e != nil {
			fail(e)
		}
		seconds := time.Since(start).Seconds()
		if !isFiniteLoss(float64(loss)) {
			fail(fmt.Errorf("step %d nonfinite loss", step))
		}
		if step >= 5 {
			samples = append(samples, seconds)
		}
		report["completed_steps"] = step + 1
		report["last_loss"] = loss
		t.Logf("step=%d seconds=%.6f loss=%g", step, seconds, loss)
	}
	report["samples_seconds"], report["median_seconds"], report["status"] = samples, gridBenchmarkMedian(samples), "pass"
	report["compile_stats"] = formatCompileStats(tr)
}
