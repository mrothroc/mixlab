//go:build mlx && cgo && (darwin || linux)

package train

import (
	"fmt"
	"math"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/gpu"
)

// TestTTTMLPRealScaleTrainingPerformance is an opt-in release gate. It locks
// the D384/T512 shape that exposed per-step diagnostic graph retention without
// adding a multi-minute test to ordinary CI.
func TestTTTMLPRealScaleTrainingPerformance(t *testing.T) {
	if os.Getenv("MIXLAB_TTT_MLP_PERF_PROBE") != "1" {
		t.Skip("set MIXLAB_TTT_MLP_PERF_PROBE=1 to run the real-scale training gate")
	}
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}
	previousCacheLimit := gpu.SetMemoryCacheLimit(8 << 30)
	defer gpu.SetMemoryCacheLimit(previousCacheLimit)

	deltaDuration := runTTTMLPPerformanceVariant(t, "gated_deltanet")
	tttDuration := runTTTMLPPerformanceVariant(t, "ttt_mlp")
	ratio := float64(tttDuration) / float64(deltaDuration)
	t.Logf("steady training: ttt_mlp=%s gated_deltanet=%s ratio=%.2fx", tttDuration, deltaDuration, ratio)
	if ratio > 3.0 {
		t.Fatalf("TTT-MLP steady training ratio %.2fx exceeds 3.0x release gate", ratio)
	}
}

func runTTTMLPPerformanceVariant(t *testing.T, mixer string) time.Duration {
	t.Helper()
	const steps = 15
	cfg := parseTTTMLPPerformanceConfig(t, mixer, steps)
	prog, err := BuildTrainingIRProgramFromConfig(cfg, arch.TrainingProgramState{Objective: arch.ObjectiveCausal})
	if err != nil {
		t.Fatalf("build %s program: %v", mixer, err)
	}
	trainerIface, err := initGPUTrainer(prog, cfg, nil, nil)
	if err != nil {
		t.Fatalf("init %s trainer: %v", mixer, err)
	}
	defer trainerIface.CloseTrainer()
	if err := enableTrainingStepComponentLossCapture(trainerIface); err != nil {
		t.Fatalf("enable %s component telemetry: %v", mixer, err)
	}

	raw := trainBatch{x: make([]int, cfg.Training.BatchTokens), y: make([]int, cfg.Training.BatchTokens)}
	for i := range raw.x {
		raw.x[i] = i % cfg.VocabSize
		raw.y[i] = (i + 1) % cfg.VocabSize
	}
	var steadyStart time.Time
	for step := 0; step < steps; step++ {
		batch, err := prepareObjectiveBatch(cfg, raw, step, arch.ObjectiveCausal)
		if err != nil {
			t.Fatalf("prepare %s step %d: %v", mixer, step, err)
		}
		if err := submitPreparedStepGPU(trainerIface, batch, cfg.Training.BatchTokens/cfg.SeqLen, cfg.SeqLen, float32(cfg.Training.LR)); err != nil {
			t.Fatalf("submit %s step %d: %v", mixer, step, err)
		}
		loss, err := trainerIface.CollectLossGPU()
		if err != nil {
			t.Fatalf("collect %s step %d: %v", mixer, step, err)
		}
		if !(loss > 0) || math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
			t.Fatalf("%s step %d loss=%g", mixer, step, loss)
		}
		if step == 0 {
			steadyStart = time.Now()
		}
	}
	return time.Since(steadyStart)
}

func parseTTTMLPPerformanceConfig(t *testing.T, mixer string, steps int) *ArchConfig {
	t.Helper()
	var mixerJSON string
	switch mixer {
	case "ttt_mlp":
		mixerJSON = `{"type":"ttt_mlp","heads":6,"chunk_size":16}`
	case "gated_deltanet":
		mixerJSON = `{"type":"gated_deltanet","heads":6,"d_k":64,"d_v":128,"scan_chunk_size":64}`
	default:
		t.Fatalf("unknown mixer %q", mixer)
	}
	blocks := make([]string, 0, 28)
	for i := 0; i < 14; i++ {
		blocks = append(blocks, mixerJSON, `{"type":"swiglu"}`)
	}
	configJSON := fmt.Sprintf(`{
		"name":"ttt_mlp_perf_%s",
		"model_dim":384,"vocab_size":16384,"seq_len":512,
		"blocks":[%s],
		"training":{"objective":"causal","optimizer":"adamw","steps":%d,
			"lr":0.0001,"seed":19,"batch_tokens":8192,"grad_clip":1.0,"weight_decay":0.0}
	}`, mixer, strings.Join(blocks, ","), steps)
	cfg, err := ParseArchConfig([]byte(configJSON), "ttt_mlp_training_perf")
	if err != nil {
		t.Fatalf("parse %s config: %v", mixer, err)
	}
	return cfg
}
