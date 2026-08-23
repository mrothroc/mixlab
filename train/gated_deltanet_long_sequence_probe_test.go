//go:build mlx && cgo && (darwin || linux)

package train

import (
	"fmt"
	"math"
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/gpu"
)

// TestGatedDeltaNetLongSequenceMemoryProbe is opt-in because its production
// shape intentionally exercises tens of thousands of timesteps. Keep it as a
// stable reproduction for memory and throughput regressions.
func TestGatedDeltaNetLongSequenceMemoryProbe(t *testing.T) {
	if os.Getenv("MIXLAB_GATED_DELTA_LONG_PROBE") != "1" {
		t.Skip("set MIXLAB_GATED_DELTA_LONG_PROBE=1 to run the long-sequence probe")
	}
	if !gpu.Available() {
		t.Skip("MLX backend not available")
	}

	seqLen := gatedDeltaProbeInt("MIXLAB_GATED_DELTA_PROBE_T", 16000)
	batchSize := gatedDeltaProbeInt("MIXLAB_GATED_DELTA_PROBE_BATCH", 4)
	modelDim := gatedDeltaProbeInt("MIXLAB_GATED_DELTA_PROBE_D", 128)
	layers := gatedDeltaProbeInt("MIXLAB_GATED_DELTA_PROBE_LAYERS", 6)
	heads := gatedDeltaProbeInt("MIXLAB_GATED_DELTA_PROBE_HEADS", 4)
	dk := gatedDeltaProbeInt("MIXLAB_GATED_DELTA_PROBE_DK", 16)
	dv := gatedDeltaProbeInt("MIXLAB_GATED_DELTA_PROBE_DV", 32)
	chunkSize := gatedDeltaProbeInt("MIXLAB_GATED_DELTA_PROBE_CHUNK", 64)
	mode := os.Getenv("MIXLAB_GATED_DELTA_PROBE_MODE")
	if mode == "" {
		mode = "train"
	}
	if mode != "eval" && mode != "train" {
		t.Fatalf("MIXLAB_GATED_DELTA_PROBE_MODE=%q, want eval or train", mode)
	}

	blocks := ""
	for layer := 0; layer < layers; layer++ {
		if layer > 0 {
			blocks += ","
		}
		blocks += fmt.Sprintf(
			`{"type":"gated_deltanet","heads":%d,"d_k":%d,"d_v":%d,"scan_chunk_size":%d,"bidirectional":true}`,
			heads, dk, dv, chunkSize)
	}
	raw := fmt.Sprintf(`{
		"name":"gated_deltanet_long_sequence_probe",
		"model_dim":%d,
		"seq_len":%d,
		"positional_embedding":"none",
		"norm_type":"batchnorm",
		"norm_placement":"pre",
		"final_norm":true,
		"input_adapter":{"kind":"linear_frames","feature_dim":1,"bias":true,"norm":"none"},
		"blocks":[%s],
		"training":{
			"objective":"classification",
			"classification":{"num_labels":35,"pooling":"mean","classifier_dropout":0},
			"optimizer":"adamw",
			"batch_tokens":%d,
			"steps":1,
			"lr":0.0001,
			"grad_clip":1,
			"weight_decay":0,
			"seed":37
		}
	}`, modelDim, seqLen, blocks, batchSize*seqLen)
	cfg, err := ParseArchConfig([]byte(raw), t.Name())
	if err != nil {
		t.Fatal(err)
	}
	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	trainerInterface, err := initGPUTrainer(prog, cfg, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	trainer := trainerInterface.(*mlxGPUTrainer)
	defer trainer.CloseTrainer()

	rows := batchSize * seqLen
	frames := make([]float32, rows)
	validMask := make([]float32, rows)
	positions := make([]int32, batchSize)
	labels := make([]int32, batchSize)
	for row := 0; row < batchSize; row++ {
		positions[row] = int32(seqLen - 1)
		labels[row] = int32(row % 2)
		for pos := 0; pos < seqLen; pos++ {
			frames[row*seqLen+pos] = float32(math.Sin(float64(pos+row) * 0.01))
			validMask[row*seqLen+pos] = 1
		}
	}
	batch := objectiveBatch{
		frames:               frames,
		classificationLabels: labels,
		classificationMask:   validMask,
		classificationPos:    positions,
	}

	gpu.ClearMemoryCache()
	start := time.Now()
	var loss float32
	if mode == "eval" {
		loss, err = trainer.EvaluateObjectiveGPU(batch, batchSize, seqLen)
	} else {
		loss, err = trainer.TrainObjectiveStepGPU(batch, batchSize, seqLen, float32(cfg.Training.LR))
	}
	if err != nil {
		t.Fatal(err)
	}
	if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
		t.Fatalf("non-finite %s loss=%g", mode, loss)
	}
	memory := gpu.MemoryStatsSnapshot()
	elapsed := time.Since(start)
	tokensPerSecond := float64(rows) / elapsed.Seconds()
	t.Logf(
		"mode=%s B=%d T=%d D=%d layers=%d H=%d Dk=%d Dv=%d chunk=%d loss=%g elapsed=%s tok/s=%.0f active_mib=%.2f cache_mib=%.2f peak_mib=%.2f",
		mode, batchSize, seqLen, modelDim, layers, heads, dk, dv, chunkSize, loss,
		elapsed, tokensPerSecond, float64(memory.ActiveBytes)/(1<<20),
		float64(memory.CacheBytes)/(1<<20), float64(memory.PeakBytes)/(1<<20))
}

func gatedDeltaProbeInt(name string, fallback int) int {
	raw := os.Getenv(name)
	if raw == "" {
		return fallback
	}
	value, err := strconv.Atoi(raw)
	if err != nil || value <= 0 {
		return fallback
	}
	return value
}
