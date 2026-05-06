//go:build mlx

package train

import (
	"math"
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/gpu"
)

func TestMamba3CanonicalFullStepOOMProbe(t *testing.T) {
	if os.Getenv("MIXLAB_MAMBA3_FULL_OOM_PROBE") != "1" {
		t.Skip("set MIXLAB_MAMBA3_FULL_OOM_PROBE=1 to run the full Mamba3 OOM probe")
	}
	if !gpu.Available() {
		t.Skip("MLX backend not available")
	}

	d := mamba3ProbeInt("MIXLAB_MAMBA3_PROBE_D", 448)
	seqLen := mamba3ProbeInt("MIXLAB_MAMBA3_PROBE_T", 4096)
	layers := mamba3ProbeInt("MIXLAB_MAMBA3_PROBE_LAYERS", 8)
	chunkSize := mamba3ProbeInt("MIXLAB_MAMBA3_PROBE_CHUNK", 64)

	blocks := make([]BlockSpec, 0, layers*2)
	for i := 0; i < layers; i++ {
		blocks = append(blocks, BlockSpec{Type: "mamba3-canonical", ScanChunkSize: &chunkSize})
		blocks = append(blocks, BlockSpec{Type: "swiglu"})
	}
	cfg := &ArchConfig{
		Name:          "mamba3_full_oom_probe",
		ModelDim:      d,
		VocabSize:     1024,
		SeqLen:        seqLen,
		TieEmbeddings: false,
		Blocks:        blocks,
		Training: TrainingSpec{
			Steps:             1,
			LR:                3e-4,
			EmbedLR:           3e-4,
			MatrixLR:          3e-4,
			ScalarLR:          3e-4,
			HeadLR:            3e-4,
			Beta1:             0.9,
			Beta2:             0.95,
			Epsilon:           1e-8,
			BatchTokens:       seqLen,
			Seed:              7,
			GradClip:          1,
			WeightDecay:       0.02,
			MuonMomentum:      0.9,
			MuonBackendSteps:  5,
			WeightInit:        "normal",
			WeightInitStd:     0.02,
			EmbedWeightDecay:  0.02,
			MatrixWeightDecay: 0.02,
			ScalarWeightDecay: 0.02,
			HeadWeightDecay:   0.02,
		},
	}

	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig: %v", err)
	}
	t.Logf("IR program: %d ops, %d weights", len(prog.Ops), prog.NumWeights)

	trainer, err := initGPUTrainer(prog, cfg, nil, nil)
	if err != nil {
		t.Fatalf("initGPUTrainer: %v", err)
	}
	defer trainer.CloseTrainer()

	xTok := make([]int, seqLen)
	yTok := make([]int, seqLen)
	for i := 0; i < seqLen; i++ {
		xTok[i] = i % cfg.VocabSize
		yTok[i] = (i + 1) % cfg.VocabSize
	}

	start := time.Now()
	if err := trainer.SubmitStepGPU(xTok, yTok, 1, seqLen, float32(cfg.Training.LR)); err != nil {
		t.Fatalf("SubmitStepGPU: %v", err)
	}
	loss, err := trainer.CollectLossGPU()
	if err != nil {
		t.Fatalf("CollectLossGPU: %v", err)
	}
	if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
		t.Fatalf("non-finite loss %g", loss)
	}
	t.Logf("loss=%g elapsed=%s", loss, time.Since(start))
}

func mamba3ProbeInt(name string, fallback int) int {
	raw := os.Getenv(name)
	if raw == "" {
		return fallback
	}
	v, err := strconv.Atoi(raw)
	if err != nil || v <= 0 {
		return fallback
	}
	return v
}
