package train

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/mrothroc/mixlab/gpu"
)

// runSmoke performs GPU diagnostics: checks backend availability, validates the
// IR training pipeline with a tiny model, and benchmarks IR evaluation.
func runSmoke() error {
	fmt.Println("=== SMOKE: GPU Diagnostics ===")

	// --- Step 1: Backend availability ---
	fmt.Println("\n[1/2] Checking MLX backend...")
	if !gpu.Available() {
		fmt.Println("FAIL: MLX backend is not available.")
		fmt.Println("  -> Rebuild with: CGO_ENABLED=1 go build -tags mlx -o mixlab ./cmd/mixlab")
		fmt.Println("  -> Ensure MLX is installed (pip install mlx)")
		return fmt.Errorf("MLX backend unavailable")
	}
	devName := gpu.DeviceName()
	fmt.Printf("PASS: MLX backend available, device: %s\n", devName)

	// --- Step 2: IR health check + benchmark ---
	fmt.Println("\n[2/2] IR training + benchmark...")
	if err := runIRHealthCheck(); err != nil {
		fmt.Printf("FAIL: IR health check: %v\n", err)
		return fmt.Errorf("IR health check failed: %w", err)
	}

	fmt.Println("\n=== SMOKE: All checks passed ===")
	return nil
}

// runIRHealthCheck builds a tiny 2-block model, creates a trainer, runs 1
// training step on random data, verifies the loss is finite, then benchmarks
// evaluation throughput on the same IR program.
func runIRHealthCheck() error {
	// Parse a minimal config
	cfgJSON := []byte(`{
		"name": "smoke_check",
		"model_dim": 32,
		"vocab_size": 64,
		"seq_len": 16,
		"blocks": [
			{"type": "plain", "heads": 2},
			{"type": "swiglu"}
		],
		"training": {"steps": 1, "lr": 3e-4, "seed": 1, "batch_tokens": 32, "grad_clip": 1.0, "weight_decay": 0.01}
	}`)

	cfg, err := ParseArchConfig(cfgJSON, "smoke")
	if err != nil {
		return fmt.Errorf("parse config: %w", err)
	}

	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		return fmt.Errorf("build IR program: %w", err)
	}
	fmt.Printf("  IR program: %d weights, %d ops\n", prog.NumWeights, len(prog.Ops))

	// Create GPU trainer (nil loadedWeights = Xavier init from scratch)
	trainer, err := initGPUTrainer(prog, cfg, nil)
	if err != nil {
		return fmt.Errorf("init GPU trainer: %w", err)
	}
	defer trainer.CloseTrainer()

	// Generate random token data
	batchSize := cfg.Training.BatchTokens / cfg.SeqLen
	if batchSize <= 0 {
		batchSize = 1
	}
	seqLen := cfg.SeqLen
	nTokens := batchSize * seqLen
	rng := rand.New(rand.NewSource(42))

	xTok := make([]int, nTokens)
	yTok := make([]int, nTokens)
	for i := 0; i < nTokens; i++ {
		xTok[i] = rng.Intn(cfg.VocabSize)
		yTok[i] = rng.Intn(cfg.VocabSize)
	}

	// Run one training step
	loss, err := trainer.TrainStepGPU(xTok, yTok, batchSize, seqLen, float32(cfg.Training.LR))
	if err != nil {
		return fmt.Errorf("training step: %w", err)
	}

	if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
		return fmt.Errorf("training produced non-finite loss: %v", loss)
	}

	fmt.Printf("  Training check: loss=%.4f (finite, OK)\n", loss)

	const nEvalIter = 100
	start := time.Now()
	for i := 0; i < nEvalIter; i++ {
		evalLoss, err := trainer.EvaluateGPU(xTok, yTok, batchSize, seqLen)
		if err != nil {
			return fmt.Errorf("evaluation iter %d: %w", i, err)
		}
		if math.IsNaN(float64(evalLoss)) || math.IsInf(float64(evalLoss), 0) {
			return fmt.Errorf("evaluation iter %d produced non-finite loss: %v", i, evalLoss)
		}
	}
	elapsed := time.Since(start)
	iterPerSec := float64(nEvalIter) / elapsed.Seconds()
	fmt.Printf("  Benchmark: %d eval iterations in %.2fs (%.1f iter/s)\n",
		nEvalIter, elapsed.Seconds(), iterPerSec)
	return nil
}
