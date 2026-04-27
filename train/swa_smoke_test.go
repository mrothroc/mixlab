//go:build mlx && cgo && (darwin || linux)

package train

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/mrothroc/mixlab/data"
)

func TestSWAWindow128Smoke(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}

	cfg, err := LoadArchConfig(filepath.Join("experiments", "swa_test", "window128_smoke.json"))
	if err != nil {
		t.Fatalf("LoadArchConfig: %v", err)
	}
	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig: %v", err)
	}
	trainer, err := initGPUTrainer(prog, cfg, nil, nil)
	if err != nil {
		t.Fatalf("initGPUTrainer: %v", err)
	}
	defer trainer.CloseTrainer()

	trainPattern := filepath.Join("..", "data", "example", "train_*.bin")
	loader, err := data.NewLoader(trainPattern, cfg.Training.Seed, effectiveShuffleChunkTokens(cfg))
	if err != nil {
		t.Fatalf("data.NewLoader: %v", err)
	}

	batchSize := cfg.Training.BatchTokens / cfg.SeqLen
	if batchSize <= 0 {
		t.Fatalf("invalid batch size from batch_tokens=%d seq_len=%d", cfg.Training.BatchTokens, cfg.SeqLen)
	}

	losses := make([]float64, 0, cfg.Training.TotalSteps())
	lr := float32(cfg.Training.LR)
	for step := 0; step < cfg.Training.TotalSteps(); step++ {
		x, y, err := loader.NextBatch(cfg.Training.BatchTokens, cfg.SeqLen)
		if err != nil {
			t.Fatalf("loader.NextBatch step %d: %v", step, err)
		}
		loss, err := trainer.TrainStepGPU(x, y, batchSize, cfg.SeqLen, lr)
		if err != nil {
			t.Fatalf("TrainStepGPU step %d: %v", step, err)
		}
		if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
			t.Fatalf("loss is not finite at step %d: %v", step, loss)
		}
		losses = append(losses, float64(loss))
		t.Logf("step %02d loss=%.4f", step, loss)
	}

	if len(losses) != cfg.Training.TotalSteps() {
		t.Fatalf("recorded %d losses, want %d", len(losses), cfg.Training.TotalSteps())
	}
	const window = 10
	if len(losses) < window*2 {
		t.Fatalf("need at least %d losses to compare training trend, got %d", window*2, len(losses))
	}
	var headSum, tailSum float64
	for i := 0; i < window; i++ {
		headSum += losses[i]
		tailSum += losses[len(losses)-window+i]
	}
	headAvg := headSum / window
	tailAvg := tailSum / window
	if tailAvg >= headAvg {
		t.Fatalf("loss trend did not improve: head_avg=%.4f tail_avg=%.4f first=%.4f last=%.4f", headAvg, tailAvg, losses[0], losses[len(losses)-1])
	}
}
