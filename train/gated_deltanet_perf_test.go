//go:build mlx && cgo && (darwin || linux)

package train

import (
	"fmt"
	"testing"
)

func BenchmarkGatedDeltaNetForwardScan(b *testing.B) {
	if !mlxAvailable() {
		b.Skip("MLX backend not available")
	}

	const (
		modelDim    = 320
		seqLen      = 1024
		batchTokens = 4096
		heads       = 4
		dk          = 64
		dv          = 128
		vocabSize   = 8192
	)
	batchSize := batchTokens / seqLen

	for _, tc := range []struct {
		name      string
		chunkJSON string
	}{
		{name: "naive", chunkJSON: `"scan_chunk_size": 0,`},
		{name: "chunked", chunkJSON: `"scan_chunk_size": 64,`},
	} {
		b.Run(tc.name, func(b *testing.B) {
			cfgJSON := fmt.Sprintf(`{
				"name": "gdn_perf_%s",
				"model_dim": %d,
				"vocab_size": %d,
				"seq_len": %d,
				"blocks": [
					{"type": "gated_deltanet", "heads": %d, "d_k": %d, "d_v": %d, %s "kv_share": true}
				],
				"training": {"steps": 1, "lr": 1e-4, "seed": 7, "batch_tokens": %d}
			}`, tc.name, modelDim, vocabSize, seqLen, heads, dk, dv, tc.chunkJSON, batchTokens)

			cfg, err := ParseArchConfig([]byte(cfgJSON), "gdn_perf")
			if err != nil {
				b.Fatalf("ParseArchConfig: %v", err)
			}
			prog, err := BuildIRProgramFromConfig(cfg)
			if err != nil {
				b.Fatalf("BuildIRProgramFromConfig: %v", err)
			}
			trainerIface, err := initGPUTrainer(prog, cfg, nil, nil)
			if err != nil {
				b.Fatalf("initGPUTrainer: %v", err)
			}
			trainer, ok := trainerIface.(*mlxGPUTrainer)
			if !ok {
				b.Fatalf("trainer type=%T, want *mlxGPUTrainer", trainerIface)
			}
			defer trainer.CloseTrainer()

			tokens := make([]int, batchTokens)
			targets := make([]int, batchTokens)
			for i := range tokens {
				tokens[i] = i % vocabSize
				targets[i] = (i + 1) % vocabSize
			}

			if _, err := trainer.EvaluateGPU(tokens, targets, batchSize, seqLen); err != nil {
				b.Fatalf("warmup EvaluateGPU: %v", err)
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := trainer.EvaluateGPU(tokens, targets, batchSize, seqLen); err != nil {
					b.Fatalf("EvaluateGPU: %v", err)
				}
			}
			b.StopTimer()
			b.ReportMetric(float64(batchTokens*b.N)/b.Elapsed().Seconds(), "tok/s")
		})
	}
}
