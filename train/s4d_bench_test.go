//go:build mlx && cgo && (darwin || linux)

package train

import (
	"fmt"
	"testing"
	"time"
)

func BenchmarkS4DVsMamba3CanonicalLongSequenceForward(b *testing.B) {
	if !mlxAvailable() {
		b.Skip("MLX backend not available")
	}
	const (
		modelDim  = 64
		stateSize = 64
		seqLen    = 4096
	)
	cases := []struct {
		name  string
		block string
	}{
		{"s4d_fft", `{"type":"s4d","state_size":64}`},
		{"mamba3_canonical", `{"type":"mamba3-canonical","inner_dim":64,"state_size":64,"n_groups":4,"dt_rank":4,"scan_chunk_size":64}`},
	}
	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			cfg, err := ParseArchConfig([]byte(fmt.Sprintf(`{
				"name":"%s_long_forward",
				"model_dim":%d,
				"seq_len":%d,
				"positional_embedding":"none",
				"input_adapter":{"kind":"linear_frames","feature_dim":1,"bias":true,"norm":"layernorm"},
				"blocks":[%s],
				"training":{
					"objective":"classification",
					"classification":{"num_labels":2,"pooling":"last","classifier_dropout":0},
					"optimizer":"adamw",
					"batch_tokens":%d,
					"steps":1,
					"lr":0.0001,
					"weight_decay":0
				}
			}`, tc.name, modelDim, seqLen, tc.block, seqLen)), tc.name)
			if err != nil {
				b.Fatal(err)
			}
			prog, err := BuildEvalIRProgramFromConfig(cfg)
			if err != nil {
				b.Fatal(err)
			}
			trainer, err := initGPUTrainer(prog, cfg, nil, nil)
			if err != nil {
				b.Fatal(err)
			}
			defer trainer.CloseTrainer()
			batch := objectiveBatch{
				frames:               make([]float32, seqLen),
				classificationLabels: []int32{0},
				classificationMask:   repeatFloat32Train(seqLen, 1),
				classificationPos:    []int32{seqLen - 1},
			}
			if _, err := trainer.EvaluateObjectiveGPU(batch, 1, seqLen); err != nil {
				b.Fatal(err)
			}
			b.ResetTimer()
			start := time.Now()
			for i := 0; i < b.N; i++ {
				if _, err := trainer.EvaluateObjectiveGPU(batch, 1, seqLen); err != nil {
					b.Fatal(err)
				}
			}
			elapsed := time.Since(start)
			b.ReportMetric(float64(b.N*seqLen)/elapsed.Seconds(), "tokens/s")
		})
	}
}
