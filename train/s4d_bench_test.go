//go:build mlx && cgo && (darwin || linux)

package train

import (
	"fmt"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/gpu"
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
				classificationPos:    []int32{int32(seqLen - 1)},
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

func BenchmarkS4DBidirectionalSpeechCommandsFFTLengths(b *testing.B) {
	if !mlxAvailable() {
		b.Skip("MLX backend not available")
	}
	for _, batchSize := range []int{1, 8} {
		for _, seqLen := range []int{16000, 16384} {
			b.Run(fmt.Sprintf("B_%d/T_%d", batchSize, seqLen), func(b *testing.B) {
				cfg := s4dSpeechCommandsBenchmarkConfig(b, batchSize, seqLen)
				prog, err := BuildEvalIRProgramFromConfig(cfg)
				if err != nil {
					b.Fatal(err)
				}
				trainer, err := initGPUTrainer(prog, cfg, nil, nil)
				if err != nil {
					b.Fatal(err)
				}
				defer trainer.CloseTrainer()
				batch := s4dSpeechCommandsBenchmarkBatch(batchSize, seqLen)
				if _, err := trainer.EvaluateObjectiveGPU(batch, batchSize, seqLen); err != nil {
					b.Fatal(err)
				}
				b.ResetTimer()
				start := time.Now()
				for i := 0; i < b.N; i++ {
					if _, err := trainer.EvaluateObjectiveGPU(batch, batchSize, seqLen); err != nil {
						b.Fatal(err)
					}
				}
				elapsed := time.Since(start)
				b.ReportMetric(float64(b.N*batchSize*seqLen)/elapsed.Seconds(), "tokens/s")
			})
		}
	}
}

func BenchmarkS4DBidirectionalSpeechCommandsTrainingStep(b *testing.B) {
	if !mlxAvailable() {
		b.Skip("MLX backend not available")
	}
	const batchSize, seqLen = 8, 16000
	cfg := s4dSpeechCommandsBenchmarkConfig(b, batchSize, seqLen)
	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		b.Fatal(err)
	}
	trainerInterface, err := initGPUTrainer(prog, cfg, nil, nil)
	if err != nil {
		b.Fatal(err)
	}
	trainer := trainerInterface.(*mlxGPUTrainer)
	defer trainer.CloseTrainer()
	batch := s4dSpeechCommandsBenchmarkBatch(batchSize, seqLen)
	if _, err := trainer.TrainObjectiveStepGPU(
		batch, batchSize, seqLen, float32(cfg.Training.LR),
	); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := trainer.TrainObjectiveStepGPU(
			batch, batchSize, seqLen, float32(cfg.Training.LR),
		); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
	b.ReportMetric(float64(b.N*batchSize*seqLen)/b.Elapsed().Seconds(), "tokens/s")
}

// BenchmarkBidirectionalRecurrentMixersLongSequenceTraining keeps the
// production-shaped Speech Commands mixer comparison reproducible. Run each
// sub-benchmark in a separate process so MLX peak memory is attributable to
// that mixer rather than the process-wide high-water mark from an earlier arm.
func BenchmarkBidirectionalRecurrentMixersLongSequenceTraining(b *testing.B) {
	if !mlxAvailable() {
		b.Skip("MLX backend not available")
	}
	const batchSize, seqLen, layers = 4, 16000, 6
	for _, tc := range []struct {
		name  string
		block string
	}{
		{
			name:  "s4d",
			block: `{"type":"s4d","state_size":64,"n_ssm":2,"bidirectional":true,"discretization":"bilinear","measure":"diag-lin","output_transform":"glu"}`,
		},
		{
			name:  "mamba3_canonical",
			block: `{"type":"mamba3-canonical","inner_dim":128,"state_size":64,"n_groups":4,"dt_rank":4,"scan_chunk_size":64,"bidirectional":true}`,
		},
		{
			name:  "gated_deltanet",
			block: `{"type":"gated_deltanet","heads":4,"d_k":16,"d_v":32,"scan_chunk_size":64,"bidirectional":true}`,
		},
	} {
		b.Run(tc.name, func(b *testing.B) {
			cfg := recurrentMixerLongSequenceBenchmarkConfig(
				b, tc.name, tc.block, batchSize, seqLen, layers)
			prog, err := BuildIRProgramFromConfig(cfg)
			if err != nil {
				b.Fatal(err)
			}
			trainerInterface, err := initGPUTrainer(prog, cfg, nil, nil)
			if err != nil {
				b.Fatal(err)
			}
			trainer := trainerInterface.(*mlxGPUTrainer)
			defer trainer.CloseTrainer()
			batch := s4dSpeechCommandsBenchmarkBatch(batchSize, seqLen)
			if _, err := trainer.TrainObjectiveStepGPU(
				batch, batchSize, seqLen, float32(cfg.Training.LR),
			); err != nil {
				b.Fatal(err)
			}
			gpu.ClearMemoryCache()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := trainer.TrainObjectiveStepGPU(
					batch, batchSize, seqLen, float32(cfg.Training.LR),
				); err != nil {
					b.Fatal(err)
				}
			}
			b.StopTimer()
			memory := gpu.MemoryStatsSnapshot()
			b.ReportMetric(float64(b.N*batchSize*seqLen)/b.Elapsed().Seconds(), "tokens/s")
			b.ReportMetric(float64(memory.PeakBytes)/(1<<20), "peak-MiB")
		})
	}
}

func recurrentMixerLongSequenceBenchmarkConfig(
	tb testing.TB,
	name, block string,
	batchSize, seqLen, layers int,
) *ArchConfig {
	tb.Helper()
	blocks := ""
	for layer := 0; layer < layers; layer++ {
		if layer > 0 {
			blocks += ","
		}
		blocks += block
	}
	const modelDim = 128
	cfg, err := ParseArchConfig([]byte(fmt.Sprintf(`{
		"name":"%s_long_sequence_training",
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
			"optimizer":"adamw","batch_tokens":%d,"steps":1,
			"lr":0.0001,"grad_clip":1,"weight_decay":0,"seed":37
		}
	}`, name, modelDim, seqLen, blocks, batchSize*seqLen)), name)
	if err != nil {
		tb.Fatal(err)
	}
	return cfg
}

func s4dSpeechCommandsBenchmarkConfig(tb testing.TB, batchSize, seqLen int) *ArchConfig {
	tb.Helper()
	const modelDim = 128
	cfg, err := ParseArchConfig([]byte(fmt.Sprintf(`{
		"name":"s4d_bidirectional_fft_%d",
		"model_dim":%d,
		"seq_len":%d,
		"positional_embedding":"none",
		"input_adapter":{"kind":"linear_frames","feature_dim":1,"bias":true,"norm":"none"},
		"norm_type":"batchnorm",
		"norm_placement":"pre",
		"blocks":[{
			"type":"s4d","state_size":64,"n_ssm":2,"bidirectional":true,
			"discretization":"bilinear","trainable_b":true,"output_transform":"glu"
		}],
		"training":{
			"objective":"classification",
			"classification":{"num_labels":35,"pooling":"mean","classifier_dropout":0},
			"optimizer":"adamw","batch_tokens":%d,"steps":1,"lr":0.01,"weight_decay":0.05
		}
	}`, seqLen, modelDim, seqLen, batchSize*seqLen)), "s4d_bidirectional_fft")
	if err != nil {
		tb.Fatal(err)
	}
	return cfg
}

func s4dSpeechCommandsBenchmarkBatch(batchSize, seqLen int) objectiveBatch {
	positions := make([]int32, batchSize)
	for i := range positions {
		positions[i] = int32(seqLen - 1)
	}
	return objectiveBatch{
		frames:               make([]float32, batchSize*seqLen),
		classificationLabels: make([]int32, batchSize),
		classificationMask:   repeatFloat32Train(batchSize*seqLen, 1),
		classificationPos:    positions,
	}
}
