package train

import (
	"fmt"
	"math"
	"time"

	"github.com/mrothroc/mixlab/data"
	"github.com/mrothroc/mixlab/gpu"
)

func runFullEvalLegalChunkSGD(cfg *ArchConfig, valPattern string, sourceTrainer GPUTrainer, lutDir string) error {
	initialWeights, err := readTrainerWeights(sourceTrainer)
	if err != nil {
		return fmt.Errorf("read weights for legal eval TTT: %w", err)
	}
	return runFullEvalLegalChunkSGDFromWeights(cfg, valPattern, initialWeights, lutDir)
}

func runFullEvalLegalChunkSGDFromWeights(cfg *ArchConfig, valPattern string, initialWeights [][]float32, lutDir string) error {
	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		return fmt.Errorf("build IR program for legal eval TTT: %w", err)
	}
	trainer, err := initLegalChunkSGDTrainer(prog, cfg, initialWeights)
	if err != nil {
		return err
	}
	defer trainer.CloseTrainer()
	return runFullEvalLegalChunkSGDWithTrainer(cfg, valPattern, trainer, lutDir)
}

func initLegalChunkSGDTrainer(prog *Program, cfg *ArchConfig, initialWeights [][]float32) (GPUTrainer, error) {
	evalSpec := cfg.EffectiveEvalSpec()
	return initGPUTrainer(prog, cfg, cloneWeights(initialWeights), legalChunkSGDOptimizerOverride(evalSpec))
}

func legalChunkSGDOptimizerOverride(evalSpec EvalSpec) func(gpu.TrainerOptimizerSpec, []WeightShape) (gpu.TrainerOptimizerSpec, error) {
	return func(_ gpu.TrainerOptimizerSpec, shapes []WeightShape) (gpu.TrainerOptimizerSpec, error) {
		wmeta := make([]gpu.OptimizerWeightMetadata, len(shapes))
		for i, shape := range shapes {
			wmeta[i] = gpu.OptimizerWeightMetadata{
				Name:        shape.Name,
				Shape:       shape.Shape,
				IsNormScale: shape.IsNormScale,
			}
		}
		settings := gpu.OptimizerSettings{
			Name:        "sgd",
			LR:          float32(evalSpec.TTTLR),
			Beta1:       float32(evalSpec.EffectiveTTTMomentum()),
			WeightDecay: 0,
		}
		return gpu.BuildTrainerOptimizerSpec(gpu.TrainerOptimizerConfig{
			Weights:       wmeta,
			Embed:         settings,
			Head:          settings,
			Scalar:        settings,
			Matrix:        settings,
			MaxGradNorm:   0,
			DefaultBaseLR: float32(evalSpec.TTTLR),
		})
	}
}

func runFullEvalLegalChunkSGDWithTrainer(cfg *ArchConfig, valPattern string, trainer GPUTrainer, lutDir string) error {
	evalSpec := cfg.EffectiveEvalSpec()
	if !evalSpec.LegalChunkSGDEnabled() {
		return fmt.Errorf("eval.ttt_mode must be legal_chunk_sgd")
	}

	name := cfg.Name
	seqLen := cfg.SeqLen
	batchTokens := cfg.Training.BatchTokens
	if batchTokens%seqLen != 0 {
		return fmt.Errorf("batch_tokens (%d) must be divisible by seq_len (%d)", batchTokens, seqLen)
	}
	batchSize := batchTokens / seqLen

	luts, err := loadBPBLUTs(lutDir, cfg.VocabSize)
	if err != nil {
		return fmt.Errorf("load BPB LUTs from %s: %w\n  generate LUTs with: mixlab -mode prepare, or set -lut-dir to the directory containing bytes_per_token.bin, has_leading_space.bin, is_boundary_token.bin", lutDir, err)
	}

	valTokens, err := data.LoadValidationTokens(valPattern, seqLen)
	if err != nil {
		return err
	}
	if len(valTokens) < 2 {
		return fmt.Errorf("validation tokens too short: %d", len(valTokens))
	}

	totalEvalTokens := ((len(valTokens) - 1) / batchTokens) * batchTokens
	if totalEvalTokens <= 0 {
		return fmt.Errorf("validation set has no complete eval batches")
	}
	totalChunks := (totalEvalTokens + evalSpec.ChunkTokens - 1) / evalSpec.ChunkTokens
	totalSeqs := totalEvalTokens / seqLen

	totalLossNats := 0.0
	totalBytes := 0.0
	totalTokens := 0
	processedSeqs := 0
	evalStart := time.Now()
	lastProgress := evalStart

	for chunkIndex, chunkStart := 0, 0; chunkStart < totalEvalTokens; chunkIndex, chunkStart = chunkIndex+1, chunkStart+evalSpec.ChunkTokens {
		chunkEnd := chunkStart + evalSpec.ChunkTokens
		if chunkEnd > totalEvalTokens {
			chunkEnd = totalEvalTokens
		}
		chunkLR := legalChunkSGDLR(float32(evalSpec.TTTLR), evalSpec.TTTLRSchedule, chunkIndex, totalChunks)

		for rawStart := chunkStart; rawStart < chunkEnd; rawStart += batchTokens {
			xTok, yTok := evalTokenBatch(valTokens, rawStart, batchTokens)
			lossV, err := trainer.EvaluateGPU(xTok, yTok, batchSize, seqLen)
			if err != nil {
				return fmt.Errorf("legal eval score failed at token offset %d: %w", rawStart, err)
			}
			totalLossNats += float64(lossV) * float64(batchTokens)
			batchBytes, err := tokenByteCount(luts, xTok, yTok, rawStart)
			if err != nil {
				return err
			}
			totalBytes += batchBytes
			totalTokens += batchTokens
			processedSeqs += batchTokens / seqLen
		}

		for epoch := 0; epoch < evalSpec.TTTEpochs; epoch++ {
			for rawStart := chunkStart; rawStart < chunkEnd; rawStart += batchTokens {
				xTok, yTok := evalTokenBatch(valTokens, rawStart, batchTokens)
				if _, err := trainer.TrainStepGPU(xTok, yTok, batchSize, seqLen, chunkLR); err != nil {
					return fmt.Errorf("legal eval SGD epoch %d failed at token offset %d: %w", epoch+1, rawStart, err)
				}
			}
		}

		now := time.Now()
		if now.Sub(lastProgress) >= 30*time.Second || chunkEnd >= totalEvalTokens {
			pct := 100.0 * float64(processedSeqs) / float64(totalSeqs)
			elapsed := now.Sub(evalStart)
			etaStr := "0s"
			if processedSeqs < totalSeqs && processedSeqs > 0 {
				remainingSeqs := totalSeqs - processedSeqs
				eta := time.Duration((elapsed.Seconds() / float64(processedSeqs)) * float64(remainingSeqs) * float64(time.Second))
				etaStr = eta.Round(time.Second).String()
			}
			fmt.Printf("  [%s] legal_ttt progress chunk=%d/%d seq=%d/%d (%.1f%%) lr=%g elapsed=%s eta=%s\n",
				name, chunkIndex+1, totalChunks, processedSeqs, totalSeqs, pct, chunkLR, elapsed.Round(time.Second), etaStr)
			lastProgress = now
		}
	}

	if totalTokens == 0 || totalBytes <= 0 {
		return fmt.Errorf("invalid BPB totals: tokens=%d bytes=%f", totalTokens, totalBytes)
	}
	bpb := (totalLossNats / math.Log(2.0)) / totalBytes
	avgNLL := totalLossNats / float64(totalTokens)
	fmt.Printf("  [%s] full_val nll=%.6f bpb=%.6f tokens=%d bytes=%.0f eval_ttt_mode=legal_chunk_sgd chunk_tokens=%d ttt_epochs=%d ttt_lr=%g ttt_momentum=%g ttt_lr_schedule=%s\n",
		name, avgNLL, bpb, totalTokens, totalBytes, evalSpec.ChunkTokens, evalSpec.TTTEpochs, evalSpec.TTTLR, evalSpec.EffectiveTTTMomentum(), evalSpec.TTTLRSchedule)
	return nil
}

func evalTokenBatch(tokens []uint16, start, n int) ([]int, []int) {
	xTok := make([]int, n)
	yTok := make([]int, n)
	local := tokens[start : start+n+1]
	for i := 0; i < n; i++ {
		xTok[i] = int(local[i])
		yTok[i] = int(local[i+1])
	}
	return xTok, yTok
}

func legalChunkSGDLR(base float32, schedule string, chunkIndex, totalChunks int) float32 {
	if schedule == "constant" || totalChunks <= 0 {
		return base
	}
	return base * float32(0.5*(1+math.Cos(math.Pi*float64(chunkIndex)/float64(totalChunks))))
}
