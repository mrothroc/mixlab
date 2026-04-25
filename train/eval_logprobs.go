package train

import (
	"fmt"
	"math"
	"os"
	"strings"
	"time"

	"github.com/mrothroc/mixlab/data"
	"github.com/mrothroc/mixlab/logprobs"
)

func runEvalLogprobs(configPath, trainPattern, safetensorsLoad, lutDir, outputPath string) error {
	if configPath == "" {
		return fmt.Errorf("-config is required for eval mode; pass a JSON config file, e.g.: mixlab -mode eval -config examples/plain_3L.json -safetensors-load weights.st -train 'data/train_*.bin'")
	}
	if safetensorsLoad == "" {
		return fmt.Errorf("-safetensors-load is required for eval mode")
	}
	if trainPattern == "" {
		return fmt.Errorf("-train is required for eval mode; pass a glob pattern for data shards, e.g.: -train 'data/train_*.bin'")
	}
	if outputPath == "" {
		return fmt.Errorf("-logprobs-out is required for logprob export")
	}

	session, err := NewInferenceSession(configPath, safetensorsLoad)
	if err != nil {
		return err
	}
	defer func() { _ = session.Close() }()

	cfg := session.Config()
	if cfg.VocabSize > math.MaxUint16 {
		return fmt.Errorf("vocab_size=%d exceeds uint16 record format limit", cfg.VocabSize)
	}

	valPattern := strings.Replace(trainPattern, "train", "val", 1)
	if err := runFullEvalLogprobs(session, valPattern, lutDir, outputPath); err != nil {
		return err
	}

	fmt.Printf("loaded config %q: model_dim=%d vocab_size=%d seq_len=%d blocks=%d\n",
		cfg.Name, cfg.ModelDim, cfg.VocabSize, cfg.SeqLen, len(cfg.Blocks))
	fmt.Printf("  [%s] loaded %d weights from %s\n", cfg.Name, session.weightCount, safetensorsLoad)
	fmt.Printf("  [%s] wrote per-token eval NLLs to %s\n", cfg.Name, outputPath)
	return nil
}

func runFullEvalLogprobs(session *InferenceSession, valPattern, lutDir, outputPath string) error {
	cfg := session.Config()
	seqLen := cfg.SeqLen
	batchTokens := cfg.Training.BatchTokens
	if batchTokens%seqLen != 0 {
		return fmt.Errorf("batch_tokens (%d) must be divisible by seq_len (%d)", batchTokens, seqLen)
	}
	if cfg.Training.TTTMode == "lora" && cfg.Training.TTTSteps > 0 {
		return fmt.Errorf("logprob export does not support ttt_mode=lora; per-token export only supports plain eval or score-first full TTT")
	}
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
	// Truncate to full batches to avoid partial-batch extended-target errors.
	totalTokens := ((len(valTokens) - 1) / batchTokens) * batchTokens
	if totalTokens <= 0 {
		return fmt.Errorf("validation set has no complete sequences")
	}

	f, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("create logprob output %q: %w", outputPath, err)
	}
	defer func() {
		_ = f.Close()
	}()

	writer, err := logprobs.NewWriter(f, uint32(cfg.VocabSize), uint32(totalTokens))
	if err != nil {
		return fmt.Errorf("initialize logprob writer: %w", err)
	}

	totalLossNats := 0.0
	totalBytes := 0.0
	writtenTokens := 0
	rawStart := 0
	processedSeqs := 0
	totalSeqs := totalTokens / seqLen
	evalStart := time.Now()
	lastProgress := evalStart

	for rawStart < totalTokens {
		chunkTokens := batchTokens
		if totalTokens-rawStart < chunkTokens {
			chunkTokens = totalTokens - rawStart
		}
		if chunkTokens <= 0 || chunkTokens%seqLen != 0 {
			break
		}

		local := valTokens[rawStart : rawStart+chunkTokens+1]
		yIDs := make([]uint16, chunkTokens)
		for i := 0; i < chunkTokens; i++ {
			tgtID := int(local[i+1])
			if tgtID < 0 || tgtID > math.MaxUint16 {
				return fmt.Errorf("target token id out of uint16 range at offset %d: %d", rawStart+i, tgtID)
			}
			yIDs[i] = uint16(tgtID)
			prevID := int(local[i])
			if prevID < 0 || prevID >= len(luts.isBoundary) || tgtID < 0 || tgtID >= len(luts.baseBytes) || tgtID >= len(luts.hasLeading) {
				return fmt.Errorf("token id out of LUT bounds at offset %d: prev=%d tgt=%d", rawStart+i, prevID, tgtID)
			}
			tokenBytes := float64(luts.baseBytes[tgtID])
			if luts.hasLeading[tgtID] && !luts.isBoundary[prevID] {
				tokenBytes += 1
			}
			totalBytes += tokenBytes
		}
		nlls, err := session.EvalTokens(local)
		if err != nil {
			return fmt.Errorf("eval failed at token offset %d: %w", rawStart, err)
		}
		if len(nlls) != chunkTokens {
			return fmt.Errorf("per-token eval length mismatch at offset %d: got=%d want=%d", rawStart, len(nlls), chunkTokens)
		}
		batchLoss := 0.0
		for _, nll := range nlls {
			if !logprobs.IsFinite(nll) {
				return fmt.Errorf("non-finite NLL at token offset %d", rawStart)
			}
			batchLoss += float64(nll)
		}
		totalLossNats += batchLoss
		if err := writer.AppendBatch(yIDs, nlls); err != nil {
			return fmt.Errorf("write logprobs at token offset %d: %w", rawStart, err)
		}
		writtenTokens += chunkTokens

		if cfg.Training.TTTSteps > 0 {
			xTok := make([]int, chunkTokens)
			yTok := make([]int, chunkTokens)
			for i := 0; i < chunkTokens; i++ {
				xTok[i] = int(local[i])
				yTok[i] = int(local[i+1])
			}
			batchSize := chunkTokens / seqLen
			for step := 0; step < cfg.Training.TTTSteps; step++ {
				if _, err := session.trainer.TrainStepGPU(xTok, yTok, batchSize, seqLen, float32(cfg.Training.TTTLR)); err != nil {
					return fmt.Errorf("ttt step %d failed at token offset %d: %w", step+1, rawStart, err)
				}
			}
		}

		rawStart += chunkTokens
		processedSeqs += chunkTokens / seqLen

		now := time.Now()
		if now.Sub(lastProgress) >= 30*time.Second || processedSeqs >= totalSeqs {
			pct := 100.0 * float64(processedSeqs) / float64(totalSeqs)
			elapsed := now.Sub(evalStart)
			etaStr := "0s"
			if processedSeqs < totalSeqs && processedSeqs > 0 {
				remainingSeqs := totalSeqs - processedSeqs
				eta := time.Duration((elapsed.Seconds() / float64(processedSeqs)) * float64(remainingSeqs) * float64(time.Second))
				etaStr = eta.Round(time.Second).String()
			}
			fmt.Printf("  [%s] full_val+logprobs progress seq=%d/%d (%.1f%%) elapsed=%s eta=%s\n",
				cfg.Name, processedSeqs, totalSeqs, pct, elapsed.Round(time.Second), etaStr)
			lastProgress = now
		}
	}

	if err := writer.Close(); err != nil {
		return fmt.Errorf("finalize logprob output: %w", err)
	}
	if writtenTokens != totalTokens {
		return fmt.Errorf("written token count mismatch: wrote=%d total=%d", writtenTokens, totalTokens)
	}
	if totalBytes <= 0 {
		return fmt.Errorf("invalid BPB total bytes: %f", totalBytes)
	}

	avgNLL := totalLossNats / float64(writtenTokens)
	bpb := (totalLossNats / math.Log(2.0)) / totalBytes
	fmt.Printf("  [%s] full_val+logprobs nll=%.6f bpb=%.6f tokens=%d bytes=%.0f output=%s\n", cfg.Name, avgNLL, bpb, writtenTokens, totalBytes, outputPath)
	return nil
}
