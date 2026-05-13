package train

import (
	"fmt"
	"math"
	"os"
	"strings"
	"time"

	"github.com/mrothroc/mixlab/data"
	"github.com/mrothroc/mixlab/logprobs"
	"github.com/mrothroc/mixlab/ranks"
)

func runEvalLogprobs(configPath, trainPattern, safetensorsLoad, lutDir, logprobsOut, ranksOut string) error {
	if configPath == "" {
		return fmt.Errorf("-config is required for eval mode; pass a JSON config file, e.g.: mixlab -mode eval -config examples/plain_3L.json -safetensors-load weights.st -train 'data/train_*.bin'")
	}
	if safetensorsLoad == "" {
		return fmt.Errorf("-safetensors-load is required for eval mode")
	}
	if trainPattern == "" {
		return fmt.Errorf("-train is required for eval mode; pass a glob pattern for data shards, e.g.: -train 'data/train_*.bin'")
	}
	if logprobsOut == "" && ranksOut == "" {
		return fmt.Errorf("at least one of -logprobs-out or -ranks-out is required for export")
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
	if err := runFullEvalLogprobs(session, valPattern, lutDir, logprobsOut, ranksOut); err != nil {
		return err
	}

	fmt.Printf("loaded config %q: model_dim=%d vocab_size=%d seq_len=%d blocks=%d\n",
		cfg.Name, cfg.ModelDim, cfg.VocabSize, cfg.SeqLen, len(cfg.Blocks))
	fmt.Printf("  [%s] loaded %d weights from %s\n", cfg.Name, session.weightCount, safetensorsLoad)
	if logprobsOut != "" {
		fmt.Printf("  [%s] wrote per-token eval NLLs to %s\n", cfg.Name, logprobsOut)
	}
	if ranksOut != "" {
		fmt.Printf("  [%s] wrote per-token target ranks to %s\n", cfg.Name, ranksOut)
	}
	return nil
}

// logprobsRanksWriters bundles the optional writers for the per-token NLL and
// rank export files so the eval loop can stay clean.
type logprobsRanksWriters struct {
	logprobs *logprobs.Writer
	ranks    *ranks.Writer
}

func runFullEvalLogprobs(session *InferenceSession, valPattern, lutDir, logprobsOut, ranksOut string) error {
	cfg := session.Config()
	seqLen := cfg.SeqLen
	batchTokens := cfg.Training.BatchTokens
	if batchTokens%seqLen != 0 {
		return fmt.Errorf("batch_tokens (%d) must be divisible by seq_len (%d)", batchTokens, seqLen)
	}
	if cfg.Training.TTTMode == "lora" && cfg.Training.TTTSteps > 0 {
		return fmt.Errorf("logprob export does not support ttt_mode=lora; per-token export only supports plain eval or score-first full TTT")
	}
	if cfg.EffectiveEvalSpec().LegalChunkSGDEnabled() {
		return fmt.Errorf("logprob export does not support eval.ttt_mode=legal_chunk_sgd")
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

	writers, closeWriters, err := openExportWriters(cfg.VocabSize, totalTokens, logprobsOut, ranksOut)
	if err != nil {
		return err
	}
	defer closeWriters()

	// When ranks are requested we read logits once per batch and derive both
	// NLL and rank on the CPU; this avoids a second GPU pass for the same data.
	deriveFromLogits := ranksOut != ""

	totalLossNats := 0.0
	totalBytes := 0.0
	writtenTokens := 0
	rawStart := 0
	processedSeqs := 0
	totalSeqs := totalTokens / seqLen
	evalStart := time.Now()
	lastProgress := evalStart

	// Per-batch scratch reused across iterations.
	nllsBuf := make([]float32, batchTokens)
	ranksBuf := make([]uint16, batchTokens)

	for rawStart < totalTokens {
		chunkTokens := min(batchTokens, totalTokens-rawStart)
		if chunkTokens <= 0 || chunkTokens%seqLen != 0 {
			break
		}

		local := valTokens[rawStart : rawStart+chunkTokens+1]
		yIDs := make([]uint16, chunkTokens)
		for i := range chunkTokens {
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

		if deriveFromLogits {
			logits, err := session.evalLogitsBatch(local)
			if err != nil {
				return fmt.Errorf("eval failed at token offset %d: %w", rawStart, err)
			}
			vocab := cfg.VocabSize
			if len(logits) != chunkTokens*vocab {
				return fmt.Errorf("logits length mismatch at offset %d: got=%d want=%d", rawStart, len(logits), chunkTokens*vocab)
			}
			for i := range chunkTokens {
				row := logits[i*vocab : (i+1)*vocab]
				nll, err := targetNLLFromLogits(row, vocab, yIDs[i])
				if err != nil {
					return fmt.Errorf("nll derivation failed at token offset %d: %w", rawStart+i, err)
				}
				if !logprobs.IsFinite(nll) {
					return fmt.Errorf("non-finite NLL at token offset %d", rawStart+i)
				}
				rk, err := targetRankFromLogits(row, vocab, yIDs[i])
				if err != nil {
					return fmt.Errorf("rank derivation failed at token offset %d: %w", rawStart+i, err)
				}
				nllsBuf[i] = nll
				ranksBuf[i] = rk
			}
		} else {
			batchNLLs, err := session.EvalTokens(local)
			if err != nil {
				return fmt.Errorf("eval failed at token offset %d: %w", rawStart, err)
			}
			if len(batchNLLs) != chunkTokens {
				return fmt.Errorf("per-token eval length mismatch at offset %d: got=%d want=%d", rawStart, len(batchNLLs), chunkTokens)
			}
			for i, nll := range batchNLLs {
				if !logprobs.IsFinite(nll) {
					return fmt.Errorf("non-finite NLL at token offset %d", rawStart+i)
				}
				nllsBuf[i] = nll
			}
		}

		batchLoss := 0.0
		for i := range chunkTokens {
			batchLoss += float64(nllsBuf[i])
		}
		totalLossNats += batchLoss

		if writers.logprobs != nil {
			if err := writers.logprobs.AppendBatch(yIDs, nllsBuf[:chunkTokens]); err != nil {
				return fmt.Errorf("write logprobs at token offset %d: %w", rawStart, err)
			}
		}
		if writers.ranks != nil {
			if err := writers.ranks.AppendBatch(yIDs, ranksBuf[:chunkTokens]); err != nil {
				return fmt.Errorf("write ranks at token offset %d: %w", rawStart, err)
			}
		}
		writtenTokens += chunkTokens

		if cfg.Training.TTTSteps > 0 {
			xTok := make([]int, chunkTokens)
			yTok := make([]int, chunkTokens)
			for i := range chunkTokens {
				xTok[i] = int(local[i])
				yTok[i] = int(local[i+1])
			}
			batchSize := chunkTokens / seqLen
			for step := range cfg.Training.TTTSteps {
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

	if writers.logprobs != nil {
		if err := writers.logprobs.Close(); err != nil {
			return fmt.Errorf("finalize logprob output: %w", err)
		}
	}
	if writers.ranks != nil {
		if err := writers.ranks.Close(); err != nil {
			return fmt.Errorf("finalize ranks output: %w", err)
		}
	}
	if writtenTokens != totalTokens {
		return fmt.Errorf("written token count mismatch: wrote=%d total=%d", writtenTokens, totalTokens)
	}
	if totalBytes <= 0 {
		return fmt.Errorf("invalid BPB total bytes: %f", totalBytes)
	}

	avgNLL := totalLossNats / float64(writtenTokens)
	bpb := (totalLossNats / math.Log(2.0)) / totalBytes
	outputLabel := logprobsOut
	if outputLabel == "" {
		outputLabel = ranksOut
	}
	fmt.Printf("  [%s] full_val+logprobs nll=%.6f bpb=%.6f tokens=%d bytes=%.0f output=%s\n", cfg.Name, avgNLL, bpb, writtenTokens, totalBytes, outputLabel)
	return nil
}

// openExportWriters creates the optional per-token NLL and rank writers. The
// returned closer must always be called (it best-effort-closes the underlying
// files on early return paths).
func openExportWriters(vocabSize, totalTokens int, logprobsOut, ranksOut string) (logprobsRanksWriters, func(), error) {
	var w logprobsRanksWriters
	closers := make([]func() error, 0, 2)
	cleanup := func() {
		for i := len(closers) - 1; i >= 0; i-- {
			_ = closers[i]()
		}
	}

	if logprobsOut != "" {
		f, err := os.Create(logprobsOut)
		if err != nil {
			cleanup()
			return logprobsRanksWriters{}, func() {}, fmt.Errorf("create logprob output %q: %w", logprobsOut, err)
		}
		closers = append(closers, f.Close)
		lw, err := logprobs.NewWriter(f, uint32(vocabSize), uint32(totalTokens))
		if err != nil {
			cleanup()
			return logprobsRanksWriters{}, func() {}, fmt.Errorf("initialize logprob writer: %w", err)
		}
		w.logprobs = lw
	}
	if ranksOut != "" {
		f, err := os.Create(ranksOut)
		if err != nil {
			cleanup()
			return logprobsRanksWriters{}, func() {}, fmt.Errorf("create ranks output %q: %w", ranksOut, err)
		}
		closers = append(closers, f.Close)
		rw, err := ranks.NewWriter(f, uint32(vocabSize), uint32(totalTokens))
		if err != nil {
			cleanup()
			return logprobsRanksWriters{}, func() {}, fmt.Errorf("initialize ranks writer: %w", err)
		}
		w.ranks = rw
	}
	return w, cleanup, nil
}
