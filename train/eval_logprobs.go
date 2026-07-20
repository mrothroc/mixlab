package train

import (
	"fmt"
	"math"
	"os"
	"strings"
	"time"

	"github.com/mrothroc/mixlab/data"
	"github.com/mrothroc/mixlab/logits"
	"github.com/mrothroc/mixlab/logprobs"
	"github.com/mrothroc/mixlab/ranks"
	"github.com/mrothroc/mixlab/uncertainty"
)

// EvalExportOptions describes the per-token output files mixlab eval mode
// should produce in a single GPU pass over the validation set. At least one
// output path must be non-empty.
type EvalExportOptions struct {
	LogprobsOut    string
	RanksOut       string
	UncertaintyOut string
	LogitsOut      string
	// LogitsDType selects the on-disk dtype for -logits-out. Defaults to
	// DTypeFloat16 when LogitsOut is set.
	LogitsDType logits.DType
	// LogitsForm selects whether -logits-out stores raw logits or
	// log_softmax(logits). Defaults to FormRaw when LogitsOut is set.
	LogitsForm logits.Form
}

// validatePaths returns an error if any two non-empty export paths point at
// the same file. Without this check, two writers would share an inode and
// silently corrupt each other's output.
func (e EvalExportOptions) validatePaths() error {
	paths := []struct {
		flag string
		path string
	}{
		{"-logprobs-out", e.LogprobsOut},
		{"-ranks-out", e.RanksOut},
		{"-uncertainty-out", e.UncertaintyOut},
		{"-logits-out", e.LogitsOut},
	}
	seen := make(map[string]string, len(paths))
	for _, p := range paths {
		if p.path == "" {
			continue
		}
		if prev, ok := seen[p.path]; ok {
			return fmt.Errorf("export paths must be distinct: %s and %s both point at %q", prev, p.flag, p.path)
		}
		seen[p.path] = p.flag
	}
	return nil
}

func runEvalExports(configPath, trainPattern, safetensorsLoad, lutDir string, exports EvalExportOptions) error {
	if configPath == "" {
		return fmt.Errorf("-config is required for eval mode; pass a JSON config file, e.g.: mixlab -mode eval -config examples/plain_3L.json -safetensors-load weights.st -train 'data/train_*.bin'")
	}
	if safetensorsLoad == "" {
		return fmt.Errorf("-safetensors-load is required for eval mode")
	}
	if trainPattern == "" {
		return fmt.Errorf("-train is required for eval mode; pass a glob pattern for data shards, e.g.: -train 'data/train_*.bin'")
	}
	if exports.LogprobsOut == "" && exports.RanksOut == "" && exports.UncertaintyOut == "" && exports.LogitsOut == "" {
		return fmt.Errorf("at least one of -logprobs-out, -ranks-out, -uncertainty-out, or -logits-out is required for export")
	}
	if err := exports.validatePaths(); err != nil {
		return err
	}

	session, err := newInferenceSession(configPath, safetensorsLoad, trainPattern)
	if err != nil {
		return err
	}
	defer func() { _ = session.Close() }()

	cfg := session.Config()
	if cfg.VocabSize > math.MaxUint16 {
		return fmt.Errorf("vocab_size=%d exceeds uint16 record format limit", cfg.VocabSize)
	}

	valPattern := strings.Replace(trainPattern, "train", "val", 1)
	if err := runFullEvalLogprobs(session, valPattern, lutDir, exports); err != nil {
		return err
	}

	fmt.Printf("loaded config %q: model_dim=%d vocab_size=%d seq_len=%d blocks=%d\n",
		cfg.Name, cfg.ModelDim, cfg.VocabSize, cfg.SeqLen, len(cfg.Blocks))
	fmt.Printf("  [%s] loaded %d weights from %s\n", cfg.Name, session.weightCount, safetensorsLoad)
	if exports.LogprobsOut != "" {
		fmt.Printf("  [%s] wrote per-token eval NLLs to %s\n", cfg.Name, exports.LogprobsOut)
	}
	if exports.RanksOut != "" {
		fmt.Printf("  [%s] wrote per-token target ranks to %s\n", cfg.Name, exports.RanksOut)
	}
	if exports.UncertaintyOut != "" {
		fmt.Printf("  [%s] wrote per-token uncertainty metrics to %s\n", cfg.Name, exports.UncertaintyOut)
	}
	if exports.LogitsOut != "" {
		fmt.Printf("  [%s] wrote per-token logits (dtype=%s form=%s) to %s\n",
			cfg.Name, exports.LogitsDType, exports.LogitsForm, exports.LogitsOut)
	}
	return nil
}

// runEvalLogprobs is the legacy three-output entry point kept for backwards
// compatibility; it forwards to runEvalExports.
func runEvalLogprobs(configPath, trainPattern, safetensorsLoad, lutDir, logprobsOut, ranksOut, uncertaintyOut string) error {
	return runEvalExports(configPath, trainPattern, safetensorsLoad, lutDir, EvalExportOptions{
		LogprobsOut:    logprobsOut,
		RanksOut:       ranksOut,
		UncertaintyOut: uncertaintyOut,
	})
}

// evalExportWriters bundles the optional writers for per-token export files so
// the eval loop can stay clean.
type evalExportWriters struct {
	logprobs    *logprobs.Writer
	ranks       *ranks.Writer
	uncertainty *uncertainty.Writer
	logits      *logits.Writer
}

func runFullEvalLogprobs(session *InferenceSession, valPattern, lutDir string, exports EvalExportOptions) error {
	cfg := session.Config()
	seqLen := cfg.SeqLen
	batchTokens := cfg.Training.BatchTokens
	if cfg.Training.ExampleFramingEnabled() {
		return fmt.Errorf("training.example_framing is not supported by continuous-stream per-token eval/export in v1")
	}
	if cfg.Training.RecordFramingEnabled() {
		return fmt.Errorf("one-record-per-row datasets are not supported by continuous-stream per-token eval/export; use framed validation loss")
	}
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

	writers, closeWriters, err := openExportWriters(cfg.VocabSize, totalTokens, exports)
	if err != nil {
		return err
	}
	defer closeWriters()

	// When ranks, uncertainty, or full logits are requested we read logits once
	// per batch and derive all requested outputs on the CPU; this avoids a
	// second GPU pass for the same data.
	deriveFromLogits := exports.RanksOut != "" || exports.UncertaintyOut != "" || exports.LogitsOut != ""

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
	top1Buf := make([]float32, batchTokens)
	entropyBuf := make([]float32, batchTokens)
	marginBuf := make([]float32, batchTokens)
	// logitsRowScratch holds a single converted row when -logits-out form is
	// "logprobs"; for "raw" form we write the model logits row in place.
	var logitsRowScratch []float32
	if writers.logits != nil && exports.LogitsForm == logits.FormLogprobs {
		logitsRowScratch = make([]float32, cfg.VocabSize)
	}

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
			logitsBatch, err := session.evalLogitsBatch(local)
			if err != nil {
				return fmt.Errorf("eval failed at token offset %d: %w", rawStart, err)
			}
			vocab := cfg.VocabSize
			if len(logitsBatch) != chunkTokens*vocab {
				return fmt.Errorf("logits length mismatch at offset %d: got=%d want=%d", rawStart, len(logitsBatch), chunkTokens*vocab)
			}
			for i := range chunkTokens {
				row := logitsBatch[i*vocab : (i+1)*vocab]
				nll, rk, top1, entropy, margin, err := evalMetricsFromLogits(row, vocab, yIDs[i], writers.ranks != nil, writers.uncertainty != nil)
				if err != nil {
					return fmt.Errorf("metric derivation failed at token offset %d: %w", rawStart+i, err)
				}
				if !logprobs.IsFinite(nll) {
					return fmt.Errorf("non-finite NLL at token offset %d", rawStart+i)
				}
				nllsBuf[i] = nll
				ranksBuf[i] = rk
				if writers.uncertainty != nil {
					top1Buf[i] = top1
					entropyBuf[i] = entropy
					marginBuf[i] = margin
				}
				if writers.logits != nil {
					writeRow := row
					if exports.LogitsForm == logits.FormLogprobs {
						logSoftmaxRow(row, vocab, logitsRowScratch)
						writeRow = logitsRowScratch
					}
					if err := writers.logits.Append(yIDs[i], writeRow); err != nil {
						return fmt.Errorf("write logits at token offset %d: %w", rawStart+i, err)
					}
				}
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
		if writers.uncertainty != nil {
			if err := writers.uncertainty.AppendBatch(yIDs, top1Buf[:chunkTokens], entropyBuf[:chunkTokens], marginBuf[:chunkTokens]); err != nil {
				return fmt.Errorf("write uncertainty at token offset %d: %w", rawStart, err)
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
			fmt.Printf("  [%s] full_val+exports progress seq=%d/%d (%.1f%%) elapsed=%s eta=%s\n",
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
	if writers.uncertainty != nil {
		if err := writers.uncertainty.Close(); err != nil {
			return fmt.Errorf("finalize uncertainty output: %w", err)
		}
	}
	if writers.logits != nil {
		if err := writers.logits.Close(); err != nil {
			return fmt.Errorf("finalize logits output: %w", err)
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
	outputLabel := exports.LogprobsOut
	if outputLabel == "" {
		outputLabel = exports.RanksOut
	}
	if outputLabel == "" {
		outputLabel = exports.UncertaintyOut
	}
	if outputLabel == "" {
		outputLabel = exports.LogitsOut
	}
	fmt.Printf("  [%s] full_val+exports nll=%.6f bpb=%.6f tokens=%d bytes=%.0f output=%s\n", cfg.Name, avgNLL, bpb, writtenTokens, totalBytes, outputLabel)
	return nil
}

// openExportWriters creates the optional per-token export writers. The
// returned closer must always be called (it best-effort-closes the underlying
// files on early return paths).
func openExportWriters(vocabSize, totalTokens int, exports EvalExportOptions) (evalExportWriters, func(), error) {
	var w evalExportWriters
	closers := make([]func() error, 0, 4)
	cleanup := func() {
		for i := len(closers) - 1; i >= 0; i-- {
			_ = closers[i]()
		}
	}

	if exports.LogprobsOut != "" {
		f, err := os.Create(exports.LogprobsOut)
		if err != nil {
			cleanup()
			return evalExportWriters{}, func() {}, fmt.Errorf("create logprob output %q: %w", exports.LogprobsOut, err)
		}
		closers = append(closers, f.Close)
		lw, err := logprobs.NewWriter(f, uint32(vocabSize), uint32(totalTokens))
		if err != nil {
			cleanup()
			return evalExportWriters{}, func() {}, fmt.Errorf("initialize logprob writer: %w", err)
		}
		w.logprobs = lw
	}
	if exports.RanksOut != "" {
		f, err := os.Create(exports.RanksOut)
		if err != nil {
			cleanup()
			return evalExportWriters{}, func() {}, fmt.Errorf("create ranks output %q: %w", exports.RanksOut, err)
		}
		closers = append(closers, f.Close)
		rw, err := ranks.NewWriter(f, uint32(vocabSize), uint32(totalTokens))
		if err != nil {
			cleanup()
			return evalExportWriters{}, func() {}, fmt.Errorf("initialize ranks writer: %w", err)
		}
		w.ranks = rw
	}
	if exports.UncertaintyOut != "" {
		f, err := os.Create(exports.UncertaintyOut)
		if err != nil {
			cleanup()
			return evalExportWriters{}, func() {}, fmt.Errorf("create uncertainty output %q: %w", exports.UncertaintyOut, err)
		}
		closers = append(closers, f.Close)
		uw, err := uncertainty.NewWriter(f, uint32(vocabSize), uint32(totalTokens))
		if err != nil {
			cleanup()
			return evalExportWriters{}, func() {}, fmt.Errorf("initialize uncertainty writer: %w", err)
		}
		w.uncertainty = uw
	}
	if exports.LogitsOut != "" {
		f, err := os.Create(exports.LogitsOut)
		if err != nil {
			cleanup()
			return evalExportWriters{}, func() {}, fmt.Errorf("create logits output %q: %w", exports.LogitsOut, err)
		}
		closers = append(closers, f.Close)
		lw, err := logits.NewWriter(f, uint32(vocabSize), uint32(totalTokens), exports.LogitsDType, exports.LogitsForm)
		if err != nil {
			cleanup()
			return evalExportWriters{}, func() {}, fmt.Errorf("initialize logits writer: %w", err)
		}
		w.logits = lw
	}
	return w, cleanup, nil
}

// logSoftmaxRow writes log_softmax(row) into dst (must have len >= vocab). Uses
// max-shifted log-sum-exp in float64 for numerical stability.
func logSoftmaxRow(row []float32, vocab int, dst []float32) {
	maxLogit := row[0]
	for j := 1; j < vocab; j++ {
		if row[j] > maxLogit {
			maxLogit = row[j]
		}
	}
	sumExp := 0.0
	for j := range vocab {
		sumExp += math.Exp(float64(row[j] - maxLogit))
	}
	logNorm := float64(maxLogit) + math.Log(sumExp)
	for j := range vocab {
		dst[j] = float32(float64(row[j]) - logNorm)
	}
}
