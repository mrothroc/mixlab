package train

import (
	"bytes"
	_ "embed"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/mrothroc/mixlab/data"
	"github.com/mrothroc/mixlab/logprobs"
)

//go:embed testdata/hf_cli_parity.py
var hfCLIParityScript []byte

const (
	parityTokensMagic uint32 = 0x504b544d // "MTKP" little-endian
	parityLogitsMagic uint32 = 0x474c504d // "MPLG" little-endian
	parityFileVersion uint32 = 1
)

// ParityOptions describes a native-vs-Hugging Face export parity run.
type ParityOptions struct {
	ConfigPath      string
	SafetensorsLoad string
	HFDir           string
	TokenPattern    string
	Python          string
	LossThreshold   float64
	MaxLogitDiff    float64
	LogitTokens     int
}

type hfParitySummary struct {
	HFLoss             float64 `json:"hf_loss"`
	MaxLogitDiff       float64 `json:"max_logit_diff"`
	ScoredPairs        int     `json:"scored_pairs"`
	SamplePairs        int     `json:"sample_pairs"`
	BackboneHiddenSize int     `json:"backbone_hidden_size"`
}

// RunParity compares native MLX inference against an exported Hugging Face
// directory on the same token shards. The native side computes the full mean
// next-token loss, while the HF checker computes the same loss and compares a
// bounded sample of native logits to keep memory use predictable.
func RunParity(opts ParityOptions) error {
	if err := validateParityOptions(&opts); err != nil {
		return err
	}

	session, err := newInferenceSession(opts.ConfigPath, opts.SafetensorsLoad, opts.TokenPattern)
	if err != nil {
		return err
	}
	cfg := session.Config()
	batchTokens := cfg.Training.BatchTokens
	seqLen := cfg.SeqLen
	vocab := cfg.VocabSize

	tokens, err := data.LoadValidationTokens(opts.TokenPattern, seqLen)
	if err != nil {
		_ = session.Close()
		return err
	}
	totalPairs := parityCompletePairs(len(tokens), batchTokens)
	if totalPairs <= 0 {
		_ = session.Close()
		return fmt.Errorf("parity token shards contain no complete eval batch: tokens=%d batch_tokens=%d", len(tokens), batchTokens)
	}
	tokens = tokens[:totalPairs+1]
	samplePairs := paritySamplePairs(totalPairs, batchTokens, opts.LogitTokens)

	nativeLoss, nativeSampleLogits, err := evalNativeParity(session, tokens, samplePairs)
	closeErr := session.Close()
	if err != nil {
		return err
	}
	if closeErr != nil {
		return closeErr
	}

	tmpDir, err := os.MkdirTemp("", "mixlab-parity-*")
	if err != nil {
		return fmt.Errorf("create parity temp dir: %w", err)
	}
	defer func() { _ = os.RemoveAll(tmpDir) }()

	tokensPath := filepath.Join(tmpDir, "tokens.bin")
	logitsPath := filepath.Join(tmpDir, "native_logits.bin")
	scriptPath := filepath.Join(tmpDir, "hf_cli_parity.py")
	if err := writeParityTokens(tokensPath, tokens); err != nil {
		return err
	}
	if err := writeParityLogits(logitsPath, nativeSampleLogits, samplePairs, vocab); err != nil {
		return err
	}
	if err := os.WriteFile(scriptPath, hfCLIParityScript, 0o755); err != nil {
		return fmt.Errorf("write embedded HF parity checker: %w", err)
	}

	python := opts.Python
	cmd := exec.Command(python, scriptPath,
		"--dir", opts.HFDir,
		"--tokens", tokensPath,
		"--native-logits", logitsPath,
		"--batch-tokens", fmt.Sprintf("%d", batchTokens),
		"--seq-len", fmt.Sprintf("%d", seqLen),
		"--vocab-size", fmt.Sprintf("%d", vocab),
	)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("HF parity checker failed via %q: %w\n%s\nIf this is a dependency error, install requirements-hf.txt or pass -parity-python", python, err, bytes.TrimSpace(out))
	}
	summary, err := parseHFParitySummary(out)
	if err != nil {
		return fmt.Errorf("parse HF parity checker output: %w\n%s", err, bytes.TrimSpace(out))
	}
	if summary.ScoredPairs != totalPairs {
		return fmt.Errorf("HF parity checker scored %d pairs, native scored %d", summary.ScoredPairs, totalPairs)
	}
	if summary.SamplePairs != samplePairs {
		return fmt.Errorf("HF parity checker compared %d sample pairs, native wrote %d", summary.SamplePairs, samplePairs)
	}

	lossDiff := math.Abs(nativeLoss - summary.HFLoss)
	fmt.Printf("parity: native_loss=%.6f hf_loss=%.6f loss_diff=%.6g threshold=%.6g\n",
		nativeLoss, summary.HFLoss, lossDiff, opts.LossThreshold)
	fmt.Printf("parity: max_logit_diff=%.6g threshold=%.6g sample_pairs=%d scored_pairs=%d hf=%s\n",
		summary.MaxLogitDiff, opts.MaxLogitDiff, samplePairs, totalPairs, opts.HFDir)
	if summary.BackboneHiddenSize > 0 {
		fmt.Printf("parity: AutoModel hidden_size=%d\n", summary.BackboneHiddenSize)
	}

	if err := evaluateParityThresholds(nativeLoss, summary, opts.LossThreshold, opts.MaxLogitDiff); err != nil {
		return err
	}
	return nil
}

func validateParityOptions(opts *ParityOptions) error {
	if opts.ConfigPath == "" {
		return fmt.Errorf("-config is required for parity mode")
	}
	if opts.SafetensorsLoad == "" {
		return fmt.Errorf("-safetensors-load is required for parity mode")
	}
	if opts.HFDir == "" {
		return fmt.Errorf("-hf is required for parity mode; pass a directory produced by mixlab -mode export-hf")
	}
	info, err := os.Stat(opts.HFDir)
	if err != nil {
		return fmt.Errorf("verify -hf directory %q: %w", opts.HFDir, err)
	}
	if !info.IsDir() {
		return fmt.Errorf("-hf must be a directory, got %q", opts.HFDir)
	}
	if opts.TokenPattern == "" {
		return fmt.Errorf("-train is required for parity mode; pass the validation/eval shard glob to score directly")
	}
	if opts.LossThreshold < 0 || math.IsNaN(opts.LossThreshold) || math.IsInf(opts.LossThreshold, 0) {
		return fmt.Errorf("-threshold must be a finite value >= 0")
	}
	if math.IsNaN(opts.MaxLogitDiff) || math.IsInf(opts.MaxLogitDiff, 0) {
		return fmt.Errorf("-max-logit-diff must be finite; use <= 0 to disable the logit-diff gate")
	}
	if opts.LogitTokens < 0 {
		return fmt.Errorf("-parity-logit-tokens must be >= 0")
	}
	if opts.Python == "" {
		opts.Python = os.Getenv("HF_PARITY_PYTHON")
	}
	if opts.Python == "" {
		opts.Python = "python3"
	}
	return nil
}

func parityCompletePairs(tokenCount, batchTokens int) int {
	if tokenCount < 2 || batchTokens <= 0 {
		return 0
	}
	return ((tokenCount - 1) / batchTokens) * batchTokens
}

func paritySamplePairs(totalPairs, batchTokens, requestedPairs int) int {
	if totalPairs <= 0 || batchTokens <= 0 {
		return 0
	}
	if requestedPairs <= 0 {
		requestedPairs = batchTokens
	}
	pairs := ((requestedPairs + batchTokens - 1) / batchTokens) * batchTokens
	if pairs <= 0 {
		pairs = batchTokens
	}
	if pairs > totalPairs {
		pairs = totalPairs
	}
	return pairs
}

func evalNativeParity(session *InferenceSession, tokens []uint16, samplePairs int) (float64, []float32, error) {
	if err := session.checkEvalPreconditions(tokens); err != nil {
		return 0, nil, err
	}
	batchTokens := session.cfg.Training.BatchTokens
	vocab := session.cfg.VocabSize
	totalPairs := len(tokens) - 1
	if samplePairs < 0 || samplePairs > totalPairs || samplePairs%batchTokens != 0 {
		return 0, nil, fmt.Errorf("invalid samplePairs=%d for totalPairs=%d batch_tokens=%d", samplePairs, totalPairs, batchTokens)
	}

	totalLoss := 0.0
	sampleLogits := make([]float32, 0, samplePairs*vocab)
	for start := 0; start < totalPairs; start += batchTokens {
		window := tokens[start : start+batchTokens+1]
		if start < samplePairs {
			logitsBatch, err := session.evalLogitsBatch(window)
			if err != nil {
				return 0, nil, fmt.Errorf("native logits failed at token offset %d: %w", start, err)
			}
			if len(logitsBatch) != batchTokens*vocab {
				return 0, nil, fmt.Errorf("native logits length mismatch at offset %d: got=%d want=%d", start, len(logitsBatch), batchTokens*vocab)
			}
			sampleLogits = append(sampleLogits, logitsBatch...)
			for i := range batchTokens {
				nll, _, _, _, _, err := evalMetricsFromLogits(logitsBatch[i*vocab:(i+1)*vocab], vocab, window[i+1], false, false)
				if err != nil {
					return 0, nil, fmt.Errorf("native metric derivation failed at token offset %d: %w", start+i, err)
				}
				if !logprobs.IsFinite(nll) {
					return 0, nil, fmt.Errorf("native non-finite NLL at token offset %d", start+i)
				}
				totalLoss += float64(nll)
			}
			continue
		}

		nlls, err := session.EvalTokens(window)
		if err != nil {
			return 0, nil, fmt.Errorf("native eval failed at token offset %d: %w", start, err)
		}
		if len(nlls) != batchTokens {
			return 0, nil, fmt.Errorf("native NLL length mismatch at offset %d: got=%d want=%d", start, len(nlls), batchTokens)
		}
		for i, nll := range nlls {
			if !logprobs.IsFinite(nll) {
				return 0, nil, fmt.Errorf("native non-finite NLL at token offset %d", start+i)
			}
			totalLoss += float64(nll)
		}
	}
	return totalLoss / float64(totalPairs), sampleLogits, nil
}

func writeParityTokens(path string, tokens []uint16) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create parity tokens %q: %w", path, err)
	}
	defer func() { _ = f.Close() }()
	if err := binary.Write(f, binary.LittleEndian, parityTokensMagic); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, parityFileVersion); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, uint64(len(tokens))); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, tokens); err != nil {
		return fmt.Errorf("write parity tokens %q: %w", path, err)
	}
	return nil
}

func writeParityLogits(path string, logits []float32, pairs, vocab int) error {
	if pairs < 0 || vocab <= 0 {
		return fmt.Errorf("invalid parity logits shape: pairs=%d vocab=%d", pairs, vocab)
	}
	if len(logits) != pairs*vocab {
		return fmt.Errorf("parity logits length mismatch: got=%d want=%d", len(logits), pairs*vocab)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create parity logits %q: %w", path, err)
	}
	defer func() { _ = f.Close() }()
	for _, v := range []uint32{parityLogitsMagic, parityFileVersion} {
		if err := binary.Write(f, binary.LittleEndian, v); err != nil {
			return err
		}
	}
	for _, v := range []uint64{uint64(pairs), uint64(vocab)} {
		if err := binary.Write(f, binary.LittleEndian, v); err != nil {
			return err
		}
	}
	if err := binary.Write(f, binary.LittleEndian, logits); err != nil {
		return fmt.Errorf("write parity logits %q: %w", path, err)
	}
	return nil
}

func parseHFParitySummary(out []byte) (hfParitySummary, error) {
	lines := strings.Split(string(out), "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		line := strings.TrimSpace(lines[i])
		if !strings.HasPrefix(line, "{") {
			continue
		}
		var summary hfParitySummary
		if err := json.Unmarshal([]byte(line), &summary); err != nil {
			return hfParitySummary{}, err
		}
		if !isFinite(summary.HFLoss) {
			return hfParitySummary{}, fmt.Errorf("HF loss is non-finite: %v", summary.HFLoss)
		}
		if !isFinite(summary.MaxLogitDiff) {
			return hfParitySummary{}, fmt.Errorf("max logit diff is non-finite: %v", summary.MaxLogitDiff)
		}
		return summary, nil
	}
	return hfParitySummary{}, fmt.Errorf("no JSON summary found")
}

func evaluateParityThresholds(nativeLoss float64, summary hfParitySummary, lossThreshold, maxLogitDiff float64) error {
	if !isFinite(nativeLoss) {
		return fmt.Errorf("native loss is non-finite: %v", nativeLoss)
	}
	lossDiff := math.Abs(nativeLoss - summary.HFLoss)
	if lossDiff > lossThreshold {
		return fmt.Errorf("native/HF loss diff %.6g exceeds threshold %.6g", lossDiff, lossThreshold)
	}
	if maxLogitDiff > 0 && summary.MaxLogitDiff > maxLogitDiff {
		return fmt.Errorf("native/HF max logit diff %.6g exceeds threshold %.6g", summary.MaxLogitDiff, maxLogitDiff)
	}
	return nil
}

func isFinite(v float64) bool {
	return !math.IsNaN(v) && !math.IsInf(v, 0)
}
