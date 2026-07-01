package train

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/mrothroc/mixlab/arch"
)

const (
	scoreDiffusionModeBlockCausal = "block_causal"
	scoreDiffusionAutoLogitBytes  = 256 << 20
	scoreDiffusionMaxJSONLLine    = 64 << 20
)

type ScoreDiffusionOptions struct {
	ConfigPath         string
	SafetensorsLoad    string
	ScoreIn            string
	ScoreOut           string
	ScoreMode          string
	ScoreSkipFirst     int
	ScorePositionBatch int
}

type scoreDiffusionInputRecord struct {
	ID        string `json:"id"`
	Tokens    []int  `json:"tokens"`
	ScoreFrom *int   `json:"score_from,omitempty"`
}

type scoreDiffusionOutputRecord struct {
	ID          string    `json:"id"`
	NTokens     int       `json:"n_tokens"`
	ScoreFrom   int       `json:"score_from"`
	LogprobSum  float64   `json:"logprob_sum"`
	LogprobMean float64   `json:"logprob_mean"`
	PerToken    []float64 `json:"per_token"`
}

func runScoreDiffusionWithOptions(opts ScoreDiffusionOptions) error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	mode := strings.ToLower(strings.TrimSpace(opts.ScoreMode))
	if mode == "" {
		mode = scoreDiffusionModeBlockCausal
	}
	if mode != scoreDiffusionModeBlockCausal {
		return fmt.Errorf("-score-mode %q is not supported yet; supported: %s", opts.ScoreMode, scoreDiffusionModeBlockCausal)
	}
	if strings.TrimSpace(opts.ConfigPath) == "" {
		return fmt.Errorf("-config is required for score-diffusion mode")
	}
	if strings.TrimSpace(opts.SafetensorsLoad) == "" {
		return fmt.Errorf("-safetensors-load is required for score-diffusion mode")
	}
	if strings.TrimSpace(opts.ScoreIn) == "" {
		return fmt.Errorf("-score-in is required for score-diffusion mode")
	}
	if strings.TrimSpace(opts.ScoreOut) == "" {
		return fmt.Errorf("-score-out is required for score-diffusion mode")
	}
	if opts.ScoreSkipFirst < 0 {
		return fmt.Errorf("-score-skip-first must be >= 0")
	}
	if samePath(opts.ScoreIn, opts.ScoreOut) {
		return fmt.Errorf("-score-in and -score-out must be different paths")
	}

	cfg, err := LoadArchConfig(opts.ConfigPath)
	if err != nil {
		return err
	}
	if err := validateScoreDiffusionConfig(cfg); err != nil {
		return err
	}
	positionBatch, err := effectiveScorePositionBatch(cfg.SeqLen, cfg.VocabSize, opts.ScorePositionBatch)
	if err != nil {
		return err
	}

	scoreCfg := *cfg
	scoreCfg.Training.BatchTokens = positionBatch * scoreCfg.SeqLen
	if err := configureCharFeaturesForConfigPath(&scoreCfg, opts.ConfigPath, opts.SafetensorsLoad); err != nil {
		return err
	}
	prog, err := buildGenerateDiffusionIRProgram(&scoreCfg)
	if err != nil {
		return fmt.Errorf("build diffusion scoring IR program: %w", err)
	}
	shapes, err := computeWeightShapes(&scoreCfg)
	if err != nil {
		return fmt.Errorf("compute weight shapes: %w", err)
	}
	loadedWeights, err := loadSafetensorsWeights(opts.SafetensorsLoad, shapes)
	if err != nil {
		return fmt.Errorf("load safetensors %q: %w", opts.SafetensorsLoad, err)
	}
	trainer, err := initGPUTrainer(prog, &scoreCfg, loadedWeights, nil)
	if err != nil {
		return fmt.Errorf("init GPU trainer: %w", err)
	}
	defer trainer.CloseTrainer()
	evaluator, ok := trainer.(diffusionGenerationEvaluator)
	if !ok {
		return fmt.Errorf("trainer does not support diffusion scoring logits; ensure you are using the MLX backend")
	}

	in, err := os.Open(opts.ScoreIn)
	if err != nil {
		return fmt.Errorf("open score input %q: %w", opts.ScoreIn, err)
	}
	defer func() { _ = in.Close() }()
	if err := os.MkdirAll(filepath.Dir(opts.ScoreOut), 0o755); err != nil && filepath.Dir(opts.ScoreOut) != "." {
		return fmt.Errorf("create score output directory: %w", err)
	}
	out, err := os.Create(opts.ScoreOut)
	if err != nil {
		return fmt.Errorf("create score output %q: %w", opts.ScoreOut, err)
	}
	closeOut := true
	defer func() {
		if closeOut {
			_ = out.Close()
		}
	}()

	if err := scoreDiffusionJSONL(in, out, &scoreCfg, evaluator, opts.ScoreSkipFirst, positionBatch); err != nil {
		return err
	}
	if err := out.Close(); err != nil {
		closeOut = false
		return fmt.Errorf("close score output %q: %w", opts.ScoreOut, err)
	}
	closeOut = false
	fmt.Printf("wrote diffusion PLL scores to %s\n", opts.ScoreOut)
	return nil
}

func validateScoreDiffusionConfig(cfg *ArchConfig) error {
	if cfg == nil {
		return fmt.Errorf("nil config")
	}
	if !cfg.Training.UsesBlockDiffusionObjective() {
		return fmt.Errorf("score-diffusion requires training.objective=%q or hybrid_secondary_objective=%q, got objective=%q secondary=%q", arch.ObjectiveBlockDiffusion, arch.ObjectiveBlockDiffusion, cfg.Training.EffectiveObjective(), cfg.Training.EffectiveHybridSecondaryObjective())
	}
	if cfg.Training.MLMMaskTokenID < 0 || cfg.Training.MLMMaskTokenID >= cfg.VocabSize {
		return fmt.Errorf("invalid mlm_mask_token_id=%d for vocab_size=%d", cfg.Training.MLMMaskTokenID, cfg.VocabSize)
	}
	_, err := diffusionScoringSpec(cfg)
	return err
}

func scoreDiffusionJSONL(r io.Reader, w io.Writer, cfg *ArchConfig, evaluator diffusionGenerationEvaluator, globalSkipFirst, positionBatch int) error {
	if globalSkipFirst < 0 {
		return fmt.Errorf("score-skip-first must be >= 0")
	}
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 1024), scoreDiffusionMaxJSONLLine)
	enc := json.NewEncoder(w)
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var rec scoreDiffusionInputRecord
		if err := json.Unmarshal([]byte(line), &rec); err != nil {
			return fmt.Errorf("score input line %d: invalid JSON: %w", lineNo, err)
		}
		scoreFrom := globalSkipFirst
		if rec.ScoreFrom != nil {
			scoreFrom = *rec.ScoreFrom
		}
		out, err := scoreDiffusionRecord(cfg, evaluator, rec, scoreFrom, positionBatch)
		if err != nil {
			return fmt.Errorf("score input line %d: %w", lineNo, err)
		}
		if err := enc.Encode(out); err != nil {
			return fmt.Errorf("write score output for line %d: %w", lineNo, err)
		}
	}
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("read score input: %w", err)
	}
	return nil
}

func scoreDiffusionRecord(cfg *ArchConfig, evaluator diffusionGenerationEvaluator, rec scoreDiffusionInputRecord, scoreFrom, positionBatch int) (scoreDiffusionOutputRecord, error) {
	if strings.TrimSpace(rec.ID) == "" {
		return scoreDiffusionOutputRecord{}, fmt.Errorf("id must be non-empty")
	}
	if len(rec.Tokens) == 0 {
		return scoreDiffusionOutputRecord{}, fmt.Errorf("tokens must be non-empty")
	}
	if err := validateScoreDiffusionTokens(cfg, rec.Tokens, scoreFrom); err != nil {
		return scoreDiffusionOutputRecord{}, err
	}
	perToken, err := scoreDiffusionTokens(cfg, evaluator, rec.Tokens, scoreFrom, positionBatch)
	if err != nil {
		return scoreDiffusionOutputRecord{}, err
	}
	var sum float64
	for _, lp := range perToken {
		sum += lp
	}
	mean := 0.0
	if len(perToken) > 0 {
		mean = sum / float64(len(perToken))
	}
	return scoreDiffusionOutputRecord{
		ID:          rec.ID,
		NTokens:     len(perToken),
		ScoreFrom:   scoreFrom,
		LogprobSum:  sum,
		LogprobMean: mean,
		PerToken:    perToken,
	}, nil
}

func validateScoreDiffusionTokens(cfg *ArchConfig, tokens []int, scoreFrom int) error {
	if cfg == nil {
		return fmt.Errorf("nil config")
	}
	if len(tokens) > cfg.SeqLen {
		return fmt.Errorf("token length %d exceeds seq_len %d", len(tokens), cfg.SeqLen)
	}
	if scoreFrom < 0 || scoreFrom > len(tokens) {
		return fmt.Errorf("score_from=%d must be in [0,%d]", scoreFrom, len(tokens))
	}
	for i, token := range tokens {
		if token < 0 || token >= cfg.VocabSize {
			return fmt.Errorf("token %d at position %d out of range [0,%d)", token, i, cfg.VocabSize)
		}
	}
	return nil
}

func scoreDiffusionTokens(cfg *ArchConfig, evaluator diffusionGenerationEvaluator, tokens []int, scoreFrom, positionBatch int) ([]float64, error) {
	if cfg == nil {
		return nil, fmt.Errorf("nil config")
	}
	if evaluator == nil {
		return nil, fmt.Errorf("nil diffusion scoring evaluator")
	}
	if err := validateScoreDiffusionConfig(cfg); err != nil {
		return nil, err
	}
	if err := validateScoreDiffusionTokens(cfg, tokens, scoreFrom); err != nil {
		return nil, err
	}
	if positionBatch <= 0 {
		var err error
		positionBatch, err = effectiveScorePositionBatch(cfg.SeqLen, cfg.VocabSize, positionBatch)
		if err != nil {
			return nil, err
		}
	}
	if scoreFrom == len(tokens) {
		return []float64{}, nil
	}
	spec, err := diffusionScoringSpec(cfg)
	if err != nil {
		return nil, err
	}
	out := make([]float64, 0, len(tokens)-scoreFrom)
	for start := scoreFrom; start < len(tokens); start += positionBatch {
		end := start + positionBatch
		if end > len(tokens) {
			end = len(tokens)
		}
		positions := make([]int, end-start)
		for i := range positions {
			positions[i] = start + i
		}
		batch, err := diffusionScoringBatch(tokens, positions, cfg.SeqLen, cfg.Training.MLMMaskTokenID, spec.BlockSize, positionBatch)
		if err != nil {
			return nil, err
		}
		batch = expandBatchForMultiheadDiffusion(cfg, batch, positionBatch, cfg.SeqLen)
		evalBatchSize := positionBatch
		if batch.batchSizeOverride > 0 {
			evalBatchSize = batch.batchSizeOverride
		}
		if _, err := evaluator.EvaluateObjectiveGPU(batch, evalBatchSize, cfg.SeqLen); err != nil {
			return nil, err
		}
		outputName := diffusionLogitsOutputName(cfg)
		logits, err := evaluator.ReadOutput(outputName, []int{positionBatch * cfg.SeqLen, cfg.VocabSize})
		if err != nil {
			return nil, fmt.Errorf("read %s: %w", outputName, err)
		}
		want := positionBatch * cfg.SeqLen * cfg.VocabSize
		if len(logits) != want {
			return nil, fmt.Errorf("logits length mismatch: got=%d want=%d", len(logits), want)
		}
		for row, pos := range positions {
			logitStart := (row*cfg.SeqLen + pos) * cfg.VocabSize
			lp, err := targetLogProbFromLogits(logits[logitStart:logitStart+cfg.VocabSize], tokens[pos])
			if err != nil {
				return nil, fmt.Errorf("position %d: %w", pos, err)
			}
			out = append(out, lp)
		}
	}
	return out, nil
}

func diffusionScoringBatch(tokens []int, positions []int, seqLen, maskTokenID, blockSize, batchSize int) (objectiveBatch, error) {
	if len(positions) == 0 {
		return objectiveBatch{}, fmt.Errorf("positions must be non-empty")
	}
	if batchSize <= 0 {
		return objectiveBatch{}, fmt.Errorf("invalid batch size=%d", batchSize)
	}
	if len(positions) > batchSize {
		return objectiveBatch{}, fmt.Errorf("positions=%d exceeds batch size=%d", len(positions), batchSize)
	}
	if seqLen <= 0 {
		return objectiveBatch{}, fmt.Errorf("invalid seq_len=%d", seqLen)
	}
	if len(tokens) > seqLen {
		return objectiveBatch{}, fmt.Errorf("token length %d exceeds seq_len %d", len(tokens), seqLen)
	}
	if blockSize <= 0 {
		return objectiveBatch{}, fmt.Errorf("invalid diffusion block_size=%d", blockSize)
	}
	need := batchSize * seqLen
	x := make([]int, need)
	y := make([]int, need)
	lossMask := make([]float32, need)
	unmasked := make([]int, need)
	blockStart := make([]int32, batchSize)
	blockEnd := make([]int32, batchSize)
	timestep := make([]float32, batchSize)
	for row := 0; row < batchSize; row++ {
		pos := positions[0]
		if row < len(positions) {
			pos = positions[row]
		}
		if pos < 0 || pos >= len(tokens) {
			return objectiveBatch{}, fmt.Errorf("position %d outside token length %d", pos, len(tokens))
		}
		rowStart := row * seqLen
		copy(x[rowStart:rowStart+len(tokens)], tokens)
		copy(y[rowStart:rowStart+len(tokens)], tokens)
		copy(unmasked[rowStart:rowStart+len(tokens)], tokens)
		x[rowStart+pos] = maskTokenID
		if row < len(positions) {
			lossMask[rowStart+pos] = 1
		}
		start := (pos / blockSize) * blockSize
		end := start + blockSize
		if end > len(tokens) {
			end = len(tokens)
		}
		if end <= start {
			return objectiveBatch{}, fmt.Errorf("invalid diffusion scoring block [%d,%d) for position %d", start, end, pos)
		}
		blockStart[row] = int32(start)
		blockEnd[row] = int32(end)
		timestep[row] = float32(countPositiveMask(lossMask[rowStart:rowStart+seqLen])) / float32(end-start)
	}
	return objectiveBatch{
		x:                   x,
		y:                   y,
		lossMask:            lossMask,
		unmaskedX:           unmasked,
		diffusionBlockStart: blockStart,
		diffusionBlockEnd:   blockEnd,
		diffusionTimestep:   timestep,
	}, nil
}

func diffusionScoringSpec(cfg *ArchConfig) (arch.DiffusionSpec, error) {
	if cfg == nil {
		return arch.DiffusionSpec{}, fmt.Errorf("nil config")
	}
	return diffusionSpecForObjectiveBatch(cfg, cfg.SeqLen)
}

func effectiveScorePositionBatch(seqLen, vocabSize, requested int) (int, error) {
	if seqLen <= 0 {
		return 0, fmt.Errorf("invalid seq_len=%d", seqLen)
	}
	if vocabSize <= 0 {
		return 0, fmt.Errorf("invalid vocab_size=%d", vocabSize)
	}
	if requested > 0 {
		return requested, nil
	}
	rowBytes := int64(seqLen) * int64(vocabSize) * 4
	if rowBytes <= 0 {
		return 0, fmt.Errorf("invalid scoring logits shape seq_len=%d vocab_size=%d", seqLen, vocabSize)
	}
	n := int(int64(scoreDiffusionAutoLogitBytes) / rowBytes)
	if n < 1 {
		n = 1
	}
	return n, nil
}

func targetLogProbFromLogits(row []float32, target int) (float64, error) {
	if len(row) == 0 {
		return 0, fmt.Errorf("empty logits row")
	}
	if target < 0 || target >= len(row) {
		return 0, fmt.Errorf("target token %d out of range [0,%d)", target, len(row))
	}
	maxVal := row[0]
	for _, v := range row[1:] {
		if v > maxVal {
			maxVal = v
		}
	}
	var sumExp float64
	for _, v := range row {
		sumExp += math.Exp(float64(v - maxVal))
	}
	if sumExp <= 0 || math.IsNaN(sumExp) || math.IsInf(sumExp, 0) {
		return 0, fmt.Errorf("non-finite logits normalization")
	}
	return float64(row[target]) - (float64(maxVal) + math.Log(sumExp)), nil
}

func samePath(a, b string) bool {
	aa, errA := filepath.Abs(a)
	bb, errB := filepath.Abs(b)
	if errA != nil || errB != nil {
		return a == b
	}
	return aa == bb
}
