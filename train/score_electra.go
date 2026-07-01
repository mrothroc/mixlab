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

const scoreElectraAutoBatch = 64

type ScoreElectraOptions struct {
	ConfigPath      string
	SafetensorsLoad string
	ScoreIn         string
	ScoreOut        string
	ScoreSkipFirst  int
	ScoreBatch      int
}

func runScoreElectraWithOptions(opts ScoreElectraOptions) error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	if strings.TrimSpace(opts.ConfigPath) == "" {
		return fmt.Errorf("-config is required for score-electra mode")
	}
	if strings.TrimSpace(opts.SafetensorsLoad) == "" {
		return fmt.Errorf("-safetensors-load is required for score-electra mode")
	}
	if strings.TrimSpace(opts.ScoreIn) == "" {
		return fmt.Errorf("-score-in is required for score-electra mode")
	}
	if strings.TrimSpace(opts.ScoreOut) == "" {
		return fmt.Errorf("-score-out is required for score-electra mode")
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
	if err := validateScoreElectraConfig(cfg); err != nil {
		return err
	}
	scoreBatch := opts.ScoreBatch
	if scoreBatch <= 0 {
		scoreBatch = scoreElectraAutoBatch
	}
	scoreCfg := *cfg
	scoreCfg.Training.BatchTokens = scoreBatch * scoreCfg.SeqLen
	if err := configureCharFeaturesForConfigPath(&scoreCfg, opts.ConfigPath, opts.SafetensorsLoad); err != nil {
		return err
	}
	prog, err := BuildTrainingIRProgramFromConfig(&scoreCfg, TrainingProgramState{
		Objective:       arch.ObjectiveMultihead,
		DropoutInactive: true,
	})
	if err != nil {
		return fmt.Errorf("build ELECTRA scoring IR program: %w", err)
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
		return fmt.Errorf("trainer does not support ELECTRA scoring logits; ensure you are using the MLX backend")
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

	if err := scoreElectraJSONL(in, out, &scoreCfg, evaluator, opts.ScoreSkipFirst, scoreBatch); err != nil {
		return err
	}
	if err := out.Close(); err != nil {
		closeOut = false
		return fmt.Errorf("close score output %q: %w", opts.ScoreOut, err)
	}
	closeOut = false
	fmt.Printf("wrote ELECTRA detector scores to %s\n", opts.ScoreOut)
	return nil
}

func validateScoreElectraConfig(cfg *ArchConfig) error {
	if cfg == nil {
		return fmt.Errorf("nil config")
	}
	if !rtdActive(cfg) {
		return fmt.Errorf("score-electra requires training.objective=%q with training.rtd and an objective=%q head", arch.ObjectiveMultihead, arch.ObjectiveRTD)
	}
	_, err := rtdHeadLogitsOutputName(cfg)
	return err
}

func scoreElectraJSONL(r io.Reader, w io.Writer, cfg *ArchConfig, evaluator diffusionGenerationEvaluator, globalSkipFirst, scoreBatch int) error {
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
		out, err := scoreElectraRecord(cfg, evaluator, rec, scoreFrom, scoreBatch)
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

func scoreElectraRecord(cfg *ArchConfig, evaluator diffusionGenerationEvaluator, rec scoreDiffusionInputRecord, scoreFrom, scoreBatch int) (scoreDiffusionOutputRecord, error) {
	if strings.TrimSpace(rec.ID) == "" {
		return scoreDiffusionOutputRecord{}, fmt.Errorf("id must be non-empty")
	}
	if len(rec.Tokens) == 0 {
		return scoreDiffusionOutputRecord{}, fmt.Errorf("tokens must be non-empty")
	}
	if err := validateScoreDiffusionTokens(cfg, rec.Tokens, scoreFrom); err != nil {
		return scoreDiffusionOutputRecord{}, err
	}
	perToken, err := scoreElectraTokens(cfg, evaluator, rec.Tokens, scoreFrom, scoreBatch)
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

func scoreElectraTokens(cfg *ArchConfig, evaluator diffusionGenerationEvaluator, tokens []int, scoreFrom, scoreBatch int) ([]float64, error) {
	if err := validateScoreElectraConfig(cfg); err != nil {
		return nil, err
	}
	if err := validateScoreDiffusionTokens(cfg, tokens, scoreFrom); err != nil {
		return nil, err
	}
	if scoreFrom == len(tokens) {
		return []float64{}, nil
	}
	if scoreBatch <= 0 {
		scoreBatch = scoreElectraAutoBatch
	}
	batch, err := electraScoringBatch(cfg, tokens, scoreBatch)
	if err != nil {
		return nil, err
	}
	evalBatchSize := scoreBatch
	if batch.batchSizeOverride > 0 {
		evalBatchSize = batch.batchSizeOverride
	}
	if _, err := evaluator.EvaluateObjectiveGPU(batch, evalBatchSize, cfg.SeqLen); err != nil {
		return nil, err
	}
	outputName, err := rtdHeadLogitsOutputName(cfg)
	if err != nil {
		return nil, err
	}
	logits, err := evaluator.ReadOutput(outputName, []int{scoreBatch * cfg.SeqLen, 1})
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", outputName, err)
	}
	if len(logits) != scoreBatch*cfg.SeqLen {
		return nil, fmt.Errorf("ELECTRA detector logits length mismatch: got=%d want=%d", len(logits), scoreBatch*cfg.SeqLen)
	}
	out := make([]float64, 0, len(tokens)-scoreFrom)
	for pos := scoreFrom; pos < len(tokens); pos++ {
		out = append(out, logSigmoid(float64(logits[pos])))
	}
	return out, nil
}

func electraScoringBatch(cfg *ArchConfig, tokens []int, rawBatchSize int) (objectiveBatch, error) {
	if cfg == nil {
		return objectiveBatch{}, fmt.Errorf("nil config")
	}
	if rawBatchSize <= 0 {
		return objectiveBatch{}, fmt.Errorf("invalid score batch=%d", rawBatchSize)
	}
	if len(tokens) > cfg.SeqLen {
		return objectiveBatch{}, fmt.Errorf("token length %d exceeds seq_len %d", len(tokens), cfg.SeqLen)
	}
	headCount := len(cfg.Training.Heads)
	need := rawBatchSize * cfg.SeqLen
	totalNeed := need * headCount
	x := make([]int, totalNeed)
	y := make([]int, totalNeed)
	lossMask := make([]float32, totalNeed)
	starts := make([]int32, rawBatchSize*headCount)
	ends := make([]int32, rawBatchSize*headCount)
	timestep := make([]float32, rawBatchSize*headCount)
	for headIdx := 0; headIdx < headCount; headIdx++ {
		for row := 0; row < rawBatchSize; row++ {
			rowStart := headIdx*need + row*cfg.SeqLen
			copy(x[rowStart:rowStart+len(tokens)], tokens)
			for pos := 0; pos < len(tokens); pos++ {
				y[rowStart+pos] = 1
			}
			boundaryIdx := headIdx*rawBatchSize + row
			starts[boundaryIdx] = 0
			ends[boundaryIdx] = int32(cfg.SeqLen)
		}
	}
	return objectiveBatch{
		x:                   x,
		y:                   y,
		lossMask:            lossMask,
		diffusionBlockStart: starts,
		diffusionBlockEnd:   ends,
		diffusionTimestep:   timestep,
		batchSizeOverride:   rawBatchSize * headCount,
	}, nil
}

func logSigmoid(x float64) float64 {
	if x >= 0 {
		return -math.Log1p(math.Exp(-x))
	}
	return x - math.Log1p(math.Exp(x))
}
