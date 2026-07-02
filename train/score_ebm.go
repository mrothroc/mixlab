package train

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/mrothroc/mixlab/arch"
)

const scoreEBMDefaultBatch = 64

type ScoreEBMOptions struct {
	ConfigPath      string
	SafetensorsLoad string
	ScoreIn         string
	ScoreOut        string
	ScoreBatch      int
}

type scoreEBMInputRecord struct {
	ID      string `json:"id"`
	Tokens  []int  `json:"tokens,omitempty"`
	Clean   []int  `json:"clean,omitempty"`
	Corrupt []int  `json:"corrupt,omitempty"`
	Family  string `json:"family,omitempty"`
}

type scoreEBMOutputRecord struct {
	ID            string   `json:"id"`
	NTokens       int      `json:"n_tokens,omitempty"`
	Energy        *float64 `json:"energy,omitempty"`
	Family        string   `json:"family,omitempty"`
	EnergyClean   *float64 `json:"energy_clean,omitempty"`
	EnergyCorrupt *float64 `json:"energy_corrupt,omitempty"`
	Margin        *float64 `json:"margin,omitempty"`
	Correct       *bool    `json:"correct,omitempty"`
}

func runScoreEBMWithOptions(opts ScoreEBMOptions) error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	if strings.TrimSpace(opts.ConfigPath) == "" {
		return fmt.Errorf("-config is required for score-ebm mode")
	}
	if strings.TrimSpace(opts.SafetensorsLoad) == "" {
		return fmt.Errorf("-safetensors-load is required for score-ebm mode")
	}
	if strings.TrimSpace(opts.ScoreIn) == "" {
		return fmt.Errorf("-score-in is required for score-ebm mode")
	}
	if strings.TrimSpace(opts.ScoreOut) == "" {
		return fmt.Errorf("-score-out is required for score-ebm mode")
	}
	if samePath(opts.ScoreIn, opts.ScoreOut) {
		return fmt.Errorf("-score-in and -score-out must be different paths")
	}
	scoreBatch, err := effectiveScoreEBMBatch(opts.ScoreBatch)
	if err != nil {
		return err
	}

	cfg, err := LoadArchConfig(opts.ConfigPath)
	if err != nil {
		return err
	}
	if err := validateScoreEBMConfig(cfg); err != nil {
		return err
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
		return fmt.Errorf("build EBM scoring IR program: %w", err)
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
		return fmt.Errorf("trainer does not support EBM scoring outputs; ensure you are using the MLX backend")
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
	if err := scoreEBMJSONL(in, out, &scoreCfg, evaluator, scoreBatch); err != nil {
		return err
	}
	if err := out.Close(); err != nil {
		closeOut = false
		return fmt.Errorf("close score output %q: %w", opts.ScoreOut, err)
	}
	closeOut = false
	fmt.Printf("wrote EBM energy scores to %s\n", opts.ScoreOut)
	return nil
}

func effectiveScoreEBMBatch(raw int) (int, error) {
	if raw <= 0 {
		raw = scoreEBMDefaultBatch
	}
	if raw < 2 || raw%2 != 0 {
		return 0, fmt.Errorf("-score-batch for score-ebm must be an even value >= 2")
	}
	return raw, nil
}

func validateScoreEBMConfig(cfg *ArchConfig) error {
	if cfg == nil {
		return fmt.Errorf("nil config")
	}
	if !cfg.Training.MultiheadEnabled() {
		return fmt.Errorf("score-ebm requires training.objective=%q with an objective=%q head", arch.ObjectiveMultihead, arch.ObjectiveEnergy)
	}
	_, err := energyHeadLogitsOutputName(cfg)
	return err
}

func energyHeadLogitsOutputName(cfg *ArchConfig) (string, error) {
	if cfg == nil {
		return "", fmt.Errorf("nil config")
	}
	for _, head := range cfg.Training.Heads {
		if head.Objective == arch.ObjectiveEnergy {
			return "head_" + head.Name + "_logits", nil
		}
	}
	return "", fmt.Errorf("score-ebm requires a multihead objective=%q head", arch.ObjectiveEnergy)
}

func scoreEBMJSONL(r io.Reader, w io.Writer, cfg *ArchConfig, evaluator diffusionGenerationEvaluator, scoreBatch int) error {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 1024), scoreDiffusionMaxJSONLLine)
	enc := json.NewEncoder(w)
	lineNo := 0
	summary := newScoreEBMSummary()
	for scanner.Scan() {
		lineNo++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var rec scoreEBMInputRecord
		if err := json.Unmarshal([]byte(line), &rec); err != nil {
			return fmt.Errorf("score input line %d: invalid JSON: %w", lineNo, err)
		}
		out, err := scoreEBMRecord(cfg, evaluator, rec, scoreBatch)
		if err != nil {
			return fmt.Errorf("score input line %d: %w", lineNo, err)
		}
		summary.observe(out)
		if err := enc.Encode(out); err != nil {
			return fmt.Errorf("write score output for line %d: %w", lineNo, err)
		}
	}
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("read score input: %w", err)
	}
	if summary.pairs > 0 {
		return enc.Encode(summary.outputRecord())
	}
	return nil
}

func scoreEBMRecord(cfg *ArchConfig, evaluator diffusionGenerationEvaluator, rec scoreEBMInputRecord, scoreBatch int) (scoreEBMOutputRecord, error) {
	if strings.TrimSpace(rec.ID) == "" {
		return scoreEBMOutputRecord{}, fmt.Errorf("id must be non-empty")
	}
	hasTokens := len(rec.Tokens) > 0
	hasPair := len(rec.Clean) > 0 || len(rec.Corrupt) > 0
	if hasTokens == hasPair {
		return scoreEBMOutputRecord{}, fmt.Errorf("provide exactly one of tokens or clean/corrupt")
	}
	if hasTokens {
		energy, err := scoreEBMSequences(cfg, evaluator, [][]int{rec.Tokens}, scoreBatch)
		if err != nil {
			return scoreEBMOutputRecord{}, err
		}
		v := float64(energy[0])
		return scoreEBMOutputRecord{ID: rec.ID, NTokens: len(rec.Tokens), Energy: &v}, nil
	}
	if len(rec.Clean) == 0 || len(rec.Corrupt) == 0 {
		return scoreEBMOutputRecord{}, fmt.Errorf("pair records require both clean and corrupt tokens")
	}
	energies, err := scoreEBMSequences(cfg, evaluator, [][]int{rec.Clean, rec.Corrupt}, scoreBatch)
	if err != nil {
		return scoreEBMOutputRecord{}, err
	}
	clean := float64(energies[0])
	corrupt := float64(energies[1])
	margin := corrupt - clean
	correct := clean < corrupt
	return scoreEBMOutputRecord{
		ID:            rec.ID,
		Family:        rec.Family,
		EnergyClean:   &clean,
		EnergyCorrupt: &corrupt,
		Margin:        &margin,
		Correct:       &correct,
	}, nil
}

func scoreEBMSequences(cfg *ArchConfig, evaluator diffusionGenerationEvaluator, sequences [][]int, scoreBatch int) ([]float32, error) {
	if err := validateScoreEBMConfig(cfg); err != nil {
		return nil, err
	}
	if len(sequences) == 0 {
		return nil, fmt.Errorf("no sequences to score")
	}
	if len(sequences) > scoreBatch {
		return nil, fmt.Errorf("sequence count %d exceeds score batch %d", len(sequences), scoreBatch)
	}
	for i, tokens := range sequences {
		if err := validateScoreDiffusionTokens(cfg, tokens, 0); err != nil {
			return nil, fmt.Errorf("sequence %d: %w", i, err)
		}
	}
	batch, err := ebmScoringBatch(cfg, sequences, scoreBatch)
	if err != nil {
		return nil, err
	}
	evalBatchSize := scoreBatch
	if batch.batchSizeOverride > 0 {
		evalBatchSize = batch.batchSizeOverride
	}
	outputName, err := energyHeadLogitsOutputName(cfg)
	if err != nil {
		return nil, err
	}
	if _, err := evaluateObjectiveAndCacheOutputs(evaluator, batch, evalBatchSize, cfg.SeqLen, outputName); err != nil {
		return nil, err
	}
	logits, err := evaluator.ReadOutput(outputName, []int{scoreBatch, 1})
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", outputName, err)
	}
	if len(logits) != scoreBatch {
		return nil, fmt.Errorf("EBM energy output length mismatch: got=%d want=%d", len(logits), scoreBatch)
	}
	return append([]float32(nil), logits[:len(sequences)]...), nil
}

func ebmScoringBatch(cfg *ArchConfig, sequences [][]int, rawBatchSize int) (objectiveBatch, error) {
	if cfg == nil {
		return objectiveBatch{}, fmt.Errorf("nil config")
	}
	if rawBatchSize <= 0 || rawBatchSize%2 != 0 {
		return objectiveBatch{}, fmt.Errorf("score-ebm requires an even raw batch size, got %d", rawBatchSize)
	}
	if len(sequences) == 0 || len(sequences) > rawBatchSize {
		return objectiveBatch{}, fmt.Errorf("invalid sequence count %d for raw batch %d", len(sequences), rawBatchSize)
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
	pad := minimalPairPadTokenID(cfg)
	for headIdx, head := range cfg.Training.Heads {
		tokenOffset := headIdx * need
		rowOffset := headIdx * rawBatchSize
		for row := 0; row < rawBatchSize; row++ {
			seq := sequences[0]
			if row < len(sequences) {
				seq = sequences[row]
			}
			rowStart := tokenOffset + row*cfg.SeqLen
			fillMinimalPairRow(x[rowStart:rowStart+cfg.SeqLen], y[rowStart:rowStart+cfg.SeqLen], seq, pad)
			if head.Objective == arch.ObjectiveEnergy && row < len(sequences) {
				lossMask[rowStart] = 1
			}
			starts[rowOffset+row] = 0
			switch head.Objective {
			case arch.ObjectiveCausal:
				ends[rowOffset+row] = 0
			default:
				ends[rowOffset+row] = int32(cfg.SeqLen)
			}
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

type scoreEBMSummary struct {
	pairs           int
	correct         int
	marginSum       float64
	familySummaries map[string]*scoreEBMFamilySummary
}

type scoreEBMFamilySummary struct {
	Pairs      int     `json:"pairs"`
	Correct    int     `json:"correct"`
	Accuracy   float64 `json:"accuracy"`
	MeanMargin float64 `json:"mean_margin"`
	marginSum  float64
}

func newScoreEBMSummary() *scoreEBMSummary {
	return &scoreEBMSummary{familySummaries: make(map[string]*scoreEBMFamilySummary)}
}

func (s *scoreEBMSummary) observe(out scoreEBMOutputRecord) {
	if out.Correct == nil || out.Margin == nil {
		return
	}
	s.pairs++
	if *out.Correct {
		s.correct++
	}
	s.marginSum += *out.Margin
	family := strings.TrimSpace(out.Family)
	if family == "" {
		family = "unknown"
	}
	fam := s.familySummaries[family]
	if fam == nil {
		fam = &scoreEBMFamilySummary{}
		s.familySummaries[family] = fam
	}
	fam.Pairs++
	if *out.Correct {
		fam.Correct++
	}
	fam.marginSum += *out.Margin
}

func (s *scoreEBMSummary) outputRecord() map[string]interface{} {
	acc := 0.0
	meanMargin := 0.0
	if s.pairs > 0 {
		acc = float64(s.correct) / float64(s.pairs)
		meanMargin = s.marginSum / float64(s.pairs)
	}
	families := make(map[string]scoreEBMFamilySummary, len(s.familySummaries))
	for name, fam := range s.familySummaries {
		if fam.Pairs > 0 {
			fam.Accuracy = float64(fam.Correct) / float64(fam.Pairs)
			fam.MeanMargin = fam.marginSum / float64(fam.Pairs)
		}
		families[name] = scoreEBMFamilySummary{
			Pairs:      fam.Pairs,
			Correct:    fam.Correct,
			Accuracy:   fam.Accuracy,
			MeanMargin: fam.MeanMargin,
		}
	}
	return map[string]interface{}{
		"id":          "__summary__",
		"pairs":       s.pairs,
		"accuracy":    acc,
		"mean_margin": meanMargin,
		"families":    families,
	}
}
