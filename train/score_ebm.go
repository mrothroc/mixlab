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

const (
	scoreEBMModeEnergy     = "energy"
	scoreEBMModeMLMSpanPLL = "mlm_span_pll"
)

type ScoreEBMOptions struct {
	ConfigPath      string
	SafetensorsLoad string
	ScoreIn         string
	ScoreOut        string
	ScoreBatch      int
	EmitTokenEnergy bool
}

type scoreEBMInputRecord struct {
	ID          string `json:"id"`
	Tokens      []int  `json:"tokens,omitempty"`
	Span        []int  `json:"span,omitempty"`
	Clean       []int  `json:"clean,omitempty"`
	Corrupt     []int  `json:"corrupt,omitempty"`
	CleanSpan   []int  `json:"clean_span,omitempty"`
	CorruptSpan []int  `json:"corrupt_span,omitempty"`
	Family      string `json:"family,omitempty"`
}

type scoreEBMOutputRecord struct {
	ID                 string    `json:"id"`
	NTokens            int       `json:"n_tokens,omitempty"`
	Energy             *float64  `json:"energy,omitempty"`
	Score              *float64  `json:"score,omitempty"`
	TokenEnergy        []float64 `json:"token_energy,omitempty"`
	Family             string    `json:"family,omitempty"`
	EnergyClean        *float64  `json:"energy_clean,omitempty"`
	EnergyCorrupt      *float64  `json:"energy_corrupt,omitempty"`
	ScoreClean         *float64  `json:"score_clean,omitempty"`
	ScoreCorrupt       *float64  `json:"score_corrupt,omitempty"`
	CleanTokenEnergy   []float64 `json:"clean_token_energy,omitempty"`
	CorruptTokenEnergy []float64 `json:"corrupt_token_energy,omitempty"`
	Margin             *float64  `json:"margin,omitempty"`
	Correct            *bool     `json:"correct,omitempty"`
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
	scoreMode, err := scoreEBMMode(cfg)
	if err != nil {
		return err
	}
	if opts.EmitTokenEnergy && scoreMode != scoreEBMModeEnergy {
		return fmt.Errorf("-score-emit-token-energy is only supported for native energy score-ebm configs")
	}
	if opts.EmitTokenEnergy && !scoreEBMUsesDifferingSpan(cfg) {
		return fmt.Errorf("-score-emit-token-energy requires training.minimal_pair.energy_aggregation=%q", arch.MinimalPairEnergySpan)
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
	if err := scoreEBMJSONL(in, out, &scoreCfg, evaluator, scoreBatch, opts.EmitTokenEnergy); err != nil {
		return err
	}
	if err := out.Close(); err != nil {
		closeOut = false
		return fmt.Errorf("close score output %q: %w", opts.ScoreOut, err)
	}
	closeOut = false
	fmt.Printf("wrote EBM scores to %s\n", opts.ScoreOut)
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
		return fmt.Errorf("score-ebm requires training.objective=%q with an energy or mlm_span_pll minimal-pair scorer", arch.ObjectiveMultihead)
	}
	_, err := scoreEBMMode(cfg)
	return err
}

func scoreEBMMode(cfg *ArchConfig) (string, error) {
	if cfg == nil {
		return "", fmt.Errorf("nil config")
	}
	if cfg.Training.MinimalPair != nil && cfg.Training.MinimalPair.UsesMLMSpanPLL() {
		if _, err := mlmSpanPLLScoreOutputName(cfg); err != nil {
			return "", err
		}
		return scoreEBMModeMLMSpanPLL, nil
	}
	if _, err := energyHeadLogitsOutputName(cfg); err != nil {
		return "", err
	}
	return scoreEBMModeEnergy, nil
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

func energyHeadTokenOutputName(cfg *ArchConfig) string {
	if cfg == nil {
		return ""
	}
	for _, head := range cfg.Training.Heads {
		if head.Objective == arch.ObjectiveEnergy {
			return "head_" + head.Name + "_token_energy"
		}
	}
	return ""
}

func mlmSpanPLLScoreOutputName(cfg *ArchConfig) (string, error) {
	if cfg == nil || cfg.Training.MinimalPair == nil || !cfg.Training.MinimalPair.UsesMLMSpanPLL() {
		return "", fmt.Errorf("score-ebm requires training.minimal_pair.score_source=%q or a native energy head", arch.MinimalPairScoreMLMPLL)
	}
	headName := strings.TrimSpace(cfg.Training.MinimalPair.ScoreHead)
	if headName == "" {
		headName = strings.TrimSpace(cfg.Training.ExportHead)
	}
	for _, head := range cfg.Training.Heads {
		if head.Name != headName {
			continue
		}
		if head.Objective != arch.ObjectiveMLM && head.Objective != arch.ObjectiveMNTP {
			return "", fmt.Errorf("score-ebm training.minimal_pair.score_head=%q must select an mlm or mntp head", headName)
		}
		return "head_" + head.Name + "_minimal_pair_scores", nil
	}
	return "", fmt.Errorf("score-ebm training.minimal_pair.score_head=%q does not match any training head", headName)
}

func scoreEBMUsesDifferingSpan(cfg *ArchConfig) bool {
	return cfg != nil && cfg.Training.MinimalPair != nil && cfg.Training.MinimalPair.UsesDifferingSpanEnergy()
}

func scoreEBMSequenceSpans(tokens, span []int, spanMode bool) ([][]int, error) {
	if !spanMode {
		return nil, nil
	}
	if len(span) == 0 {
		return [][]int{{0, len(tokens)}}, nil
	}
	if _, _, _, err := parseMinimalPairSpan("span", span, len(tokens)); err != nil {
		return nil, err
	}
	return [][]int{append([]int(nil), span...)}, nil
}

func scoreEBMPairSpans(rec scoreEBMInputRecord, spanMode bool) ([][]int, error) {
	if !spanMode {
		return nil, nil
	}
	pair := minimalPairRecord{
		ID:          rec.ID,
		Clean:       rec.Clean,
		Corrupt:     rec.Corrupt,
		CleanSpan:   append([]int(nil), rec.CleanSpan...),
		CorruptSpan: append([]int(nil), rec.CorruptSpan...),
		Family:      rec.Family,
	}
	if err := ensureMinimalPairRecordSpans(&pair); err != nil {
		return nil, err
	}
	return [][]int{pair.CleanSpan, pair.CorruptSpan}, nil
}

func float32SliceToFloat64(in []float32) []float64 {
	out := make([]float64, len(in))
	for i, v := range in {
		out[i] = float64(v)
	}
	return out
}

func scoreEBMJSONL(r io.Reader, w io.Writer, cfg *ArchConfig, evaluator diffusionGenerationEvaluator, scoreBatch int, emitTokenEnergy bool) error {
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
		out, err := scoreEBMRecord(cfg, evaluator, rec, scoreBatch, emitTokenEnergy)
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

func scoreEBMRecord(cfg *ArchConfig, evaluator diffusionGenerationEvaluator, rec scoreEBMInputRecord, scoreBatch int, emitTokenEnergy bool) (scoreEBMOutputRecord, error) {
	if strings.TrimSpace(rec.ID) == "" {
		return scoreEBMOutputRecord{}, fmt.Errorf("id must be non-empty")
	}
	spanMode := scoreEBMUsesDifferingSpan(cfg)
	mode, err := scoreEBMMode(cfg)
	if err != nil {
		return scoreEBMOutputRecord{}, err
	}
	if emitTokenEnergy && mode != scoreEBMModeEnergy {
		return scoreEBMOutputRecord{}, fmt.Errorf("-score-emit-token-energy is only supported for native energy score-ebm configs")
	}
	if emitTokenEnergy && !spanMode {
		return scoreEBMOutputRecord{}, fmt.Errorf("-score-emit-token-energy requires training.minimal_pair.energy_aggregation=%q", arch.MinimalPairEnergySpan)
	}
	hasTokens := len(rec.Tokens) > 0
	hasPair := len(rec.Clean) > 0 || len(rec.Corrupt) > 0
	if hasTokens == hasPair {
		return scoreEBMOutputRecord{}, fmt.Errorf("provide exactly one of tokens or clean/corrupt")
	}
	if hasTokens {
		if !spanMode && len(rec.Span) > 0 {
			return scoreEBMOutputRecord{}, fmt.Errorf("span requires training.minimal_pair.energy_aggregation=%q", arch.MinimalPairEnergySpan)
		}
		spans, err := scoreEBMSequenceSpans(rec.Tokens, rec.Span, spanMode)
		if err != nil {
			return scoreEBMOutputRecord{}, err
		}
		energy, tokenEnergy, err := scoreEBMSequencesDetailed(cfg, evaluator, [][]int{rec.Tokens}, spans, scoreBatch, emitTokenEnergy)
		if err != nil {
			return scoreEBMOutputRecord{}, err
		}
		v := float64(energy[0])
		out := scoreEBMOutputRecord{ID: rec.ID, NTokens: len(rec.Tokens)}
		if mode == scoreEBMModeEnergy {
			out.Energy = &v
		} else {
			out.Score = &v
		}
		if emitTokenEnergy {
			out.TokenEnergy = float32SliceToFloat64(tokenEnergy[0][:len(rec.Tokens)])
		}
		return out, nil
	}
	if len(rec.Clean) == 0 || len(rec.Corrupt) == 0 {
		return scoreEBMOutputRecord{}, fmt.Errorf("pair records require both clean and corrupt tokens")
	}
	if !spanMode && (len(rec.CleanSpan) > 0 || len(rec.CorruptSpan) > 0) {
		return scoreEBMOutputRecord{}, fmt.Errorf("clean_span/corrupt_span require training.minimal_pair.energy_aggregation=%q", arch.MinimalPairEnergySpan)
	}
	spans, err := scoreEBMPairSpans(rec, spanMode)
	if err != nil {
		return scoreEBMOutputRecord{}, err
	}
	energies, tokenEnergy, err := scoreEBMSequencesDetailed(cfg, evaluator, [][]int{rec.Clean, rec.Corrupt}, spans, scoreBatch, emitTokenEnergy)
	if err != nil {
		return scoreEBMOutputRecord{}, err
	}
	clean := float64(energies[0])
	corrupt := float64(energies[1])
	out := scoreEBMOutputRecord{
		ID:     rec.ID,
		Family: rec.Family,
	}
	if mode == scoreEBMModeEnergy {
		margin := corrupt - clean
		correct := clean < corrupt
		out.EnergyClean = &clean
		out.EnergyCorrupt = &corrupt
		out.Margin = &margin
		out.Correct = &correct
	} else {
		margin := clean - corrupt
		correct := clean > corrupt
		out.ScoreClean = &clean
		out.ScoreCorrupt = &corrupt
		out.Margin = &margin
		out.Correct = &correct
	}
	if emitTokenEnergy {
		out.CleanTokenEnergy = float32SliceToFloat64(tokenEnergy[0][:len(rec.Clean)])
		out.CorruptTokenEnergy = float32SliceToFloat64(tokenEnergy[1][:len(rec.Corrupt)])
	}
	return out, nil
}

func scoreEBMSequencesDetailed(cfg *ArchConfig, evaluator diffusionGenerationEvaluator, sequences [][]int, spans [][]int, scoreBatch int, emitTokenEnergy bool) ([]float32, [][]float32, error) {
	if err := validateScoreEBMConfig(cfg); err != nil {
		return nil, nil, err
	}
	if len(sequences) == 0 {
		return nil, nil, fmt.Errorf("no sequences to score")
	}
	if len(sequences) > scoreBatch {
		return nil, nil, fmt.Errorf("sequence count %d exceeds score batch %d", len(sequences), scoreBatch)
	}
	spanMode := scoreEBMUsesDifferingSpan(cfg)
	mode, err := scoreEBMMode(cfg)
	if err != nil {
		return nil, nil, err
	}
	if emitTokenEnergy && mode != scoreEBMModeEnergy {
		return nil, nil, fmt.Errorf("token energy output is only supported for native energy score-ebm configs")
	}
	if emitTokenEnergy && !spanMode {
		return nil, nil, fmt.Errorf("token energy output requires training.minimal_pair.energy_aggregation=%q", arch.MinimalPairEnergySpan)
	}
	for i, tokens := range sequences {
		if err := validateScoreDiffusionTokens(cfg, tokens, 0); err != nil {
			return nil, nil, fmt.Errorf("sequence %d: %w", i, err)
		}
		if spanMode {
			if len(spans) <= i {
				return nil, nil, fmt.Errorf("sequence %d missing energy span", i)
			}
			if _, _, _, err := parseMinimalPairSpan("span", spans[i], len(tokens)); err != nil {
				return nil, nil, fmt.Errorf("sequence %d: %w", i, err)
			}
		}
	}
	batch, err := ebmScoringBatch(cfg, sequences, spans, scoreBatch)
	if err != nil {
		return nil, nil, err
	}
	evalBatchSize := scoreBatch
	if batch.batchSizeOverride > 0 {
		evalBatchSize = batch.batchSizeOverride
	}
	outputName := ""
	if mode == scoreEBMModeEnergy {
		outputName, err = energyHeadLogitsOutputName(cfg)
		if err != nil {
			return nil, nil, err
		}
	} else {
		outputName, err = mlmSpanPLLScoreOutputName(cfg)
		if err != nil {
			return nil, nil, err
		}
	}
	outputs := []string{outputName}
	tokenOutputName := energyHeadTokenOutputName(cfg)
	if emitTokenEnergy && mode == scoreEBMModeEnergy {
		outputs = append(outputs, tokenOutputName)
	}
	if _, err := evaluateObjectiveAndCacheOutputs(evaluator, batch, evalBatchSize, cfg.SeqLen, outputs...); err != nil {
		return nil, nil, err
	}
	logits, err := evaluator.ReadOutput(outputName, []int{scoreBatch, 1})
	if err != nil {
		return nil, nil, fmt.Errorf("read %s: %w", outputName, err)
	}
	if len(logits) != scoreBatch {
		return nil, nil, fmt.Errorf("EBM score output length mismatch: got=%d want=%d", len(logits), scoreBatch)
	}
	var tokenEnergy [][]float32
	if emitTokenEnergy && mode == scoreEBMModeEnergy {
		rawTokenEnergy, err := evaluator.ReadOutput(tokenOutputName, []int{scoreBatch * cfg.SeqLen, 1})
		if err != nil {
			return nil, nil, fmt.Errorf("read %s: %w", tokenOutputName, err)
		}
		if len(rawTokenEnergy) != scoreBatch*cfg.SeqLen {
			return nil, nil, fmt.Errorf("EBM token energy output length mismatch: got=%d want=%d", len(rawTokenEnergy), scoreBatch*cfg.SeqLen)
		}
		tokenEnergy = make([][]float32, len(sequences))
		for i, seq := range sequences {
			start := i * cfg.SeqLen
			tokenEnergy[i] = append([]float32(nil), rawTokenEnergy[start:start+len(seq)]...)
		}
	}
	return append([]float32(nil), logits[:len(sequences)]...), tokenEnergy, nil
}

func ebmScoringBatch(cfg *ArchConfig, sequences [][]int, spans [][]int, rawBatchSize int) (objectiveBatch, error) {
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
	totalRows := rawBatchSize * headCount
	if minimalPairUsesMLMSpanPLL(cfg) {
		totalRows += rawBatchSize
	}
	totalNeed := totalRows * cfg.SeqLen
	x := make([]int, totalNeed)
	y := make([]int, totalNeed)
	lossMask := make([]float32, totalNeed)
	var energySpanMask []float32
	spanMode := scoreEBMUsesDifferingSpan(cfg)
	if spanMode {
		energySpanMask = make([]float32, totalNeed)
	}
	starts := make([]int32, totalRows)
	ends := make([]int32, totalRows)
	timestep := make([]float32, totalRows)
	pad := minimalPairPadTokenID(cfg)
	mode, err := scoreEBMMode(cfg)
	if err != nil {
		return objectiveBatch{}, err
	}
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
				if spanMode {
					span := []int{0, len(seq)}
					if row < len(spans) && len(spans[row]) > 0 {
						span = spans[row]
					}
					fillMinimalPairSpanMask(energySpanMask[rowStart:rowStart+cfg.SeqLen], span)
				}
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
	if mode == scoreEBMModeMLMSpanPLL {
		pairOffset := headCount * need
		pairRowOffset := headCount * rawBatchSize
		for row := 0; row < rawBatchSize; row++ {
			seq := sequences[0]
			span := []int{0, len(seq)}
			if row < len(sequences) {
				seq = sequences[row]
				if row < len(spans) && len(spans[row]) > 0 {
					span = spans[row]
				}
			}
			rowStart := pairOffset + row*cfg.SeqLen
			fillMinimalPairRow(x[rowStart:rowStart+cfg.SeqLen], y[rowStart:rowStart+cfg.SeqLen], seq, pad)
			if row < len(sequences) {
				fillMinimalPairSpanMask(energySpanMask[rowStart:rowStart+cfg.SeqLen], span)
				copy(lossMask[rowStart:rowStart+cfg.SeqLen], energySpanMask[rowStart:rowStart+cfg.SeqLen])
				maskMinimalPairSpanTokens(x[rowStart:rowStart+cfg.SeqLen], span, cfg.Training.MLMMaskTokenID)
			}
			starts[pairRowOffset+row] = 0
			ends[pairRowOffset+row] = int32(cfg.SeqLen)
		}
	}
	return objectiveBatch{
		x:                   x,
		y:                   y,
		lossMask:            lossMask,
		energySpanMask:      energySpanMask,
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
