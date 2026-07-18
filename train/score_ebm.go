package train

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/data"
)

const scoreEBMDefaultBatch = 64

const (
	scoreEBMModeEnergy     = "energy"
	scoreEBMModeMLMSpanPLL = "mlm_span_pll"
	scoreEBMModeSinglePLL  = "single_mlm_pll"
)

const (
	scoreEBMPLLAggregationConfig        = "config"
	scoreEBMPLLAggregationDifferingSpan = "differing_span"
	scoreEBMPLLAggregationDependentWin  = "dependent_window"
	scoreEBMPLLAggregationFullSeq       = "full_seq"
)

type ScoreEBMOptions struct {
	ConfigPath         string
	SafetensorsLoad    string
	ScoreIn            string
	ScoreOut           string
	ScoreBatch         int
	ScorePositionBatch int
	PLLAggregation     string
	PLLWindow          int
	PLLSkipTokenIDs    string
	PLLAttributionDump string
	EmitTokenEnergy    bool
	SequenceVocabulary string
}

type scoreEBMRuntimeOptions struct {
	scoreBatch         int
	scorePositionBatch int
	pllAggregation     string
	pllWindow          int
	pllSkipTokenIDs    map[int]bool
	pllAttributionEnc  *json.Encoder
	emitTokenEnergy    bool
	sequenceVocab      *data.NucleotideVocabulary
}

type scoreEBMInputRecord struct {
	ID              string `json:"id"`
	Tokens          []int  `json:"tokens,omitempty"`
	Span            []int  `json:"span,omitempty"`
	Clean           []int  `json:"clean,omitempty"`
	Corrupt         []int  `json:"corrupt,omitempty"`
	CleanSpan       []int  `json:"clean_span,omitempty"`
	CorruptSpan     []int  `json:"corrupt_span,omitempty"`
	Family          string `json:"family,omitempty"`
	Sequence        string `json:"sequence,omitempty"`
	CleanSequence   string `json:"clean_sequence,omitempty"`
	CorruptSequence string `json:"corrupt_sequence,omitempty"`
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
	Sequence           string    `json:"sequence,omitempty"`
	CleanSequence      string    `json:"clean_sequence,omitempty"`
	CorruptSequence    string    `json:"corrupt_sequence,omitempty"`
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

	cfg, err := LoadArchConfig(opts.ConfigPath)
	if err != nil {
		return err
	}
	sequenceVocab, err := loadSequenceVocabularyForConfig(opts.SequenceVocabulary, cfg)
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
	pllAggregation, err := effectiveScoreEBMPLLAggregation(cfg, opts.PLLAggregation)
	if err != nil {
		return err
	}
	pllFromConfig := scoreEBMPLLAggregationFromConfig(opts.PLLAggregation)
	pllUsesPositionScoring := scoreEBMUsesPositionPLL(scoreMode, pllAggregation, pllFromConfig)
	if err := validateScoreEBMPLLWindow(pllAggregation, opts.PLLWindow); err != nil {
		return err
	}
	skipIDs, err := parseScoreEBMSkipTokenIDs(opts.PLLSkipTokenIDs, cfg.VocabSize)
	if err != nil {
		return err
	}
	if cfg.Training.MLMMaskTokenID >= 0 && cfg.Training.MLMMaskTokenID < cfg.VocabSize {
		skipIDs[cfg.Training.MLMMaskTokenID] = true
	}
	scoreBatch := 0
	scorePositionBatch := 0
	if pllUsesPositionScoring {
		scorePositionBatch, err = effectiveScorePositionBatch(cfg.SeqLen, cfg.VocabSize, opts.ScorePositionBatch)
		if err != nil {
			return err
		}
	} else {
		scoreBatch, err = effectiveScoreEBMBatch(opts.ScoreBatch)
		if err != nil {
			return err
		}
	}
	if opts.EmitTokenEnergy && scoreMode != scoreEBMModeEnergy {
		return fmt.Errorf("-score-emit-token-energy is only supported for native energy score-ebm configs")
	}
	if opts.EmitTokenEnergy && !scoreEBMUsesDifferingSpan(cfg) {
		return fmt.Errorf("-score-emit-token-energy requires training.minimal_pair.energy_aggregation=%q", arch.MinimalPairEnergySpan)
	}
	if strings.TrimSpace(opts.PLLAttributionDump) != "" && scoreMode == scoreEBMModeEnergy {
		return fmt.Errorf("-score-pll-attribution-dump is only supported for MLM/MNTP PLL scorer configs")
	}
	if strings.TrimSpace(opts.PLLAttributionDump) != "" {
		if samePath(opts.ScoreIn, opts.PLLAttributionDump) || samePath(opts.ScoreOut, opts.PLLAttributionDump) {
			return fmt.Errorf("-score-pll-attribution-dump must be distinct from -score-in and -score-out")
		}
	}
	scoreCfg := *cfg
	if pllUsesPositionScoring {
		scoreCfg.Training.BatchTokens = scorePositionBatch * scoreCfg.SeqLen
	} else {
		scoreCfg.Training.BatchTokens = scoreBatch * scoreCfg.SeqLen
	}
	if err := configureCharFeaturesForConfigPath(&scoreCfg, opts.ConfigPath, opts.SafetensorsLoad); err != nil {
		return err
	}
	programAggregation := pllAggregation
	if pllUsesPositionScoring {
		programAggregation = scoreEBMPLLAggregationFullSeq
	}
	prog, err := buildScoreEBMIRProgram(&scoreCfg, scoreMode, programAggregation)
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
	var attrOut *os.File
	var attrEnc *json.Encoder
	if strings.TrimSpace(opts.PLLAttributionDump) != "" {
		if err := os.MkdirAll(filepath.Dir(opts.PLLAttributionDump), 0o755); err != nil && filepath.Dir(opts.PLLAttributionDump) != "." {
			return fmt.Errorf("create PLL attribution directory: %w", err)
		}
		attrOut, err = os.Create(opts.PLLAttributionDump)
		if err != nil {
			return fmt.Errorf("create PLL attribution dump %q: %w", opts.PLLAttributionDump, err)
		}
		defer func() {
			if attrOut != nil {
				_ = attrOut.Close()
			}
		}()
		attrEnc = json.NewEncoder(attrOut)
	}
	runtimeOpts := scoreEBMRuntimeOptions{
		scoreBatch:         scoreBatch,
		scorePositionBatch: scorePositionBatch,
		pllAggregation:     opts.PLLAggregation,
		pllWindow:          opts.PLLWindow,
		pllSkipTokenIDs:    skipIDs,
		pllAttributionEnc:  attrEnc,
		emitTokenEnergy:    opts.EmitTokenEnergy,
		sequenceVocab:      sequenceVocab,
	}
	if err := scoreEBMJSONLWithOptions(in, out, &scoreCfg, evaluator, runtimeOpts); err != nil {
		return err
	}
	if err := out.Close(); err != nil {
		closeOut = false
		return fmt.Errorf("close score output %q: %w", opts.ScoreOut, err)
	}
	if attrOut != nil {
		if err := attrOut.Close(); err != nil {
			attrOut = nil
			return fmt.Errorf("close PLL attribution dump %q: %w", opts.PLLAttributionDump, err)
		}
		attrOut = nil
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

func effectiveScoreEBMPLLAggregation(cfg *ArchConfig, raw string) (string, error) {
	mode := strings.ToLower(strings.TrimSpace(raw))
	if mode == "" {
		mode = scoreEBMPLLAggregationConfig
	}
	switch mode {
	case scoreEBMPLLAggregationConfig, scoreEBMPLLAggregationDifferingSpan, scoreEBMPLLAggregationDependentWin, scoreEBMPLLAggregationFullSeq:
	default:
		return "", fmt.Errorf("-score-pll-aggregation %q is not supported; supported: %s, %s, %s, %s", raw, scoreEBMPLLAggregationConfig, scoreEBMPLLAggregationDifferingSpan, scoreEBMPLLAggregationDependentWin, scoreEBMPLLAggregationFullSeq)
	}
	scoreMode, err := scoreEBMMode(cfg)
	if err != nil {
		return "", err
	}
	if scoreMode == scoreEBMModeEnergy {
		if mode == scoreEBMPLLAggregationFullSeq || mode == scoreEBMPLLAggregationDependentWin {
			return "", fmt.Errorf("-score-pll-aggregation=%q is only supported for MLM/MNTP scorer configs, not native energy score-ebm configs", mode)
		}
		return scoreEBMPLLAggregationDifferingSpan, nil
	}
	if scoreMode == scoreEBMModeSinglePLL {
		if mode == scoreEBMPLLAggregationConfig {
			return scoreEBMPLLAggregationFullSeq, nil
		}
		return mode, nil
	}
	if mode == scoreEBMPLLAggregationConfig {
		return scoreEBMPLLAggregationDifferingSpan, nil
	}
	return mode, nil
}

func parseScoreEBMSkipTokenIDs(raw string, vocabSize int) (map[int]bool, error) {
	out := make(map[int]bool)
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return out, nil
	}
	if vocabSize <= 0 {
		return nil, fmt.Errorf("invalid vocab_size=%d", vocabSize)
	}
	for _, part := range strings.Split(raw, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			return nil, fmt.Errorf("-score-pll-skip-token-ids contains an empty token id")
		}
		id, err := strconv.Atoi(part)
		if err != nil {
			return nil, fmt.Errorf("-score-pll-skip-token-ids token id %q is invalid: %w", part, err)
		}
		if id < 0 || id >= vocabSize {
			return nil, fmt.Errorf("-score-pll-skip-token-ids token id %d out of range [0,%d)", id, vocabSize)
		}
		out[id] = true
	}
	return out, nil
}

func validateScoreEBMConfig(cfg *ArchConfig) error {
	if cfg == nil {
		return fmt.Errorf("nil config")
	}
	_, err := scoreEBMMode(cfg)
	return err
}

func scoreEBMMode(cfg *ArchConfig) (string, error) {
	if cfg == nil {
		return "", fmt.Errorf("nil config")
	}
	if cfg.Training.MultiheadEnabled() {
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
	switch cfg.Training.EffectiveObjective() {
	case arch.ObjectiveMLM, arch.ObjectiveMNTP:
		if cfg.Training.MLMMaskTokenID < 0 || cfg.Training.MLMMaskTokenID >= cfg.VocabSize {
			return "", fmt.Errorf("score-ebm full-sequence PLL requires training.mlm_mask_token_id in [0,%d)", cfg.VocabSize)
		}
		return scoreEBMModeSinglePLL, nil
	default:
		return "", fmt.Errorf("score-ebm requires training.objective=%q with an energy or mlm_span_pll scorer, or single-objective %q/%q for full-sequence PLL", arch.ObjectiveMultihead, arch.ObjectiveMLM, arch.ObjectiveMNTP)
	}
}

func buildScoreEBMIRProgram(cfg *ArchConfig, scoreMode, pllAggregation string) (*arch.Program, error) {
	if cfg == nil {
		return nil, fmt.Errorf("nil config")
	}
	if pllAggregation != scoreEBMPLLAggregationFullSeq {
		return BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{
			Objective:          arch.ObjectiveMultihead,
			DropoutInactive:    true,
			InvarianceInactive: true,
			PLLMarginInactive:  true,
		})
	}
	objective := arch.ObjectiveMultihead
	if scoreMode == scoreEBMModeSinglePLL {
		objective = cfg.Training.EffectiveObjective()
		if scoreEBMSinglePLLCanUseFullBlockMask(cfg) {
			objective = arch.ObjectiveBlockDiffusion
		}
	}
	state := TrainingProgramState{
		RecurrenceActive:       true,
		HeadUntied:             cfg.MTPUntieEnabled(),
		MTPAuxInactive:         true,
		DistillationInactive:   true,
		Data2VecInactive:       true,
		InvarianceInactive:     true,
		PLLMarginInactive:      true,
		ZLossInactive:          true,
		DropoutInactive:        true,
		Objective:              objective,
		ExampleFramingInactive: true,
	}
	var (
		prog *arch.Program
		err  error
	)
	if len(cfg.RecurrencePhases) > 0 && scoreMode == scoreEBMModeSinglePLL {
		prog, err = arch.BuildTrainingIRProgramForRecurrencePhaseFromConfig(cfg, len(cfg.RecurrencePhases)-1, state)
	} else {
		prog, err = BuildTrainingIRProgramFromConfig(cfg, state)
	}
	if err != nil {
		return nil, err
	}
	if scoreMode == scoreEBMModeSinglePLL {
		suppressDenseEvalLossForScoreEBMFullSeqPLL(prog)
	}
	return prog, nil
}

func scoreEBMSinglePLLCanUseFullBlockMask(cfg *ArchConfig) bool {
	if cfg == nil {
		return false
	}
	hasPlain := false
	for _, block := range cfg.Blocks {
		if !strings.EqualFold(strings.TrimSpace(block.Type), "plain") {
			continue
		}
		hasPlain = true
		mask := strings.ToLower(strings.TrimSpace(block.AttentionMask))
		switch mask {
		case "", arch.AttentionMaskBidirectional, arch.AttentionMaskNone, arch.AttentionMaskBlockDiffusion:
		default:
			return false
		}
	}
	return hasPlain
}

func suppressDenseEvalLossForScoreEBMFullSeqPLL(prog *arch.Program) {
	if prog == nil {
		return
	}
	const unusedEvalLoss = "score_ebm_full_seq_unused_eval_loss"
	for i := range prog.Ops {
		for j, out := range prog.Ops[i].Outputs {
			if out == "eval_loss" {
				prog.Ops[i].Outputs[j] = unusedEvalLoss
			}
		}
	}
	outputs := prog.Outputs[:0]
	for _, out := range prog.Outputs {
		if out.Name != "eval_loss" {
			outputs = append(outputs, out)
		}
	}
	prog.Outputs = outputs
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
	return scoreEBMJSONLWithOptions(r, w, cfg, evaluator, scoreEBMRuntimeOptions{
		scoreBatch:      scoreBatch,
		pllAggregation:  scoreEBMPLLAggregationConfig,
		emitTokenEnergy: emitTokenEnergy,
	})
}

func scoreEBMJSONLWithOptions(r io.Reader, w io.Writer, cfg *ArchConfig, evaluator diffusionGenerationEvaluator, opts scoreEBMRuntimeOptions) error {
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
		out, err := scoreEBMRecordWithOptions(cfg, evaluator, rec, opts)
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

func scoreEBMRecordWithOptions(cfg *ArchConfig, evaluator diffusionGenerationEvaluator, rec scoreEBMInputRecord, opts scoreEBMRuntimeOptions) (scoreEBMOutputRecord, error) {
	if strings.TrimSpace(rec.ID) == "" {
		return scoreEBMOutputRecord{}, fmt.Errorf("id must be non-empty")
	}
	var err error
	rec, err = encodeScoreEBMSequenceFields(rec, opts.sequenceVocab)
	if err != nil {
		return scoreEBMOutputRecord{}, err
	}
	mode, err := scoreEBMMode(cfg)
	if err != nil {
		return scoreEBMOutputRecord{}, err
	}
	pllAggregation, err := effectiveScoreEBMPLLAggregation(cfg, opts.pllAggregation)
	if err != nil {
		return scoreEBMOutputRecord{}, err
	}
	if err := validateScoreEBMPLLWindow(pllAggregation, opts.pllWindow); err != nil {
		return scoreEBMOutputRecord{}, err
	}
	pllFromConfig := scoreEBMPLLAggregationFromConfig(opts.pllAggregation)
	pllUsesPositionScoring := scoreEBMUsesPositionPLL(mode, pllAggregation, pllFromConfig)
	spanMode := scoreEBMUsesDifferingSpan(cfg) && pllAggregation != scoreEBMPLLAggregationFullSeq
	if opts.emitTokenEnergy && mode != scoreEBMModeEnergy {
		return scoreEBMOutputRecord{}, fmt.Errorf("-score-emit-token-energy is only supported for native energy score-ebm configs")
	}
	if opts.pllAttributionEnc != nil && mode == scoreEBMModeEnergy {
		return scoreEBMOutputRecord{}, fmt.Errorf("-score-pll-attribution-dump is only supported for MLM/MNTP PLL scorer configs")
	}
	if opts.emitTokenEnergy && !spanMode {
		return scoreEBMOutputRecord{}, fmt.Errorf("-score-emit-token-energy requires training.minimal_pair.energy_aggregation=%q", arch.MinimalPairEnergySpan)
	}
	hasTokens := len(rec.Tokens) > 0
	hasPair := len(rec.Clean) > 0 || len(rec.Corrupt) > 0
	if hasTokens == hasPair {
		return scoreEBMOutputRecord{}, fmt.Errorf("provide exactly one of tokens or clean/corrupt")
	}
	if hasPair && (len(rec.Clean) == 0 || len(rec.Corrupt) == 0) {
		return scoreEBMOutputRecord{}, fmt.Errorf("pair records require both clean and corrupt tokens")
	}
	if pllUsesPositionScoring {
		return scoreEBMPLLRecordWithOptions(cfg, evaluator, rec, opts, pllAggregation)
	}
	if hasTokens {
		if !spanMode && pllAggregation != scoreEBMPLLAggregationFullSeq && len(rec.Span) > 0 {
			return scoreEBMOutputRecord{}, fmt.Errorf("span requires training.minimal_pair.energy_aggregation=%q", arch.MinimalPairEnergySpan)
		}
		spans, err := scoreEBMSequenceSpans(rec.Tokens, rec.Span, spanMode)
		if err != nil {
			return scoreEBMOutputRecord{}, err
		}
		energy, tokenEnergy, err := scoreEBMSequencesDetailedWithOptions(cfg, evaluator, [][]int{rec.Tokens}, spans, opts)
		if err != nil {
			return scoreEBMOutputRecord{}, err
		}
		v := float64(energy[0])
		out := scoreEBMOutputRecord{ID: rec.ID, NTokens: len(rec.Tokens), Sequence: rec.Sequence}
		if mode == scoreEBMModeEnergy {
			out.Energy = &v
		} else {
			out.Score = &v
		}
		if opts.emitTokenEnergy {
			out.TokenEnergy = float32SliceToFloat64(tokenEnergy[0][:len(rec.Tokens)])
		}
		return out, nil
	}
	if !spanMode && pllAggregation != scoreEBMPLLAggregationFullSeq && (len(rec.CleanSpan) > 0 || len(rec.CorruptSpan) > 0) {
		return scoreEBMOutputRecord{}, fmt.Errorf("clean_span/corrupt_span require training.minimal_pair.energy_aggregation=%q", arch.MinimalPairEnergySpan)
	}
	spans, err := scoreEBMPairSpans(rec, spanMode)
	if err != nil {
		return scoreEBMOutputRecord{}, err
	}
	energies, tokenEnergy, err := scoreEBMSequencesDetailedWithOptions(cfg, evaluator, [][]int{rec.Clean, rec.Corrupt}, spans, opts)
	if err != nil {
		return scoreEBMOutputRecord{}, err
	}
	clean := float64(energies[0])
	corrupt := float64(energies[1])
	out := scoreEBMOutputRecord{
		ID: rec.ID, Family: rec.Family,
		CleanSequence: rec.CleanSequence, CorruptSequence: rec.CorruptSequence,
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
	if opts.emitTokenEnergy {
		out.CleanTokenEnergy = float32SliceToFloat64(tokenEnergy[0][:len(rec.Clean)])
		out.CorruptTokenEnergy = float32SliceToFloat64(tokenEnergy[1][:len(rec.Corrupt)])
	}
	return out, nil
}

func scoreEBMSequencesDetailedWithOptions(cfg *ArchConfig, evaluator diffusionGenerationEvaluator, sequences [][]int, spans [][]int, opts scoreEBMRuntimeOptions) ([]float32, [][]float32, error) {
	if err := validateScoreEBMConfig(cfg); err != nil {
		return nil, nil, err
	}
	if len(sequences) == 0 {
		return nil, nil, fmt.Errorf("no sequences to score")
	}
	mode, err := scoreEBMMode(cfg)
	if err != nil {
		return nil, nil, err
	}
	pllAggregation, err := effectiveScoreEBMPLLAggregation(cfg, opts.pllAggregation)
	if err != nil {
		return nil, nil, err
	}
	if err := validateScoreEBMPLLWindow(pllAggregation, opts.pllWindow); err != nil {
		return nil, nil, err
	}
	if opts.emitTokenEnergy && mode != scoreEBMModeEnergy {
		return nil, nil, fmt.Errorf("token energy output is only supported for native energy score-ebm configs")
	}
	spanMode := scoreEBMUsesDifferingSpan(cfg) && pllAggregation != scoreEBMPLLAggregationFullSeq
	if opts.emitTokenEnergy && !spanMode {
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
	if scoreEBMUsesPositionPLL(mode, pllAggregation, scoreEBMPLLAggregationFromConfig(opts.pllAggregation)) {
		if opts.emitTokenEnergy {
			return nil, nil, fmt.Errorf("token energy output is not supported for PLL scoring")
		}
		positionBatch := opts.scorePositionBatch
		if positionBatch <= 0 {
			var err error
			positionBatch, err = effectiveScorePositionBatch(cfg.SeqLen, cfg.VocabSize, positionBatch)
			if err != nil {
				return nil, nil, err
			}
		}
		skipIDs := scoreEBMEffectivePLLSkipTokenIDs(cfg, opts.pllSkipTokenIDs)
		out := make([]float32, len(sequences))
		for i, seq := range sequences {
			var span []int
			if pllAggregation != scoreEBMPLLAggregationFullSeq {
				if len(spans) <= i {
					return nil, nil, fmt.Errorf("sequence %d missing PLL span", i)
				}
				span = spans[i]
			}
			score, err := scoreEBMPLLScoreOne(cfg, evaluator, seq, span, pllAggregation, opts.pllWindow, positionBatch, skipIDs, false)
			if err != nil {
				return nil, nil, fmt.Errorf("sequence %d: %w", i, err)
			}
			out[i] = float32(score)
		}
		return out, nil, nil
	}
	scoreBatch := opts.scoreBatch
	if len(sequences) > scoreBatch {
		return nil, nil, fmt.Errorf("sequence count %d exceeds score batch %d", len(sequences), scoreBatch)
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
	if opts.emitTokenEnergy && mode == scoreEBMModeEnergy {
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
	if opts.emitTokenEnergy && mode == scoreEBMModeEnergy {
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
		batchSizeOverride:   totalRows,
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
