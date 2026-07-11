package arch

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"strings"
)

const (
	PLLMarginSourceFile   = "file"
	PLLMarginSourceJSONL  = "jsonl"
	PLLMarginSourceBinary = "bin"

	// PLLMarginTargetAnnotatedSpan is an explicitly annotated token span that
	// is unchanged across a preferred and contrast view. "distractor_span" is
	// retained as an input alias for existing experiment descriptions.
	PLLMarginTargetAnnotatedSpan = "annotated_span"
	PLLMarginTargetDistractor    = "distractor_span"
)

// PLLMarginSpec configures a training-only paired masked-span PLL margin.
// The pair artifact supplies preferred/contrast views and the unchanged target
// span; Mixlab does not infer a linguistic relation from raw corpus text.
type PLLMarginSpec struct {
	Source        string  `json:"source,omitempty"`
	Path          string  `json:"path,omitempty"`
	Margin        float64 `json:"margin,omitempty"`
	Weight        float64 `json:"weight,omitempty"`
	AnchorWeight  float64 `json:"anchor_weight,omitempty"`
	BatchFraction float64 `json:"batch_fraction,omitempty"`
	Target        string  `json:"target,omitempty"`
	SkipTokenIDs  []int   `json:"skip_token_ids,omitempty"`

	marginSet        bool
	weightSet        bool
	anchorWeightSet  bool
	batchFractionSet bool
}

func (s *PLLMarginSpec) UnmarshalJSON(data []byte) error {
	type alias PLLMarginSpec
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return err
	}
	var raw alias
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&raw); err != nil {
		return err
	}
	*s = PLLMarginSpec(raw)
	_, s.marginSet = fields["margin"]
	_, s.weightSet = fields["weight"]
	_, s.anchorWeightSet = fields["anchor_weight"]
	_, s.batchFractionSet = fields["batch_fraction"]
	return nil
}

func (s *PLLMarginSpec) applyDefaults() {
	if s == nil {
		return
	}
	s.Source = strings.ToLower(strings.TrimSpace(s.Source))
	if s.Source == "" {
		s.Source = PLLMarginSourceFile
	}
	s.Target = strings.ToLower(strings.TrimSpace(s.Target))
	if s.Target == "" || s.Target == PLLMarginTargetDistractor {
		s.Target = PLLMarginTargetAnnotatedSpan
	}
	if !s.marginSet && s.Margin == 0 {
		s.Margin = 1
	}
	if !s.weightSet && s.Weight == 0 {
		// The paired anchor is deliberately sharp early in masked-LM training.
		// Keep the default additive contribution conservative; callers that have
		// validated a larger coefficient can still set it explicitly.
		s.Weight = 0.1
	}
	if !s.anchorWeightSet && s.AnchorWeight == 0 {
		s.AnchorWeight = 0.5
	}
	if !s.batchFractionSet && s.BatchFraction == 0 {
		s.BatchFraction = 0.25
	}
}

// Active reports whether the paired loss changes the program or batch path.
// A zero weight is intentionally a complete no-op, including pair loading and
// deterministic sampling.
func (s *PLLMarginSpec) Active() bool {
	return s != nil && s.Weight > 0
}

func (t TrainingSpec) PLLMarginActive() bool {
	return t.PLLMargin != nil && t.PLLMargin.Active()
}

func validateTrainingPLLMargin(cfg *ArchConfig, source string) error {
	if cfg == nil || cfg.Training.PLLMargin == nil {
		return nil
	}
	s := cfg.Training.PLLMargin
	s.applyDefaults()
	switch s.Source {
	case PLLMarginSourceFile, PLLMarginSourceJSONL, PLLMarginSourceBinary:
	default:
		return fmt.Errorf("config %q training.pll_margin.source=%q must be \"file\", \"jsonl\", or \"bin\"", source, s.Source)
	}
	if s.Target != PLLMarginTargetAnnotatedSpan {
		return fmt.Errorf("config %q training.pll_margin.target=%q must be \"annotated_span\" (\"distractor_span\" is accepted as an alias)", source, s.Target)
	}
	if s.Margin < 0 || math.IsNaN(s.Margin) || math.IsInf(s.Margin, 0) {
		return fmt.Errorf("config %q training.pll_margin.margin=%g must be finite and >= 0", source, s.Margin)
	}
	if s.Weight < 0 || math.IsNaN(s.Weight) || math.IsInf(s.Weight, 0) {
		return fmt.Errorf("config %q training.pll_margin.weight=%g must be finite and >= 0", source, s.Weight)
	}
	if s.AnchorWeight < 0 || math.IsNaN(s.AnchorWeight) || math.IsInf(s.AnchorWeight, 0) {
		return fmt.Errorf("config %q training.pll_margin.anchor_weight=%g must be finite and >= 0", source, s.AnchorWeight)
	}
	if s.BatchFraction <= 0 || s.BatchFraction > 1 || math.IsNaN(s.BatchFraction) || math.IsInf(s.BatchFraction, 0) {
		return fmt.Errorf("config %q training.pll_margin.batch_fraction=%g must be finite and in (0,1]", source, s.BatchFraction)
	}
	for _, id := range s.SkipTokenIDs {
		if id < 0 || id >= cfg.VocabSize {
			return fmt.Errorf("config %q training.pll_margin.skip_token_ids contains %d outside [0,%d)", source, id, cfg.VocabSize)
		}
	}
	if !s.Active() {
		return nil
	}
	if strings.TrimSpace(s.Path) == "" {
		return fmt.Errorf("config %q training.pll_margin.path is required when training.pll_margin.weight > 0", source)
	}
	if !cfg.Training.mlmMaskTokenIDSet || cfg.Training.MLMMaskTokenID < 0 || cfg.Training.MLMMaskTokenID >= cfg.VocabSize {
		return fmt.Errorf("config %q training.pll_margin requires training.mlm_mask_token_id in [0,%d)", source, cfg.VocabSize)
	}
	if cfg.Training.BatchTokens <= 0 || cfg.SeqLen <= 0 || cfg.Training.BatchTokens%cfg.SeqLen != 0 || cfg.Training.BatchTokens/cfg.SeqLen < 2 || (cfg.Training.BatchTokens/cfg.SeqLen)%2 != 0 {
		return fmt.Errorf("config %q training.pll_margin requires an even number of sequence rows per batch, at least two", source)
	}
	if cfg.Training.InvarianceActive() || cfg.Training.MinimalPair != nil || cfg.Training.WordStructuralActive() || cfg.Training.RTD != nil || cfg.Training.DistillationKLEffectiveActive() || cfg.Training.Data2VecActive() || cfg.MTP != nil || cfg.Training.FirstByteMask || cfg.Training.ExampleFramingEnabled() || cfg.Training.AttentionSegmentMaskEnabled() || len(cfg.Training.SeqLenSchedule) > 0 {
		return fmt.Errorf("config %q training.pll_margin cannot be combined with invariance, minimal_pair, word_structural_objective, rtd, distillation, data2vec, mtp, first_byte_mask, example_framing, attention_segment_mask, or seq_len_schedule in v1", source)
	}
	if cfg.Training.MultiheadEnabled() {
		head := cfg.Training.MultiheadExportHead()
		if head == nil || (head.Objective != ObjectiveMLM && head.Objective != ObjectiveMNTP) || (head.OutputHead != MultiheadOutputBERTMLM && head.OutputHead != MultiheadOutputLinear) {
			return fmt.Errorf("config %q training.pll_margin requires the multihead export_head to be an mlm or mntp vocab-logit head", source)
		}
		return nil
	}
	switch cfg.Training.EffectiveObjective() {
	case ObjectiveMLM, ObjectiveMNTP:
		return nil
	default:
		return fmt.Errorf("config %q training.pll_margin only supports training.objective=\"mlm\", \"mntp\", or a multihead masked export head in v1", source)
	}
}
