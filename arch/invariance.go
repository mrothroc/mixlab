package arch

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"strings"
)

const (
	InvarianceSourceFile   = "file"
	InvarianceSourceJSONL  = "jsonl"
	InvarianceSourceBinary = "bin"
	InvarianceLossSymKL    = "sym_kl"
	InvarianceTargetMasked = "masked_position"
)

// InvarianceSpec configures structured two-view consistency training. The
// examples are supplied by the caller; Mixlab only validates, samples, and
// compares the explicitly annotated masked-token distributions.
type InvarianceSpec struct {
	Source        string  `json:"source,omitempty"`
	Path          string  `json:"path,omitempty"`
	Loss          string  `json:"loss,omitempty"`
	Weight        float64 `json:"weight,omitempty"`
	BatchFraction float64 `json:"batch_fraction,omitempty"`
	Target        string  `json:"target,omitempty"`
	SkipTokenIDs  []int   `json:"skip_token_ids,omitempty"`

	weightSet        bool
	batchFractionSet bool
}

func (s *InvarianceSpec) UnmarshalJSON(data []byte) error {
	type alias InvarianceSpec
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
	*s = InvarianceSpec(raw)
	_, s.weightSet = fields["weight"]
	_, s.batchFractionSet = fields["batch_fraction"]
	return nil
}

func (s *InvarianceSpec) applyDefaults() {
	if s == nil {
		return
	}
	s.Source = strings.ToLower(strings.TrimSpace(s.Source))
	if s.Source == "" {
		s.Source = InvarianceSourceFile
	}
	s.Loss = strings.ToLower(strings.TrimSpace(s.Loss))
	if s.Loss == "" {
		s.Loss = InvarianceLossSymKL
	}
	s.Target = strings.ToLower(strings.TrimSpace(s.Target))
	if s.Target == "" {
		s.Target = InvarianceTargetMasked
	}
	if !s.weightSet && s.Weight == 0 {
		s.Weight = 1
	}
	if !s.batchFractionSet && s.BatchFraction == 0 {
		s.BatchFraction = 0.25
	}
}

// Active reports whether the objective changes the training graph or data
// path. A zero weight is intentionally a complete no-op, not an ablation that
// still samples or forwards invariance pairs.
func (s *InvarianceSpec) Active() bool {
	return s != nil && s.Weight > 0
}

func (t TrainingSpec) InvarianceActive() bool {
	return t.Invariance != nil && t.Invariance.Active()
}

// InvarianceSkipsTokenID reports whether an annotated target is excluded from
// invariance pairs. The configured MLM mask token is always excluded, matching
// score-time PLL's automatic mask-token skipping.
func (s InvarianceSpec) InvarianceSkipsTokenID(tokenID, maskTokenID int) bool {
	if tokenID == maskTokenID {
		return true
	}
	for _, id := range s.SkipTokenIDs {
		if tokenID == id {
			return true
		}
	}
	return false
}

func validateTrainingInvariance(cfg *ArchConfig, source string) error {
	if cfg == nil || cfg.Training.Invariance == nil {
		return nil
	}
	s := cfg.Training.Invariance
	s.applyDefaults()
	switch s.Source {
	case InvarianceSourceFile, InvarianceSourceJSONL, InvarianceSourceBinary:
	default:
		return fmt.Errorf("config %q training.invariance.source=%q must be \"file\", \"jsonl\", or \"bin\"", source, s.Source)
	}
	if s.Loss != InvarianceLossSymKL {
		return fmt.Errorf("config %q training.invariance.loss=%q must be \"sym_kl\"", source, s.Loss)
	}
	if s.Target != InvarianceTargetMasked {
		return fmt.Errorf("config %q training.invariance.target=%q must be \"masked_position\"", source, s.Target)
	}
	if s.Weight < 0 || math.IsNaN(s.Weight) || math.IsInf(s.Weight, 0) {
		return fmt.Errorf("config %q training.invariance.weight=%g must be finite and >= 0", source, s.Weight)
	}
	if s.BatchFraction <= 0 || s.BatchFraction > 1 || math.IsNaN(s.BatchFraction) || math.IsInf(s.BatchFraction, 0) {
		return fmt.Errorf("config %q training.invariance.batch_fraction=%g must be finite and in (0,1]", source, s.BatchFraction)
	}
	for _, id := range s.SkipTokenIDs {
		if id < 0 || id >= cfg.VocabSize {
			return fmt.Errorf("config %q training.invariance.skip_token_ids contains %d outside [0,%d)", source, id, cfg.VocabSize)
		}
	}
	if !s.Active() {
		return nil
	}
	if strings.TrimSpace(s.Path) == "" {
		return fmt.Errorf("config %q training.invariance.path is required when training.invariance.weight > 0", source)
	}
	if !cfg.Training.mlmMaskTokenIDSet || cfg.Training.MLMMaskTokenID < 0 || cfg.Training.MLMMaskTokenID >= cfg.VocabSize {
		return fmt.Errorf("config %q training.invariance requires training.mlm_mask_token_id in [0,%d)", source, cfg.VocabSize)
	}
	if cfg.Training.BatchTokens <= 0 || cfg.SeqLen <= 0 || cfg.Training.BatchTokens%cfg.SeqLen != 0 || cfg.Training.BatchTokens/cfg.SeqLen < 2 || (cfg.Training.BatchTokens/cfg.SeqLen)%2 != 0 {
		return fmt.Errorf("config %q training.invariance requires an even number of sequence rows per batch, at least two", source)
	}
	if cfg.Training.DistillationKLEffectiveActive() || cfg.Training.Data2VecActive() || cfg.MTP != nil || cfg.Training.FirstByteMask || cfg.Training.ExampleFramingEnabled() || cfg.Training.AttentionSegmentMaskEnabled() || len(cfg.Training.SeqLenSchedule) > 0 {
		return fmt.Errorf("config %q training.invariance cannot be combined with distillation, data2vec, mtp, first_byte_mask, example_framing, attention_segment_mask, or seq_len_schedule in v1", source)
	}
	if cfg.Training.MultiheadEnabled() {
		head := cfg.Training.MultiheadExportHead()
		if head == nil || (head.Objective != ObjectiveMLM && head.Objective != ObjectiveMNTP) || (head.OutputHead != MultiheadOutputBERTMLM && head.OutputHead != MultiheadOutputLinear) {
			return fmt.Errorf("config %q training.invariance requires the multihead export_head to be an mlm or mntp vocab-logit head", source)
		}
		return nil
	}
	switch cfg.Training.EffectiveObjective() {
	case ObjectiveMLM, ObjectiveMNTP:
		return nil
	default:
		return fmt.Errorf("config %q training.invariance only supports training.objective=\"mlm\", \"mntp\", or a multihead masked export head in v1", source)
	}
}
