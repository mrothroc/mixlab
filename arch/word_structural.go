package arch

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"strings"
)

// WordStructuralObjectiveSpec configures StructBERT-style local word-order
// reconstruction as a training-only auxiliary on masked vocab-logit heads.
type WordStructuralObjectiveSpec struct {
	Enabled      bool     `json:"enabled,omitempty"`
	Fraction     float64  `json:"fraction,omitempty"`
	Span         int      `json:"span,omitempty"`
	LossWeight   float64  `json:"loss_weight,omitempty"`
	SkipTokenIDs []int    `json:"skip_token_ids,omitempty"`
	Heads        []string `json:"heads,omitempty"`

	enabledSet    bool
	fractionSet   bool
	spanSet       bool
	lossWeightSet bool
	skipIDsSet    bool
	headsSet      bool
}

func (w *WordStructuralObjectiveSpec) UnmarshalJSON(data []byte) error {
	type Alias WordStructuralObjectiveSpec
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	var alias Alias
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&alias); err != nil {
		return err
	}
	*w = WordStructuralObjectiveSpec(alias)
	_, w.enabledSet = raw["enabled"]
	_, w.fractionSet = raw["fraction"]
	_, w.spanSet = raw["span"]
	_, w.lossWeightSet = raw["loss_weight"]
	_, w.skipIDsSet = raw["skip_token_ids"]
	_, w.headsSet = raw["heads"]
	return nil
}

func (w *WordStructuralObjectiveSpec) applyDefaults(maskTokenID int) {
	if w == nil {
		return
	}
	if !w.enabledSet && !w.Enabled {
		w.Enabled = true
	}
	if !w.Enabled {
		return
	}
	if !w.fractionSet && w.Fraction == 0 {
		w.Fraction = 0.05
	}
	if !w.spanSet && w.Span == 0 {
		w.Span = 3
	}
	if !w.lossWeightSet && w.LossWeight == 0 {
		w.LossWeight = 1
	}
	if !w.skipIDsSet && len(w.SkipTokenIDs) == 0 {
		w.SkipTokenIDs = []int{maskTokenID}
	}
	for i := range w.Heads {
		w.Heads[i] = strings.TrimSpace(w.Heads[i])
	}
}

func (t TrainingSpec) WordStructuralActive() bool {
	return t.WordStructuralObjective != nil && t.WordStructuralObjective.Enabled
}

func (t TrainingSpec) WordStructuralActiveForConcreteObjective(objective string) bool {
	if !t.WordStructuralActive() {
		return false
	}
	switch normalizeTrainingObjective(objective) {
	case ObjectiveMLM, ObjectiveMNTP:
		return true
	case ObjectiveHybridExample:
		secondary := t.EffectiveHybridSecondaryObjective()
		return secondary == ObjectiveMLM || secondary == ObjectiveMNTP
	default:
		return false
	}
}

func (t TrainingSpec) WordStructuralSelectedHeads() []string {
	if !t.WordStructuralActive() || t.WordStructuralObjective == nil {
		return nil
	}
	return append([]string(nil), t.WordStructuralObjective.Heads...)
}

func (t TrainingSpec) WordStructuralHeadSelected(name string) bool {
	name = strings.TrimSpace(name)
	if name == "" || !t.WordStructuralActive() || t.WordStructuralObjective == nil {
		return false
	}
	for _, head := range t.WordStructuralObjective.Heads {
		if head == name {
			return true
		}
	}
	return false
}

func validateWordStructuralObjective(cfg *ArchConfig, source string) error {
	if cfg == nil || cfg.Training.WordStructuralObjective == nil || !cfg.Training.WordStructuralObjective.Enabled {
		return nil
	}
	t := &cfg.Training
	w := t.WordStructuralObjective
	if w.Fraction <= 0 || w.Fraction > 0.5 || math.IsNaN(w.Fraction) || math.IsInf(w.Fraction, 0) {
		return fmt.Errorf("config %q training.word_structural_objective.fraction=%g must be in (0,0.5]", source, w.Fraction)
	}
	if w.Span < 2 {
		return fmt.Errorf("config %q training.word_structural_objective.span=%d must be >= 2", source, w.Span)
	}
	if w.Span > cfg.SeqLen {
		return fmt.Errorf("config %q training.word_structural_objective.span=%d must be <= seq_len=%d", source, w.Span, cfg.SeqLen)
	}
	if w.LossWeight < 0 || math.IsNaN(w.LossWeight) || math.IsInf(w.LossWeight, 0) {
		return fmt.Errorf("config %q training.word_structural_objective.loss_weight=%g must be finite and >= 0", source, w.LossWeight)
	}
	if len(w.SkipTokenIDs) == 0 {
		return fmt.Errorf("config %q training.word_structural_objective.skip_token_ids must contain at least one token id", source)
	}
	seenSkip := make(map[int]struct{}, len(w.SkipTokenIDs))
	for i, id := range w.SkipTokenIDs {
		if id < 0 || id >= cfg.VocabSize {
			return fmt.Errorf("config %q training.word_structural_objective.skip_token_ids[%d]=%d must be in [0,%d)", source, i, id, cfg.VocabSize)
		}
		if _, ok := seenSkip[id]; ok {
			return fmt.Errorf("config %q training.word_structural_objective.skip_token_ids contains duplicate id %d", source, id)
		}
		seenSkip[id] = struct{}{}
	}

	switch t.EffectiveObjective() {
	case ObjectiveMLM, ObjectiveMNTP:
		return nil
	case ObjectiveHybrid:
		secondary := t.EffectiveHybridSecondaryObjective()
		if secondary != ObjectiveMLM && secondary != ObjectiveMNTP {
			return fmt.Errorf("config %q training.word_structural_objective requires hybrid_secondary_objective to be \"mlm\" or \"mntp\"", source)
		}
		if !t.HybridHasMaskedSteps() {
			return fmt.Errorf("config %q training.word_structural_objective requires hybrid masked secondary steps; hybrid_clm_fraction=1 disables them", source)
		}
		return nil
	case ObjectiveMultihead:
		return validateWordStructuralMultihead(cfg, source)
	default:
		return fmt.Errorf("config %q training.word_structural_objective requires training.objective \"mlm\", \"mntp\", \"hybrid\", or \"multihead\"", source)
	}
}

func validateWordStructuralMultihead(cfg *ArchConfig, source string) error {
	t := &cfg.Training
	w := t.WordStructuralObjective
	if w == nil || !w.Enabled {
		return nil
	}
	if len(w.Heads) == 0 {
		head := strings.TrimSpace(t.ExportHead)
		if head == "" {
			head = defaultMultiheadExportHeadName(t.Heads)
		}
		w.Heads = []string{head}
	}
	seenNames := make(map[string]int, len(t.Heads))
	for i, h := range t.Heads {
		seenNames[h.Name] = i
	}
	seenSelected := make(map[string]struct{}, len(w.Heads))
	for i, name := range w.Heads {
		name = strings.TrimSpace(name)
		if name == "" {
			return fmt.Errorf("config %q training.word_structural_objective.heads[%d] is empty", source, i)
		}
		if _, ok := seenSelected[name]; ok {
			return fmt.Errorf("config %q training.word_structural_objective.heads duplicates %q", source, name)
		}
		seenSelected[name] = struct{}{}
		idx, ok := seenNames[name]
		if !ok {
			return fmt.Errorf("config %q training.word_structural_objective.heads[%d]=%q does not match any training.heads[].name", source, i, name)
		}
		h := t.Heads[idx]
		if h.Objective != ObjectiveMLM && h.Objective != ObjectiveMNTP {
			return fmt.Errorf("config %q training.word_structural_objective.heads[%d]=%q must select an mlm or mntp head", source, i, name)
		}
		if h.OutputHead != MultiheadOutputBERTMLM && h.OutputHead != MultiheadOutputLinear {
			return fmt.Errorf("config %q training.word_structural_objective.heads[%d]=%q must emit vocab logits", source, i, name)
		}
		w.Heads[i] = name
	}
	return nil
}
