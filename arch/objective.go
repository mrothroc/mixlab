package arch

import (
	"fmt"
	"math"
	"strings"
)

const (
	ObjectiveCausal = "causal"
	ObjectiveMLM    = "mlm"
	ObjectiveHybrid = "hybrid"
	ObjectiveMNTP   = "mntp"
	// ObjectiveHybridExample is an internal concrete training objective used
	// for per-example hybrid batches. Public configs still use objective=hybrid.
	ObjectiveHybridExample = "hybrid_example"

	AttentionMaskCausal        = "causal"
	AttentionMaskBidirectional = "bidirectional"
	AttentionMaskNone          = "none"
	AttentionMaskHybridExample = "hybrid_example"

	HybridMixGranularityBatch   = "batch"
	HybridMixGranularityExample = "example"

	AttentionSegmentMaskNone          = "none"
	AttentionSegmentMaskBoundaryToken = "boundary_token"
)

func normalizeTrainingObjective(raw string) string {
	obj := strings.ToLower(strings.TrimSpace(raw))
	if obj == "" {
		return ObjectiveCausal
	}
	return obj
}

func normalizeAttentionMask(raw string) string {
	return strings.ToLower(strings.TrimSpace(raw))
}

func isMaskedTrainingObjective(objective string) bool {
	switch normalizeTrainingObjective(objective) {
	case ObjectiveMLM, ObjectiveMNTP, ObjectiveHybridExample:
		return true
	default:
		return false
	}
}

func IsMaskedTrainingObjectiveForData2Vec(objective string) bool {
	return isMaskedTrainingObjective(objective)
}

func (t TrainingSpec) EffectiveObjective() string {
	return normalizeTrainingObjective(t.Objective)
}

func (t TrainingSpec) EffectiveHybridMixGranularity() string {
	mode := strings.ToLower(strings.TrimSpace(t.HybridMixGranularity))
	if mode == "" {
		return HybridMixGranularityBatch
	}
	return mode
}

func (t TrainingSpec) EffectiveHybridSecondaryObjective() string {
	obj := normalizeTrainingObjective(t.HybridSecondaryObjective)
	if obj == ObjectiveCausal || obj == ObjectiveHybrid {
		return ObjectiveMNTP
	}
	return obj
}

func (t TrainingSpec) EffectiveAttentionSegmentMask() string {
	mode := strings.ToLower(strings.TrimSpace(t.AttentionSegmentMask))
	if mode == "" {
		return ""
	}
	return mode
}

func (t TrainingSpec) AttentionSegmentMaskEnabled() bool {
	return t.EffectiveAttentionSegmentMask() == AttentionSegmentMaskBoundaryToken
}

func (t TrainingSpec) DefaultConcreteObjective() string {
	obj := t.EffectiveObjective()
	if obj == ObjectiveHybrid {
		return t.EffectiveHybridSecondaryObjective()
	}
	return obj
}

func (t TrainingSpec) NeedsMaskedLoss() bool {
	return isMaskedTrainingObjective(t.DefaultConcreteObjective())
}

func validateTrainingObjective(cfg *ArchConfig, source string) error {
	t := &cfg.Training
	t.Objective = normalizeTrainingObjective(t.Objective)
	switch t.Objective {
	case ObjectiveCausal, ObjectiveMLM, ObjectiveHybrid, ObjectiveMNTP:
	default:
		return fmt.Errorf("config %q has invalid training.objective=%q (must be \"causal\", \"mlm\", \"hybrid\", or \"mntp\")", source, t.Objective)
	}
	t.HybridMixGranularity = t.EffectiveHybridMixGranularity()
	switch t.HybridMixGranularity {
	case HybridMixGranularityBatch, HybridMixGranularityExample:
	default:
		return fmt.Errorf("config %q has invalid training.hybrid_mix_granularity=%q (must be \"batch\" or \"example\")", source, t.HybridMixGranularity)
	}
	if t.Objective != ObjectiveHybrid && t.HybridMixGranularity == HybridMixGranularityExample {
		return fmt.Errorf("config %q has training.hybrid_mix_granularity=\"example\" but training.objective is %q (must be \"hybrid\")", source, t.Objective)
	}
	if !t.mlmMaskProbSet && t.MLMMaskProb == 0 {
		t.MLMMaskProb = 0.15
	}
	if !t.mlmReplacementProbSet && t.MLMMaskTokenProb == 0 && t.MLMRandomTokenProb == 0 && t.MLMKeptUnchangedProb == 0 {
		t.MLMMaskTokenProb = 0.8
		t.MLMRandomTokenProb = 0.1
		t.MLMKeptUnchangedProb = 0.1
	}
	if !t.hybridCLMFractionSet && t.HybridCLMFraction == 0 {
		t.HybridCLMFraction = 0.5
	}
	if strings.TrimSpace(t.HybridSecondaryObjective) == "" {
		t.HybridSecondaryObjective = ObjectiveMNTP
	} else {
		t.HybridSecondaryObjective = normalizeTrainingObjective(t.HybridSecondaryObjective)
	}

	if t.MLMMaskProb < 0 || t.MLMMaskProb > 1 || math.IsNaN(t.MLMMaskProb) {
		return fmt.Errorf("config %q has invalid training.mlm_mask_prob=%g (must be in [0,1])", source, t.MLMMaskProb)
	}
	for name, value := range map[string]float64{
		"mlm_mask_token_prob":     t.MLMMaskTokenProb,
		"mlm_random_token_prob":   t.MLMRandomTokenProb,
		"mlm_kept_unchanged_prob": t.MLMKeptUnchangedProb,
	} {
		if value < 0 || value > 1 || math.IsNaN(value) {
			return fmt.Errorf("config %q has invalid training.%s=%g (must be in [0,1])", source, name, value)
		}
	}
	probSum := t.MLMMaskTokenProb + t.MLMRandomTokenProb + t.MLMKeptUnchangedProb
	if math.Abs(probSum-1.0) > 1e-6 {
		return fmt.Errorf("config %q has invalid MLM replacement probabilities: mlm_mask_token_prob + mlm_random_token_prob + mlm_kept_unchanged_prob = %g (must sum to 1.0)", source, probSum)
	}
	if t.HybridCLMFraction < 0 || t.HybridCLMFraction > 1 || math.IsNaN(t.HybridCLMFraction) {
		return fmt.Errorf("config %q has invalid training.hybrid_clm_fraction=%g (must be in [0,1])", source, t.HybridCLMFraction)
	}
	switch t.HybridSecondaryObjective {
	case ObjectiveMLM, ObjectiveMNTP:
	default:
		return fmt.Errorf("config %q has invalid training.hybrid_secondary_objective=%q (must be \"mlm\" or \"mntp\")", source, t.HybridSecondaryObjective)
	}
	if (t.Objective == ObjectiveMLM || t.Objective == ObjectiveMNTP || t.Objective == ObjectiveHybrid) && !t.mlmMaskTokenIDSet {
		return fmt.Errorf("config %q training.mlm_mask_token_id is required when training.objective is %q", source, t.Objective)
	}
	if t.mlmMaskTokenIDSet && (t.MLMMaskTokenID < 0 || t.MLMMaskTokenID >= cfg.VocabSize) {
		return fmt.Errorf("config %q has invalid training.mlm_mask_token_id=%d (must be in [0,%d))", source, t.MLMMaskTokenID, cfg.VocabSize)
	}
	if t.Objective != ObjectiveCausal {
		if cfg.MTP != nil {
			return fmt.Errorf("config %q training.objective=%q cannot be combined with top-level mtp in v1", source, t.Objective)
		}
		if t.FirstByteMask {
			return fmt.Errorf("config %q training.objective=%q cannot be combined with training.first_byte_mask in v1", source, t.Objective)
		}
	}
	if t.Objective == ObjectiveMNTP && cfg.SeqLen <= 1 {
		return fmt.Errorf("config %q training.objective=\"mntp\" requires seq_len > 1", source)
	}
	if t.Objective == ObjectiveHybrid && t.HybridSecondaryObjective == ObjectiveMNTP && cfg.SeqLen <= 1 {
		return fmt.Errorf("config %q training.hybrid_secondary_objective=\"mntp\" requires seq_len > 1", source)
	}
	return nil
}

func validateTrainingAttentionSegmentMask(cfg *ArchConfig, source string) error {
	t := &cfg.Training
	t.AttentionSegmentMask = t.EffectiveAttentionSegmentMask()
	switch t.AttentionSegmentMask {
	case "", AttentionSegmentMaskNone:
		return nil
	case AttentionSegmentMaskBoundaryToken:
	default:
		return fmt.Errorf("config %q has invalid training.attention_segment_mask=%q (must be \"none\" or \"boundary_token\")", source, t.AttentionSegmentMask)
	}
	if !t.attentionSegmentBoundaryTokenIDSet && t.AttentionSegmentBoundaryTokenID == 0 {
		return fmt.Errorf("config %q training.attention_segment_boundary_token_id is required when training.attention_segment_mask=\"boundary_token\"", source)
	}
	if t.AttentionSegmentBoundaryTokenID < 0 || t.AttentionSegmentBoundaryTokenID >= cfg.VocabSize {
		return fmt.Errorf("config %q has invalid training.attention_segment_boundary_token_id=%d (must be in [0,%d))", source, t.AttentionSegmentBoundaryTokenID, cfg.VocabSize)
	}
	if cfg.Training.Distillation != nil {
		return fmt.Errorf("config %q training.attention_segment_mask cannot be combined with training.distillation in v1; teacher programs do not consume segment_ids", source)
	}
	hasPlain := false
	for i, block := range cfg.Blocks {
		switch blockTypeKey(block) {
		case "plain":
			hasPlain = true
		case "swiglu", "geglu", "mlp", "moe":
		default:
			return fmt.Errorf("config %q blocks[%d].type=%q cannot be combined with training.attention_segment_mask in v1; segment masking only applies to plain self-attention plus position-wise FFN/MoE blocks", source, i, block.Type)
		}
	}
	if !hasPlain {
		return fmt.Errorf("config %q training.attention_segment_mask requires at least one type=plain block", source)
	}
	return nil
}

func resolvedPlainAttentionMask(spec BlockSpec, objective string) string {
	mode := normalizeAttentionMask(spec.AttentionMask)
	if mode != "" {
		return mode
	}
	if isMaskedTrainingObjective(objective) {
		return AttentionMaskBidirectional
	}
	return AttentionMaskCausal
}

func resolvedPlainAttentionMaskForObjective(spec BlockSpec, objective, rootObjective string) string {
	objective = normalizeTrainingObjective(objective)
	if normalizeTrainingObjective(rootObjective) == ObjectiveHybrid {
		if objective == ObjectiveHybridExample {
			return AttentionMaskHybridExample
		}
		if isMaskedTrainingObjective(objective) {
			return AttentionMaskBidirectional
		}
		return AttentionMaskCausal
	}
	return resolvedPlainAttentionMask(spec, objective)
}

func resolveBlockAttentionMasksForObjective(blocks []BlockSpec, objective, rootObjective string) []BlockSpec {
	if len(blocks) == 0 {
		return blocks
	}
	var out []BlockSpec
	for i, block := range blocks {
		if blockTypeKey(block) != "plain" {
			continue
		}
		resolved := resolvedPlainAttentionMaskForObjective(block, objective, rootObjective)
		if normalizeTrainingObjective(rootObjective) != ObjectiveHybrid && strings.TrimSpace(block.AttentionMask) != "" {
			continue
		}
		if normalizeAttentionMask(block.AttentionMask) == resolved {
			continue
		}
		if out == nil {
			out = make([]BlockSpec, len(blocks))
			copy(out, blocks)
		}
		out[i].AttentionMask = resolved
	}
	if out == nil {
		return blocks
	}
	return out
}

func validatePlainAttentionMask(cfg *ArchConfig, source string, b BlockSpec, groupName string, idx int) error {
	if blockTypeKey(b) != "plain" {
		return nil
	}
	raw := normalizeAttentionMask(b.AttentionMask)
	switch raw {
	case "", AttentionMaskCausal, AttentionMaskBidirectional, AttentionMaskNone:
	default:
		return fmt.Errorf("config %q %s[%d] type=plain has invalid attention_mask=%q (must be \"causal\", \"bidirectional\", or \"none\")", source, groupName, idx, b.AttentionMask)
	}
	if b.WindowSize > 0 && cfg.Training.EffectiveObjective() == ObjectiveHybrid && cfg.Training.HybridCLMFraction < 1 {
		return fmt.Errorf("config %q %s[%d] type=plain sets window_size but training.objective=\"hybrid\" can run masked secondary steps that require bidirectional attention", source, groupName, idx)
	}
	if b.WindowSize > 0 && cfg.Training.EffectiveObjective() == ObjectiveHybrid {
		return nil
	}
	if b.WindowSize > 0 && resolvedPlainAttentionMask(b, cfg.Training.DefaultConcreteObjective()) != AttentionMaskCausal {
		return fmt.Errorf("config %q %s[%d] type=plain sets window_size but resolved attention_mask is not causal", source, groupName, idx)
	}
	return nil
}
