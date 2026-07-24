package arch

import (
	"fmt"
	"math"
	"strings"
)

const (
	ObjectiveClassification = "classification"

	ClassificationPoolingLast = "last"
	ClassificationPoolingMean = "mean"
)

// ClassificationSpec configures native sequence-level single-label
// classification. Labels are supplied by a labeled sequence dataset.
type ClassificationSpec struct {
	NumLabels         int      `json:"num_labels"`
	Pooling           string   `json:"pooling,omitempty"`
	ClassifierDropout *float32 `json:"classifier_dropout,omitempty"`
}

func (s *ClassificationSpec) effectivePooling(blocks []BlockSpec) string {
	if s == nil {
		return ""
	}
	if mode := strings.ToLower(strings.TrimSpace(s.Pooling)); mode != "" {
		return mode
	}
	seenCausal := false
	seenBidirectional := false
	for _, block := range blocks {
		switch blockTypeKey(block) {
		case "plain":
			switch normalizeAttentionMask(block.AttentionMask) {
			case AttentionMaskBidirectional, AttentionMaskNone:
				seenBidirectional = true
			default:
				seenCausal = true
			}
		case "ttt_mlp", "mamba", "mamba3", "gated_linear_ssm", "rwkv", "retnet",
			"gated_deltanet", "hgrn2", "mlstm":
			seenCausal = true
		}
	}
	if seenBidirectional && !seenCausal {
		return ClassificationPoolingMean
	}
	return ClassificationPoolingLast
}

func (s *ClassificationSpec) effectiveDropout(hiddenDropout float32) float32 {
	if s == nil || s.ClassifierDropout == nil {
		return hiddenDropout
	}
	return *s.ClassifierDropout
}

func (c *ArchConfig) ClassificationEnabled() bool {
	return c != nil && c.Training.EffectiveObjective() == ObjectiveClassification
}

func (c *ArchConfig) EffectiveClassificationPooling() string {
	if c == nil || c.Training.Classification == nil {
		return ""
	}
	return c.Training.Classification.effectivePooling(c.Blocks)
}

func (c *ArchConfig) EffectiveClassifierDropout() float32 {
	if c == nil || c.Training.Classification == nil {
		return 0
	}
	return c.Training.Classification.effectiveDropout(c.EffectiveHiddenDropout())
}

func classificationWeightShapes(modelDim int, spec *ClassificationSpec) []WeightMeta {
	if spec == nil || spec.NumLabels <= 0 {
		return nil
	}
	return []WeightMeta{
		{Name: "head_classifier_proj", Shape: []int{modelDim, spec.NumLabels}},
		{Name: "head_classifier_bias", Shape: []int{spec.NumLabels}, InitZero: true},
	}
}

func validateTrainingClassification(cfg *ArchConfig, source string) error {
	t := &cfg.Training
	if t.EffectiveObjective() != ObjectiveClassification {
		if t.Classification != nil {
			return fmt.Errorf("config %q sets training.classification but training.objective=%q; classification settings require objective=%q", source, t.EffectiveObjective(), ObjectiveClassification)
		}
		return nil
	}
	s := t.Classification
	if s == nil {
		return fmt.Errorf("config %q training.objective=%q requires training.classification", source, ObjectiveClassification)
	}
	if s.NumLabels < 2 {
		return fmt.Errorf("config %q training.classification.num_labels=%d must be >= 2", source, s.NumLabels)
	}
	s.Pooling = s.effectivePooling(cfg.Blocks)
	switch s.Pooling {
	case ClassificationPoolingLast, ClassificationPoolingMean:
	default:
		return fmt.Errorf("config %q training.classification.pooling=%q must be %q or %q", source, s.Pooling, ClassificationPoolingLast, ClassificationPoolingMean)
	}
	dropout := s.effectiveDropout(cfg.EffectiveHiddenDropout())
	if math.IsNaN(float64(dropout)) || math.IsInf(float64(dropout), 0) || dropout < 0 || dropout > 1 {
		return fmt.Errorf("config %q training.classification.classifier_dropout=%g must be finite and in [0,1]", source, dropout)
	}
	if cfg.MTP != nil || t.FirstByteMask || t.Distillation != nil || t.Data2Vec != nil ||
		t.ExampleFraming != nil || t.RTD != nil || t.MinimalPair != nil || t.Invariance != nil ||
		t.PLLMargin != nil || t.WordStructuralObjective != nil || t.Diffusion != nil ||
		t.EffectiveAttentionSegmentMask() != "" || len(t.SeqLenSchedule) > 0 {
		return fmt.Errorf("config %q native classification cannot be combined with MTP, masked/auxiliary objectives, example framing, segment masking, diffusion, or seq_len_schedule in v1", source)
	}
	if len(t.Heads) > 0 || strings.TrimSpace(t.ExportHead) != "" || strings.TrimSpace(t.DiffusionHead) != "" {
		return fmt.Errorf("config %q native classification does not use multihead heads/export_head/diffusion_head", source)
	}
	if t.ReverseComplementProb != 0 {
		return fmt.Errorf("config %q native classification does not support reverse_complement_prob in v1", source)
	}
	return nil
}
