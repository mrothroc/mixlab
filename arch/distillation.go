package arch

import (
	"fmt"
	"math"
	"strings"
)

const (
	DistillationMeanLogits   = "mean_logits"
	DistillationMeanLogProbs = "mean_logprobs"
)

func (d *DistillationSpec) Enabled() bool {
	return d != nil
}

func (d *DistillationSpec) EffectiveEnsembleStrategy() string {
	if d == nil {
		return ""
	}
	strategy := strings.ToLower(strings.TrimSpace(d.EnsembleStrategy))
	if strategy == "" {
		return DistillationMeanLogits
	}
	return strategy
}

func validateTrainingDistillation(cfg *ArchConfig, source string) error {
	if cfg == nil || cfg.Training.Distillation == nil {
		return nil
	}
	d := cfg.Training.Distillation
	if len(d.TeacherCheckpoints) == 0 {
		return fmt.Errorf("config %q training.distillation.teacher_checkpoints must include at least one checkpoint", source)
	}
	if len(d.TeacherConfigs) == 0 {
		return fmt.Errorf("config %q training.distillation.teacher_configs must include at least one config", source)
	}
	if len(d.TeacherCheckpoints) != len(d.TeacherConfigs) {
		return fmt.Errorf("config %q training.distillation teacher_checkpoints length=%d must match teacher_configs length=%d", source, len(d.TeacherCheckpoints), len(d.TeacherConfigs))
	}
	for i, path := range d.TeacherCheckpoints {
		if strings.TrimSpace(path) == "" {
			return fmt.Errorf("config %q training.distillation.teacher_checkpoints[%d] is empty", source, i)
		}
	}
	for i, path := range d.TeacherConfigs {
		if strings.TrimSpace(path) == "" {
			return fmt.Errorf("config %q training.distillation.teacher_configs[%d] is empty", source, i)
		}
	}
	if d.LossWeightKL < 0 || math.IsNaN(d.LossWeightKL) || math.IsInf(d.LossWeightKL, 0) {
		return fmt.Errorf("config %q has invalid training.distillation.loss_weight_kl=%g (must be finite and >= 0)", source, d.LossWeightKL)
	}
	if d.LossWeightCE < 0 || math.IsNaN(d.LossWeightCE) || math.IsInf(d.LossWeightCE, 0) {
		return fmt.Errorf("config %q has invalid training.distillation.loss_weight_ce=%g (must be finite and >= 0)", source, d.LossWeightCE)
	}
	if d.LossWeightKL+d.LossWeightCE <= 0 {
		return fmt.Errorf("config %q training.distillation loss weights must sum to > 0", source)
	}
	d.EnsembleStrategy = d.EffectiveEnsembleStrategy()
	switch d.EnsembleStrategy {
	case DistillationMeanLogits, DistillationMeanLogProbs:
	default:
		return fmt.Errorf("config %q has invalid training.distillation.ensemble_strategy=%q (must be \"mean_logits\" or \"mean_logprobs\")", source, d.EnsembleStrategy)
	}
	if cfg.Training.EffectiveObjective() != ObjectiveCausal {
		return fmt.Errorf("config %q training.distillation is only supported with training.objective=\"causal\" in v1", source)
	}
	if cfg.MTP != nil {
		return fmt.Errorf("config %q training.distillation cannot be combined with top-level mtp in v1", source)
	}
	if cfg.Training.FirstByteMask {
		return fmt.Errorf("config %q training.distillation cannot be combined with training.first_byte_mask in v1", source)
	}
	return nil
}
