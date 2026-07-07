package arch

import (
	"encoding/json"
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

func (d *DistillationSpec) EffectiveKLActive() bool {
	return d != nil && d.LossWeightKL > 0
}

func (d *DistillationSpec) EffectiveTemperature() float64 {
	if d == nil || d.Temperature == 0 {
		return 1.0
	}
	return d.Temperature
}

func (t TrainingSpec) DistillationKLEffectiveActive() bool {
	return t.Distillation != nil && t.Distillation.EffectiveKLActive()
}

func (t TrainingSpec) DistillationActiveForConcreteObjective(objective string) bool {
	if !t.DistillationKLEffectiveActive() {
		return false
	}
	objective = normalizeTrainingObjective(objective)
	switch t.EffectiveObjective() {
	case ObjectiveCausal:
		return objective == ObjectiveCausal
	case ObjectiveMLM:
		return objective == ObjectiveMLM
	case ObjectiveMNTP:
		return objective == ObjectiveMNTP
	case ObjectiveHybrid:
		if objective == ObjectiveHybridExample {
			return true
		}
		return isMaskedDistillationObjective(objective)
	default:
		return false
	}
}

func (d *DistillationSpec) UnmarshalJSON(data []byte) error {
	type alias DistillationSpec
	var raw alias
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	*d = DistillationSpec(raw)
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return err
	}
	_, d.temperatureSet = fields["temperature"]
	return nil
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

func isMaskedDistillationObjective(objective string) bool {
	switch normalizeTrainingObjective(objective) {
	case ObjectiveMLM, ObjectiveMNTP:
		return true
	default:
		return false
	}
}

func validateTrainingDistillation(cfg *ArchConfig, source string) error {
	if cfg == nil || cfg.Training.Distillation == nil {
		return nil
	}
	d := cfg.Training.Distillation
	if d.LossWeightKL < 0 || math.IsNaN(d.LossWeightKL) || math.IsInf(d.LossWeightKL, 0) {
		return fmt.Errorf("config %q has invalid training.distillation.loss_weight_kl=%g (must be finite and >= 0)", source, d.LossWeightKL)
	}
	if d.LossWeightCE < 0 || math.IsNaN(d.LossWeightCE) || math.IsInf(d.LossWeightCE, 0) {
		return fmt.Errorf("config %q has invalid training.distillation.loss_weight_ce=%g (must be finite and >= 0)", source, d.LossWeightCE)
	}
	if d.LossWeightKL+d.LossWeightCE <= 0 {
		return fmt.Errorf("config %q training.distillation loss weights must sum to > 0", source)
	}
	if !d.temperatureSet && d.Temperature == 0 {
		d.Temperature = 1.0
	}
	if d.Temperature <= 0 || math.IsNaN(d.Temperature) || math.IsInf(d.Temperature, 0) {
		return fmt.Errorf("config %q has invalid training.distillation.temperature=%g (must be finite and > 0)", source, d.Temperature)
	}
	d.EnsembleStrategy = d.EffectiveEnsembleStrategy()
	switch d.EnsembleStrategy {
	case DistillationMeanLogits, DistillationMeanLogProbs:
	default:
		return fmt.Errorf("config %q has invalid training.distillation.ensemble_strategy=%q (must be \"mean_logits\" or \"mean_logprobs\")", source, d.EnsembleStrategy)
	}
	if !d.EffectiveKLActive() {
		if math.Abs(d.LossWeightCE-1.0) > 1e-9 {
			return fmt.Errorf("config %q training.distillation with loss_weight_kl=0 requires loss_weight_ce=1 for disabled-parity training", source)
		}
		return nil
	}
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
	switch cfg.Training.EffectiveObjective() {
	case ObjectiveCausal, ObjectiveMLM, ObjectiveMNTP:
	case ObjectiveHybrid:
		secondary := cfg.Training.EffectiveHybridSecondaryObjective()
		if !isMaskedDistillationObjective(secondary) {
			return fmt.Errorf("config %q training.distillation with objective=\"hybrid\" requires hybrid_secondary_objective \"mlm\" or \"mntp\"", source)
		}
		if !cfg.Training.HybridHasMaskedSteps() {
			return fmt.Errorf("config %q training.distillation with objective=\"hybrid\" requires masked secondary steps; hybrid_clm_fraction=1 disables them", source)
		}
	default:
		return fmt.Errorf("config %q training.distillation is only supported with training.objective \"causal\", \"mlm\", \"mntp\", or compatible \"hybrid\" in v1", source)
	}
	if cfg.MTP != nil {
		return fmt.Errorf("config %q training.distillation cannot be combined with top-level mtp in v1", source)
	}
	if cfg.Training.FirstByteMask {
		return fmt.Errorf("config %q training.distillation cannot be combined with training.first_byte_mask in v1", source)
	}
	return nil
}
