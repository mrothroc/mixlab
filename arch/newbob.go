package arch

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"
)

const (
	LRScheduleCosine = "cosine"
	LRScheduleNewBob = "newbob"

	NewBobMetricValLoss      = "val_loss"
	NewBobMetricValErrorRate = "val_error_rate"
)

const (
	defaultNewBobAnnealingFactor      = 0.5
	defaultNewBobImprovementThreshold = 0.0025
)

// NewBobSpec configures validation-driven, monotonically decreasing learning
// rates. Metrics are normalized by the training validation layer so the
// scheduler itself only sees a lower-is-better scalar.
type NewBobSpec struct {
	AnnealingFactor      float64 `json:"annealing_factor,omitempty"`
	ImprovementThreshold float64 `json:"improvement_threshold"`
	Patient              int     `json:"patient,omitempty"`
	Metric               string  `json:"metric,omitempty"`

	annealingFactorSet      bool
	improvementThresholdSet bool
}

func (s *NewBobSpec) UnmarshalJSON(data []byte) error {
	type alias NewBobSpec
	var decoded alias
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}
	*s = NewBobSpec(decoded)
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return err
	}
	_, s.annealingFactorSet = fields["annealing_factor"]
	_, s.improvementThresholdSet = fields["improvement_threshold"]
	return nil
}

func (s *NewBobSpec) applyDefaults() {
	if s == nil {
		return
	}
	if !s.annealingFactorSet && s.AnnealingFactor == 0 {
		s.AnnealingFactor = defaultNewBobAnnealingFactor
	}
	if !s.improvementThresholdSet && s.ImprovementThreshold == 0 {
		s.ImprovementThreshold = defaultNewBobImprovementThreshold
	}
	if strings.TrimSpace(s.Metric) == "" {
		s.Metric = NewBobMetricValLoss
	} else {
		s.Metric = strings.ToLower(strings.TrimSpace(s.Metric))
	}
}

// EffectiveLRSchedule returns the selected outer learning-rate schedule.
func (t TrainingSpec) EffectiveLRSchedule() string {
	mode := strings.ToLower(strings.TrimSpace(t.LRSchedule))
	if mode == "" {
		return LRScheduleCosine
	}
	return mode
}

// ConfiguredMetricValidation reports whether config, rather than the legacy
// sampled-monitoring defaults, owns validation cadence and split size.
func (t TrainingSpec) ConfiguredMetricValidation() bool {
	return t.valEveryStepsSet || t.ValEverySteps > 0 || t.EffectiveLRSchedule() == LRScheduleNewBob
}

func validateMetricDrivenLRSchedule(cfg *ArchConfig, source string) error {
	if cfg == nil {
		return nil
	}
	t := &cfg.Training
	if t.ValEverySteps < 0 {
		return fmt.Errorf("config %q has invalid training.val_every_steps=%d (must be > 0 when configured)", source, t.ValEverySteps)
	}
	if t.ValExamples < 0 {
		return fmt.Errorf("config %q has invalid training.val_examples=%d (must be >= 0; 0 means the full split)", source, t.ValExamples)
	}
	if t.valEveryStepsSet && t.ValEverySteps == 0 {
		return fmt.Errorf("config %q has invalid training.val_every_steps=0 (omit it for sampled monitoring or set a positive completed-step cadence)", source)
	}
	if t.valExamplesSet && !t.ConfiguredMetricValidation() {
		return fmt.Errorf("config %q training.val_examples requires training.val_every_steps", source)
	}
	if t.ConfiguredMetricValidation() && t.EffectiveObjective() != ObjectiveClassification {
		return fmt.Errorf("config %q configured full-split validation currently supports training.objective=%q only", source, ObjectiveClassification)
	}

	switch t.EffectiveLRSchedule() {
	case LRScheduleCosine:
		if t.NewBob != nil {
			return fmt.Errorf("config %q sets training.newbob but training.lr_schedule is not %q", source, LRScheduleNewBob)
		}
		return nil
	case LRScheduleNewBob:
	default:
		return fmt.Errorf("config %q has invalid training.lr_schedule=%q (must be %q or %q)", source, t.LRSchedule, LRScheduleCosine, LRScheduleNewBob)
	}

	if t.NewBob == nil {
		return fmt.Errorf("config %q training.lr_schedule=%q requires training.newbob", source, LRScheduleNewBob)
	}
	if t.EffectiveObjective() != ObjectiveClassification {
		return fmt.Errorf("config %q training.lr_schedule=%q currently supports training.objective=%q only", source, LRScheduleNewBob, ObjectiveClassification)
	}
	if t.ValEverySteps <= 0 {
		return fmt.Errorf("config %q training.lr_schedule=%q requires training.val_every_steps > 0", source, LRScheduleNewBob)
	}
	if len(t.Phases) > 0 || t.LRScheduleSteps > 0 || t.WarmupSteps > 0 || t.WarmupRatio > 0 ||
		t.HoldSteps > 0 || t.WarmdownSteps > 0 || t.MinLRFraction > 0 {
		return fmt.Errorf("config %q training.lr_schedule=%q cannot be combined with phases, lr_schedule_steps, warmup, hold, warmdown, or min_lr_fraction", source, LRScheduleNewBob)
	}
	s := t.NewBob
	if math.IsNaN(s.AnnealingFactor) || math.IsInf(s.AnnealingFactor, 0) || s.AnnealingFactor <= 0 || s.AnnealingFactor > 1 {
		return fmt.Errorf("config %q training.newbob.annealing_factor=%g must be finite and in (0,1]", source, s.AnnealingFactor)
	}
	if math.IsNaN(s.ImprovementThreshold) || math.IsInf(s.ImprovementThreshold, 0) || s.ImprovementThreshold < 0 {
		return fmt.Errorf("config %q training.newbob.improvement_threshold=%g must be finite and >= 0", source, s.ImprovementThreshold)
	}
	if s.Patient < 0 {
		return fmt.Errorf("config %q training.newbob.patient=%d must be >= 0", source, s.Patient)
	}
	switch s.Metric {
	case NewBobMetricValLoss, NewBobMetricValErrorRate:
	default:
		return fmt.Errorf("config %q training.newbob.metric=%q must be %q or %q", source, s.Metric, NewBobMetricValLoss, NewBobMetricValErrorRate)
	}
	return nil
}
