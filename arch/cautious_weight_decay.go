package arch

import "fmt"

// EffectiveCautiousWeightDecayActivationStep returns the first zero-based
// training step at which cautious weight decay should replace standard decay.
func (t TrainingSpec) EffectiveCautiousWeightDecayActivationStep() int {
	if !t.CautiousWeightDecay || t.CautiousWeightDecayActivationFrac <= 0 {
		return 0
	}
	total := t.TotalSteps()
	if total <= 0 {
		return 0
	}
	return int(t.CautiousWeightDecayActivationFrac * float64(total))
}

func validateCautiousWeightDecay(cfg *ArchConfig, source string) error {
	frac := cfg.Training.CautiousWeightDecayActivationFrac
	// frac != frac is a NaN check (NaN is not equal to itself).
	if frac < 0 || frac > 1 || frac != frac {
		return fmt.Errorf("config %q has invalid training.cautious_weight_decay_activation_frac=%g (must be in [0,1])", source, frac)
	}
	return nil
}
