package arch

// EffectiveRecurrenceActivationStep returns the first step that should use the
// full recurrence execution path. A return value <= 0 preserves current behavior
// and activates recurrence from step 0.
func (t TrainingSpec) EffectiveRecurrenceActivationStep() int {
	if t.RecurrenceActivationStep > 0 {
		return t.RecurrenceActivationStep
	}
	if t.RecurrenceActivationFrac <= 0 {
		return 0
	}
	total := t.TotalSteps()
	if total <= 0 {
		return 0
	}
	return int(t.RecurrenceActivationFrac * float64(total))
}
