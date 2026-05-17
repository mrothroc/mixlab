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

// PhaseStartSteps returns the zero-based start step for each configured
// recurrence phase. The first returned step is always 0 for valid configs.
// A nil return means the config does not use recurrence_phases.
func (c *ArchConfig) PhaseStartSteps() []int {
	if c == nil || len(c.RecurrencePhases) == 0 {
		return nil
	}
	total := c.Training.TotalSteps()
	if total <= 0 {
		return nil
	}
	steps := make([]int, len(c.RecurrencePhases))
	for i, phase := range c.RecurrencePhases {
		steps[i] = int(phase.Frac * float64(total))
	}
	return steps
}
