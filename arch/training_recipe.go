package arch

import (
	"fmt"
	"math"
	"strings"
)

// EffectiveSeqLenForStep returns the sequence length active for a training
// step. The top-level seq_len remains the maximum/eval length; schedule entries
// choose smaller or equal training shapes until a later entry supersedes them.
func (t TrainingSpec) EffectiveSeqLenForStep(maxSeqLen, step int) int {
	if maxSeqLen <= 0 {
		return maxSeqLen
	}
	if len(t.SeqLenSchedule) == 0 {
		return maxSeqLen
	}
	out := maxSeqLen
	for _, pair := range t.SeqLenSchedule {
		if len(pair) != 2 {
			continue
		}
		if pair[0] > step {
			break
		}
		out = pair[1]
	}
	return out
}

// EffectiveMLMMaskProbForStep returns the MLM/MNTP mask probability active for
// a training step. Without a schedule it is the fixed mlm_mask_prob value.
func (t TrainingSpec) EffectiveMLMMaskProbForStep(step int) float64 {
	if len(t.MLMMaskProbSchedule) == 0 {
		return t.MLMMaskProb
	}
	if strings.EqualFold(strings.TrimSpace(t.MLMMaskProbScheduleMode), "linear") {
		return t.effectiveLinearMLMMaskProbForStep(step)
	}
	out := t.MLMMaskProb
	for _, pair := range t.MLMMaskProbSchedule {
		if len(pair) != 2 {
			continue
		}
		if int(pair[0]) > step {
			break
		}
		out = pair[1]
	}
	return out
}

func (t TrainingSpec) effectiveLinearMLMMaskProbForStep(step int) float64 {
	sched := t.MLMMaskProbSchedule
	if len(sched) == 0 {
		return t.MLMMaskProb
	}
	if step <= int(sched[0][0]) {
		return sched[0][1]
	}
	for i := 1; i < len(sched); i++ {
		prevStep := int(sched[i-1][0])
		nextStep := int(sched[i][0])
		prevProb := sched[i-1][1]
		nextProb := sched[i][1]
		if step < nextStep {
			den := nextStep - prevStep
			if den <= 0 {
				return nextProb
			}
			progress := float64(step-prevStep) / float64(den)
			return prevProb + (nextProb-prevProb)*progress
		}
	}
	return sched[len(sched)-1][1]
}

// EffectiveHybridCLMFractionForStep returns the causal-batch probability active
// for a hybrid training step. Without a schedule it is hybrid_clm_fraction.
func (t TrainingSpec) EffectiveHybridCLMFractionForStep(step int) float64 {
	if len(t.HybridCLMFractionSchedule) == 0 {
		return t.HybridCLMFraction
	}
	if strings.EqualFold(strings.TrimSpace(t.HybridCLMFractionScheduleMode), "linear") {
		return t.effectiveLinearHybridCLMFractionForStep(step)
	}
	out := t.HybridCLMFraction
	for _, pair := range t.HybridCLMFractionSchedule {
		if len(pair) != 2 {
			continue
		}
		if int(pair[0]) > step {
			break
		}
		out = pair[1]
	}
	return out
}

func (t TrainingSpec) effectiveLinearHybridCLMFractionForStep(step int) float64 {
	sched := t.HybridCLMFractionSchedule
	if len(sched) == 0 {
		return t.HybridCLMFraction
	}
	if step <= int(sched[0][0]) {
		return sched[0][1]
	}
	for i := 1; i < len(sched); i++ {
		prevStep := int(sched[i-1][0])
		nextStep := int(sched[i][0])
		prevFraction := sched[i-1][1]
		nextFraction := sched[i][1]
		if step < nextStep {
			den := nextStep - prevStep
			if den <= 0 {
				return nextFraction
			}
			progress := float64(step-prevStep) / float64(den)
			return prevFraction + (nextFraction-prevFraction)*progress
		}
	}
	return sched[len(sched)-1][1]
}

func (t TrainingSpec) HybridHasMaskedSteps() bool {
	if t.EffectiveObjective() != ObjectiveHybrid {
		return false
	}
	if len(t.HybridCLMFractionSchedule) == 0 {
		return t.HybridCLMFraction < 1
	}
	for _, pair := range t.HybridCLMFractionSchedule {
		if len(pair) == 2 && pair[1] < 1 {
			return true
		}
	}
	return false
}

func validateTrainingRecipeKnobs(cfg *ArchConfig, source string) error {
	if cfg.Training.ZLoss < 0 || math.IsNaN(cfg.Training.ZLoss) || math.IsInf(cfg.Training.ZLoss, 0) {
		return fmt.Errorf("config %q has invalid training.z_loss=%g (must be finite and >= 0)", source, cfg.Training.ZLoss)
	}
	if err := validateSeqLenSchedule(cfg, source); err != nil {
		return err
	}
	if err := validateMLMMaskProbSchedule(cfg, source); err != nil {
		return err
	}
	if err := validateHybridCLMFractionSchedule(cfg, source); err != nil {
		return err
	}
	if err := validateEarlyStopSpec(cfg, source); err != nil {
		return err
	}
	if len(cfg.Training.SeqLenSchedule) > 0 {
		if cfg.Training.Distillation != nil {
			return fmt.Errorf("config %q has training.seq_len_schedule but distillation teacher runtimes use fixed seq_len in v1", source)
		}
		if cfg.Training.Data2VecActive() {
			return fmt.Errorf("config %q has training.seq_len_schedule but training.data2vec teacher runtimes use fixed seq_len in v1", source)
		}
	}
	return nil
}

func validateEarlyStopSpec(cfg *ArchConfig, source string) error {
	spec := cfg.Training.EarlyStop
	if spec == nil {
		return nil
	}
	switch strings.ToLower(strings.TrimSpace(spec.Metric)) {
	case "", "val", "validation", "val_loss", "validation_loss":
	default:
		return fmt.Errorf("config %q has invalid training.early_stop.metric=%q (v1 supports \"val\")", source, spec.Metric)
	}
	if spec.Patience < 0 {
		return fmt.Errorf("config %q has invalid training.early_stop.patience=%d (must be >= 0)", source, spec.Patience)
	}
	if spec.MinDelta < 0 || math.IsNaN(spec.MinDelta) || math.IsInf(spec.MinDelta, 0) {
		return fmt.Errorf("config %q has invalid training.early_stop.min_delta=%g (must be finite and >= 0)", source, spec.MinDelta)
	}
	if spec.MinSteps < 0 {
		return fmt.Errorf("config %q has invalid training.early_stop.min_steps=%d (must be >= 0)", source, spec.MinSteps)
	}
	if spec.ValGT < 0 || math.IsNaN(spec.ValGT) || math.IsInf(spec.ValGT, 0) {
		return fmt.Errorf("config %q has invalid training.early_stop.val_gt=%g (must be finite and >= 0)", source, spec.ValGT)
	}
	if spec.AtStep < 0 {
		return fmt.Errorf("config %q has invalid training.early_stop.at_step=%d (must be >= 0)", source, spec.AtStep)
	}
	return nil
}

func validateSeqLenSchedule(cfg *ArchConfig, source string) error {
	sched := cfg.Training.SeqLenSchedule
	if len(sched) == 0 {
		return nil
	}
	prevStep := -1
	for i, pair := range sched {
		if len(pair) != 2 {
			return fmt.Errorf("config %q has invalid training.seq_len_schedule[%d] length=%d (must be [step, seq_len])", source, i, len(pair))
		}
		step, seqLen := pair[0], pair[1]
		if i == 0 && step != 0 {
			return fmt.Errorf("config %q has invalid training.seq_len_schedule[0][0]=%d (first step must be 0)", source, step)
		}
		if step <= prevStep {
			return fmt.Errorf("config %q has invalid training.seq_len_schedule[%d][0]=%d (steps must be strictly increasing)", source, i, step)
		}
		if seqLen <= 0 || seqLen > cfg.SeqLen {
			return fmt.Errorf("config %q has invalid training.seq_len_schedule[%d][1]=%d (must be in [1, seq_len=%d])", source, i, seqLen, cfg.SeqLen)
		}
		if cfg.Training.BatchTokens%seqLen != 0 {
			return fmt.Errorf("config %q has invalid training.seq_len_schedule[%d][1]=%d (training.batch_tokens=%d must be divisible by every scheduled seq_len)", source, i, seqLen, cfg.Training.BatchTokens)
		}
		if cfg.MTP != nil && cfg.MTP.EffectiveN() > seqLen {
			return fmt.Errorf("config %q has invalid training.seq_len_schedule[%d][1]=%d (must be >= mtp.n=%d)", source, i, seqLen, cfg.MTP.EffectiveN())
		}
		if (cfg.Training.Objective == ObjectiveMNTP || cfg.Training.EffectiveHybridSecondaryObjective() == ObjectiveMNTP) && seqLen <= 1 {
			return fmt.Errorf("config %q has invalid training.seq_len_schedule[%d][1]=%d (MNTP requires seq_len > 1)", source, i, seqLen)
		}
		prevStep = step
	}
	return nil
}

func validateMLMMaskProbSchedule(cfg *ArchConfig, source string) error {
	mode := strings.ToLower(strings.TrimSpace(cfg.Training.MLMMaskProbScheduleMode))
	switch mode {
	case "", "step":
		cfg.Training.MLMMaskProbScheduleMode = "step"
	case "linear":
		cfg.Training.MLMMaskProbScheduleMode = "linear"
	default:
		return fmt.Errorf("config %q has invalid training.mlm_mask_prob_schedule_mode=%q (must be \"step\" or \"linear\")", source, cfg.Training.MLMMaskProbScheduleMode)
	}
	sched := cfg.Training.MLMMaskProbSchedule
	if len(sched) == 0 {
		return nil
	}
	prevStep := -1
	for i, pair := range sched {
		if len(pair) != 2 {
			return fmt.Errorf("config %q has invalid training.mlm_mask_prob_schedule[%d] length=%d (must be [step, prob])", source, i, len(pair))
		}
		stepFloat, prob := pair[0], pair[1]
		step := int(stepFloat)
		if float64(step) != stepFloat {
			return fmt.Errorf("config %q has invalid training.mlm_mask_prob_schedule[%d][0]=%g (step must be an integer)", source, i, stepFloat)
		}
		if i == 0 && step != 0 {
			return fmt.Errorf("config %q has invalid training.mlm_mask_prob_schedule[0][0]=%d (first step must be 0)", source, step)
		}
		if step <= prevStep {
			return fmt.Errorf("config %q has invalid training.mlm_mask_prob_schedule[%d][0]=%d (steps must be strictly increasing)", source, i, step)
		}
		if prob < 0 || prob > 1 || math.IsNaN(prob) || math.IsInf(prob, 0) {
			return fmt.Errorf("config %q has invalid training.mlm_mask_prob_schedule[%d][1]=%g (must be in [0,1])", source, i, prob)
		}
		prevStep = step
	}
	return nil
}

func validateHybridCLMFractionSchedule(cfg *ArchConfig, source string) error {
	mode := strings.ToLower(strings.TrimSpace(cfg.Training.HybridCLMFractionScheduleMode))
	switch mode {
	case "", "step":
		cfg.Training.HybridCLMFractionScheduleMode = "step"
	case "linear":
		cfg.Training.HybridCLMFractionScheduleMode = "linear"
	default:
		return fmt.Errorf("config %q has invalid training.hybrid_clm_fraction_schedule_mode=%q (must be \"step\" or \"linear\")", source, cfg.Training.HybridCLMFractionScheduleMode)
	}
	sched := cfg.Training.HybridCLMFractionSchedule
	if len(sched) == 0 {
		return nil
	}
	if cfg.Training.EffectiveObjective() != ObjectiveHybrid {
		return fmt.Errorf("config %q sets training.hybrid_clm_fraction_schedule but training.objective=%q (must be \"hybrid\")", source, cfg.Training.EffectiveObjective())
	}
	prevStep := -1
	for i, pair := range sched {
		if len(pair) != 2 {
			return fmt.Errorf("config %q has invalid training.hybrid_clm_fraction_schedule[%d] length=%d (must be [step, fraction])", source, i, len(pair))
		}
		stepFloat, fraction := pair[0], pair[1]
		step := int(stepFloat)
		if float64(step) != stepFloat {
			return fmt.Errorf("config %q has invalid training.hybrid_clm_fraction_schedule[%d][0]=%g (step must be an integer)", source, i, stepFloat)
		}
		if i == 0 && step != 0 {
			return fmt.Errorf("config %q has invalid training.hybrid_clm_fraction_schedule[0][0]=%d (first step must be 0)", source, step)
		}
		if step <= prevStep {
			return fmt.Errorf("config %q has invalid training.hybrid_clm_fraction_schedule[%d][0]=%d (steps must be strictly increasing)", source, i, step)
		}
		if fraction < 0 || fraction > 1 || math.IsNaN(fraction) || math.IsInf(fraction, 0) {
			return fmt.Errorf("config %q has invalid training.hybrid_clm_fraction_schedule[%d][1]=%g (must be in [0,1])", source, i, fraction)
		}
		prevStep = step
	}
	return nil
}
