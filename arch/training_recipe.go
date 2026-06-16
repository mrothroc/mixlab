package arch

import (
	"fmt"
	"math"
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
