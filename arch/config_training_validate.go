package arch

import (
	"fmt"
	"math"
	"strings"
)

// validateCommonTrainingSettings is shared by sequence and spatial task builders.
func validateCommonTrainingSettings(cfg *ArchConfig, source string) error {
	if cfg.Training.WeightDecay < 0 {
		return fmt.Errorf("config %q has invalid training.weight_decay=%g (must be >= 0)", source, cfg.Training.WeightDecay)
	}
	if cfg.Training.EmbedLR < 0 || cfg.Training.MatrixLR < 0 || cfg.Training.ScalarLR < 0 || cfg.Training.HeadLR < 0 {
		return fmt.Errorf("config %q has invalid per-group learning rate (must be >= 0)", source)
	}
	switch cfg.Training.EffectiveWeightDecayPolicy() {
	case WeightDecayPolicyMatrixOnly, WeightDecayPolicyAll:
	default:
		return fmt.Errorf("config %q has invalid training.weight_decay_policy=%q (must be \"matrix_only\" or \"all\")", source, cfg.Training.WeightDecayPolicy)
	}
	if cfg.Training.MuonMomentum < 0 {
		return fmt.Errorf("config %q has invalid training.muon_momentum=%g (must be >= 0)", source, cfg.Training.MuonMomentum)
	}
	switch cfg.Training.Optimizer {
	case "", "adamw", "muon", "muon_eq_r", "normuon", "lamb":
	default:
		return fmt.Errorf("config %q has invalid training.optimizer=%q (must be \"adamw\", \"muon\", \"muon_eq_r\", \"normuon\", or \"lamb\")", source, cfg.Training.Optimizer)
	}
	if err := validateWeightInitialization(cfg, source); err != nil {
		return err
	}
	if err := validateTrainingLearningRates(cfg.Training, source); err != nil {
		return err
	}
	switch cfg.Training.EffectiveComputeDType() {
	case "float32", "bf16":
	default:
		return fmt.Errorf("config %q has invalid training.compute_dtype=%q (must be \"float32\" or \"bf16\")", source, cfg.Training.ComputeDType)
	}
	if cfg.Training.LAMBBeta1 < 0 || cfg.Training.LAMBBeta1 >= 1 {
		return fmt.Errorf("config %q has invalid training.lamb_beta1=%g (must be in [0,1))", source, cfg.Training.LAMBBeta1)
	}
	if cfg.Training.LAMBBeta2 < 0 || cfg.Training.LAMBBeta2 >= 1 {
		return fmt.Errorf("config %q has invalid training.lamb_beta2=%g (must be in [0,1))", source, cfg.Training.LAMBBeta2)
	}
	if cfg.Training.LAMBEps <= 0 {
		return fmt.Errorf("config %q has invalid training.lamb_eps=%g (must be > 0)", source, cfg.Training.LAMBEps)
	}
	if cfg.Training.LAMBTrustRatioCap < 0 || math.IsNaN(float64(cfg.Training.LAMBTrustRatioCap)) || math.IsInf(float64(cfg.Training.LAMBTrustRatioCap), 0) {
		return fmt.Errorf("config %q has invalid training.lamb_trust_ratio_cap=%g (must be finite and >= 0; 0 disables capping)", source, cfg.Training.LAMBTrustRatioCap)
	}
	switch strings.ToLower(strings.TrimSpace(cfg.Training.NewtonSchulzVariant)) {
	case "", "fixed":
		cfg.Training.NewtonSchulzVariant = "fixed"
	case "polar_express":
		cfg.Training.NewtonSchulzVariant = "polar_express"
	default:
		return fmt.Errorf("config %q has invalid training.newton_schulz_variant=%q (must be \"fixed\" or \"polar_express\")", source, cfg.Training.NewtonSchulzVariant)
	}
	if cfg.Training.GradClip < 0 {
		return fmt.Errorf("config %q has invalid training.grad_clip=%g (must be >= 0)", source, cfg.Training.GradClip)
	}
	if cfg.Training.MinLRFraction < 0 || cfg.Training.MinLRFraction >= 1 {
		return fmt.Errorf("config %q has invalid training.min_lr_fraction=%g (must be in [0,1))", source, cfg.Training.MinLRFraction)
	}
	if cfg.Training.EmbedWeightDecay < 0 || cfg.Training.MatrixWeightDecay < 0 ||
		cfg.Training.ScalarWeightDecay < 0 || cfg.Training.HeadWeightDecay < 0 {
		return fmt.Errorf("config %q has invalid per-group weight decay (must be >= 0)", source)
	}
	if cfg.Training.SWAStart < 0 {
		return fmt.Errorf("config %q has invalid training.swa_start=%d (must be >= 0)", source, cfg.Training.SWAStart)
	}
	if cfg.Training.SWADecay < 0 || cfg.Training.SWADecay >= 1 {
		return fmt.Errorf("config %q has invalid training.swa_decay=%g (must be in [0,1))", source, cfg.Training.SWADecay)
	}
	if cfg.Training.SWAInterval <= 0 {
		return fmt.Errorf("config %q has invalid training.swa_interval=%d (must be > 0)", source, cfg.Training.SWAInterval)
	}
	if cfg.Training.WarmupSteps < 0 {
		return fmt.Errorf("config %q has invalid training.warmup_steps=%d (must be >= 0)", source, cfg.Training.WarmupSteps)
	}
	if cfg.Training.LRScheduleSteps < 0 {
		return fmt.Errorf("config %q has invalid training.lr_schedule_steps=%d (must be >= 0)", source, cfg.Training.LRScheduleSteps)
	}
	if cfg.Training.WarmupRatio < 0 || cfg.Training.WarmupRatio > 1 {
		return fmt.Errorf("config %q has invalid training.warmup_ratio=%g (must be in [0,1])", source, cfg.Training.WarmupRatio)
	}
	if cfg.Training.WarmupStepsConfigured() && cfg.Training.WarmupRatioConfigured() {
		return fmt.Errorf("config %q cannot set both training.warmup_steps and training.warmup_ratio", source)
	}
	if cfg.Training.HoldSteps < 0 {
		return fmt.Errorf("config %q has invalid training.hold_steps=%d (must be >= 0)", source, cfg.Training.HoldSteps)
	}
	if cfg.Training.WarmdownSteps < 0 {
		return fmt.Errorf("config %q has invalid training.warmdown_steps=%d (must be >= 0)", source, cfg.Training.WarmdownSteps)
	}
	if cfg.Training.RecurrenceActivationFrac < 0 || cfg.Training.RecurrenceActivationFrac > 1 {
		return fmt.Errorf("config %q has invalid training.recurrence_activation_frac=%g (must be in [0,1])", source, cfg.Training.RecurrenceActivationFrac)
	}
	if cfg.Training.RecurrenceActivationStep < 0 {
		return fmt.Errorf("config %q has invalid training.recurrence_activation_step=%d (must be >= 0)", source, cfg.Training.RecurrenceActivationStep)
	}
	if cfg.Training.RecurrenceActivationFrac > 0 && cfg.Training.RecurrenceActivationStep > 0 {
		return fmt.Errorf("config %q cannot set both training.recurrence_activation_frac and training.recurrence_activation_step", source)
	}
	if err := validateCautiousWeightDecay(cfg, source); err != nil {
		return err
	}
	for i, phase := range cfg.Training.Phases {
		if phase.Steps <= 0 {
			return fmt.Errorf("config %q has invalid training.phases[%d].steps=%d (must be > 0)", source, i, phase.Steps)
		}
		if phase.LR <= 0 {
			return fmt.Errorf("config %q has invalid training.phases[%d].lr=%g (must be > 0)", source, i, phase.LR)
		}
	}
	if err := validatePhaseBaseLR(cfg, source); err != nil {
		return err
	}
	if len(cfg.Training.Phases) > 0 {
		if cfg.Training.LRScheduleSteps != 0 {
			return fmt.Errorf("config %q cannot set training.lr_schedule_steps with training.phases; phase steps define their own schedule", source)
		}
		cfg.Training.Steps = cfg.Training.TotalSteps()
	}
	if err := validateRecurrencePhases(cfg, source); err != nil {
		return err
	}
	if cfg.Training.TargetValLoss < 0 {
		return fmt.Errorf("config %q has invalid training.target_val_loss=%g (must be >= 0)", source, cfg.Training.TargetValLoss)
	}
	if cfg.Training.HardwareTFLOPs < 0 {
		return fmt.Errorf("config %q has invalid training.hardware_tflops=%g (must be >= 0)", source, cfg.Training.HardwareTFLOPs)
	}
	if cfg.Training.TTTSteps < 0 {
		return fmt.Errorf("config %q has invalid training.ttt_steps=%d (must be >= 0)", source, cfg.Training.TTTSteps)
	}
	if cfg.Training.TTTMode != "full" && cfg.Training.TTTMode != "lora" {
		return fmt.Errorf("config %q has invalid training.ttt_mode=%q (must be \"full\" or \"lora\")", source, cfg.Training.TTTMode)
	}
	if cfg.Training.QAT != "none" && cfg.Training.QAT != "int8" && cfg.Training.QAT != "int6" {
		return fmt.Errorf("config %q has invalid training.qat=%q (must be \"none\", \"int8\", or \"int6\")", source, cfg.Training.QAT)
	}
	if cfg.Training.QATStart < 0 {
		return fmt.Errorf("config %q has invalid training.qat_start=%d (must be >= 0)", source, cfg.Training.QATStart)
	}
	if cfg.Training.QATStart > 0 && cfg.Training.QAT == "none" {
		return fmt.Errorf("config %q has training.qat_start=%d but training.qat is not set", source, cfg.Training.QATStart)
	}
	if cfg.Training.TTTLR < 0 {
		return fmt.Errorf("config %q has invalid training.ttt_lr=%g (must be >= 0)", source, cfg.Training.TTTLR)
	}
	if cfg.Training.TTTRank <= 0 {
		return fmt.Errorf("config %q has invalid training.ttt_rank=%d (must be > 0)", source, cfg.Training.TTTRank)
	}
	if err := validateEvalSpec(cfg, source); err != nil {
		return err
	}
	if err := ValidateDistributedConfig(cfg); err != nil {
		return fmt.Errorf("config %q: %w", source, err)
	}

	return nil
}
