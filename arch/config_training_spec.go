package arch

import "strings"

// WarmupStepsConfigured reports whether training.warmup_steps was provided.
func (t TrainingSpec) WarmupStepsConfigured() bool {
	return t.warmupStepsSet || t.WarmupSteps != 0
}

// WarmupRatioConfigured reports whether training.warmup_ratio was provided.
func (t TrainingSpec) WarmupRatioConfigured() bool {
	return t.warmupRatioSet || t.WarmupRatio != 0
}

// HoldStepsConfigured reports whether training.hold_steps was provided.
func (t TrainingSpec) HoldStepsConfigured() bool {
	return t.holdStepsSet || t.HoldSteps != 0
}

// TotalSteps returns the effective training step count.
// When phases are configured, their summed length takes precedence.
func (t TrainingSpec) TotalSteps() int {
	if len(t.Phases) == 0 {
		return t.Steps
	}
	total := 0
	for _, phase := range t.Phases {
		total += phase.Steps
	}
	return total
}

// EffectiveLRScheduleSteps returns the horizon used by the standard cosine
// schedule. It is independent from TotalSteps so epoch-bounded references can
// stop before their configured scheduler horizon.
func (t TrainingSpec) EffectiveLRScheduleSteps() int {
	if t.LRScheduleSteps > 0 {
		return t.LRScheduleSteps
	}
	return t.Steps
}

// EffectiveComputeDType returns the training compute dtype, defaulting to fp32.
func (t TrainingSpec) EffectiveComputeDType() string {
	dtype := strings.ToLower(strings.TrimSpace(t.ComputeDType))
	if dtype == "" {
		return "float32"
	}
	return dtype
}

// DefaultTrainingSpec returns sensible training defaults.
func DefaultTrainingSpec() TrainingSpec {
	return TrainingSpec{
		Steps:             200,
		LR:                3e-4,
		WeightDecay:       0.01,
		Beta1:             0.9,
		Beta2:             0.95,
		Epsilon:           1e-8,
		LAMBBeta1:         0.9,
		LAMBBeta2:         0.999,
		LAMBEps:           1e-6,
		LAMBTrustRatioCap: 10,
		Seed:              42,
		BatchTokens:       1024,
		MLMMaskUnit:       MLMMaskUnitToken,
		TTTMode:           "full",
		TTTLR:             1e-5,
		TTTRank:           4,
		QAT:               "none",
		MuonMomentum:      0.9,
		MuonBackendSteps:  5,
		EmbedWeightDecay:  0.01,
		MatrixWeightDecay: 0.01,
		ScalarWeightDecay: 0.01,
		HeadWeightDecay:   0.01,
		SWADecay:          0.999,
		SWAInterval:       10,
	}
}

// ApplyDefaults fills omitted zero-valued training fields using the same
// defaults applied when parsing a JSON architecture config.
func (t *TrainingSpec) ApplyDefaults() {
	if t == nil {
		return
	}
	d := DefaultTrainingSpec()
	if t.Steps <= 0 {
		t.Steps = d.Steps
	}
	if t.LR <= 0 {
		t.LR = d.LR
	}
	// Note: seed=0 in JSON is indistinguishable from omitted; defaults to 42.
	if t.Seed <= 0 {
		t.Seed = d.Seed
	}
	if t.BatchTokens <= 0 && !t.BatchSizeConfigured() {
		t.BatchTokens = d.BatchTokens
	}
	t.Objective = normalizeTrainingObjective(t.Objective)
	t.MLMMaskUnit = normalizeMLMMaskUnit(t.MLMMaskUnit)
	for i := range t.MLMMaskUnitSchedule {
		t.MLMMaskUnitSchedule[i].Unit = strings.ToLower(strings.TrimSpace(t.MLMMaskUnitSchedule[i].Unit))
	}
	if !t.mlmMaskProbSet && t.MLMMaskProb == 0 {
		t.MLMMaskProb = 0.15
	}
	if !t.mlmReplacementProbSet && t.MLMMaskTokenProb == 0 && t.MLMRandomTokenProb == 0 && t.MLMKeptUnchangedProb == 0 {
		t.MLMMaskTokenProb = 0.8
		t.MLMRandomTokenProb = 0.1
		t.MLMKeptUnchangedProb = 0.1
	}
	if !t.hybridCLMFractionSet && t.HybridCLMFraction == 0 {
		t.HybridCLMFraction = 0.5
	}
	if strings.TrimSpace(t.HybridSecondaryObjective) == "" {
		t.HybridSecondaryObjective = ObjectiveMNTP
	} else {
		t.HybridSecondaryObjective = normalizeTrainingObjective(t.HybridSecondaryObjective)
	}
	t.HybridMixGranularity = t.EffectiveHybridMixGranularity()
	t.AttentionSegmentMask = t.EffectiveAttentionSegmentMask()
	if t.Data2Vec != nil {
		t.Data2Vec.applyDefaults()
	}
	if t.RTD != nil {
		t.RTD.applyDefaults(t.MLMMaskProb)
	}
	if t.MinimalPair != nil {
		t.MinimalPair.applyDefaults()
	}
	if t.Invariance != nil {
		t.Invariance.applyDefaults()
	}
	if t.PLLMargin != nil {
		t.PLLMargin.applyDefaults()
	}
	if t.WordStructuralObjective != nil {
		t.WordStructuralObjective.applyDefaults(t.MLMMaskTokenID)
	}
	if t.NewBob != nil {
		t.NewBob.applyDefaults()
	}
	if !t.weightDecaySet && t.WeightDecay == 0 {
		t.WeightDecay = d.WeightDecay
	}
	if t.Beta1 == 0 {
		t.Beta1 = d.Beta1
	}
	if t.Beta2 == 0 {
		t.Beta2 = d.Beta2
	}
	if t.Epsilon == 0 {
		t.Epsilon = d.Epsilon
	}
	if !t.lambBeta1Set && t.LAMBBeta1 == 0 {
		t.LAMBBeta1 = d.LAMBBeta1
	}
	if !t.lambBeta2Set && t.LAMBBeta2 == 0 {
		t.LAMBBeta2 = d.LAMBBeta2
	}
	if !t.lambEpsSet && t.LAMBEps == 0 {
		t.LAMBEps = d.LAMBEps
	}
	if !t.lambTrustRatioCapSet && t.LAMBTrustRatioCap == 0 {
		t.LAMBTrustRatioCap = d.LAMBTrustRatioCap
	}
	if t.EmbedLR == 0 {
		t.EmbedLR = float32(t.LR)
	}
	if t.MatrixLR == 0 {
		t.MatrixLR = float32(t.LR)
	}
	if t.ScalarLR == 0 {
		t.ScalarLR = float32(t.LR)
	}
	if t.HeadLR == 0 {
		t.HeadLR = float32(t.LR)
	}
	if t.TTTLR == 0 {
		t.TTTLR = d.TTTLR
	}
	if t.TTTMode == "" {
		t.TTTMode = d.TTTMode
	}
	t.QAT = strings.ToLower(strings.TrimSpace(t.QAT))
	if t.QAT == "" {
		t.QAT = d.QAT
	}
	if t.TTTRank == 0 {
		t.TTTRank = d.TTTRank
	}
	if t.MuonMomentum == 0 {
		t.MuonMomentum = t.Beta1
	}
	if t.MuonBackendSteps <= 0 {
		t.MuonBackendSteps = d.MuonBackendSteps
	}
	t.Optimizer = strings.ToLower(strings.TrimSpace(t.Optimizer))
	t.ComputeDType = strings.ToLower(strings.TrimSpace(t.ComputeDType))
	t.WeightInit = strings.ToLower(strings.TrimSpace(t.WeightInit))
	t.WeightDecayPolicy = t.EffectiveWeightDecayPolicy()
	if !t.embedWeightDecaySet && t.EmbedWeightDecay == 0 {
		t.EmbedWeightDecay = t.WeightDecay
	}
	if !t.matrixWeightDecaySet && t.MatrixWeightDecay == 0 {
		t.MatrixWeightDecay = t.WeightDecay
	}
	if !t.scalarWeightDecaySet && t.ScalarWeightDecay == 0 {
		t.ScalarWeightDecay = t.WeightDecay
	}
	if !t.headWeightDecaySet && t.HeadWeightDecay == 0 {
		t.HeadWeightDecay = t.WeightDecay
	}
	if !t.swaDecaySet && t.SWADecay == 0 {
		t.SWADecay = d.SWADecay
	}
	if !t.swaIntervalSet && t.SWAInterval <= 0 {
		t.SWAInterval = d.SWAInterval
	}
}
