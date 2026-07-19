package arch

import (
	"fmt"
	"strings"
)

// BuildIRProgramFromConfig constructs an IR forward-pass program from an
// ArchConfig. The batchSize is derived from config (batch_tokens / seq_len).
func BuildIRProgramFromConfig(cfg *ArchConfig) (*Program, error) {
	if cfg == nil {
		return nil, fmt.Errorf("nil config")
	}
	if len(cfg.RecurrencePhases) > 0 {
		phaseIdx := MaxCostRecurrencePhaseIndex(cfg)
		if phaseIdx < 0 {
			return nil, fmt.Errorf("recurrence_phases configured but no max-cost phase found")
		}
		return BuildTrainingIRProgramForRecurrencePhaseFromConfig(cfg, phaseIdx, TrainingProgramState{
			RecurrenceActive: true,
			HeadUntied:       cfg.MTPUntieEnabled(),
			MTPAuxInactive:   false,
		})
	}

	return BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{
		RecurrenceActive: true,
		HeadUntied:       cfg.MTPUntieEnabled(),
		MTPAuxInactive:   false,
	})
}

// BuildEvalIRProgramFromConfig constructs a next-token-only program for
// validation, inference, and generation. MTP auxiliary losses are intentionally
// excluded because they are training-only signal.
func BuildEvalIRProgramFromConfig(cfg *ArchConfig) (*Program, error) {
	if cfg == nil {
		return nil, fmt.Errorf("nil config")
	}
	if cfg.Training.MultiheadEnabled() {
		return nil, fmt.Errorf("multihead eval IR is not a single-head next-token graph; export the configured scorer head or use native diffusion generation/scoring for the diffusion head")
	}
	if len(cfg.RecurrencePhases) > 0 {
		return buildIRProgramFromConfigWithStateAndOrder(cfg, TrainingProgramState{
			RecurrenceActive:       true,
			HeadUntied:             cfg.MTPUntieEnabled(),
			Objective:              ObjectiveCausal,
			DistillationInactive:   true,
			Data2VecInactive:       true,
			InvarianceInactive:     true,
			PLLMarginInactive:      true,
			ZLossInactive:          true,
			DropoutInactive:        true,
			SegmentMaskInactive:    true,
			ExampleFramingInactive: true,
		}, nil, cfg.RecurrencePhases[len(cfg.RecurrencePhases)-1].Order, false)
	}
	return buildIRProgramFromConfigWithState(cfg, TrainingProgramState{
		RecurrenceActive:       true,
		HeadUntied:             cfg.MTPUntieEnabled(),
		Objective:              ObjectiveCausal,
		DistillationInactive:   true,
		Data2VecInactive:       true,
		InvarianceInactive:     true,
		PLLMarginInactive:      true,
		ZLossInactive:          true,
		DropoutInactive:        true,
		SegmentMaskInactive:    true,
		ExampleFramingInactive: true,
	}, nil, false)
}

// BuildGenerationIRProgramFromConfig constructs a fixed-width causal
// generation graph. It gathers one normalized hidden row per batch element
// before the vocabulary projection so generation reads back [B,V], not
// [B*T,V].
func BuildGenerationIRProgramFromConfig(cfg *ArchConfig) (*Program, error) {
	prog, err := BuildEvalIRProgramFromConfig(cfg)
	if err != nil {
		return nil, err
	}
	if cfg.SeqLen <= 0 || cfg.Training.BatchTokens <= 0 || cfg.Training.BatchTokens%cfg.SeqLen != 0 {
		return nil, fmt.Errorf("generation requires batch_tokens to be a positive multiple of seq_len")
	}
	batchSize := cfg.Training.BatchTokens / cfg.SeqLen
	prog.DeclareInput("generation_positions", TensorInt32, []int{batchSize})
	// x_final_norm is [B*T,D]. Flattened row offsets allow every batch row to
	// select a different current position with the ordinary embedding gather.
	prog.Embed("x_final_norm", "generation_positions", "generation_hidden")
	if cfg.TieEmbeddings && !cfg.MTPUntieEnabled() {
		prog.Transpose(weightName(0), []int{1, 0}, "generation_tied_head")
		prog.MatMul("generation_hidden", "generation_tied_head", "generation_logits_raw")
	} else {
		prog.MatMul("generation_hidden", weightName(1), "generation_logits_raw")
	}
	logits := "generation_logits_raw"
	if cfg.LogitSoftcap > 0 {
		prog.ScalarMul(logits, 1/cfg.LogitSoftcap, "generation_logits_scaled")
		prog.Tanh("generation_logits_scaled", "generation_logits_tanh")
		prog.ScalarMul("generation_logits_tanh", cfg.LogitSoftcap, "generation_logits")
		logits = "generation_logits"
	}
	if logits != "generation_logits" {
		prog.ScalarMul(logits, 1, "generation_logits")
	}
	// The trainer evaluator historically requires a scalar loss. This constant
	// keeps generation output-only: the native evaluator recognizes it and
	// prunes the normal CE/logits branch from the compiled graph.
	prog.Full([]int{1}, 0, "generation_eval_loss")
	prog.DeclareOutput("generation_eval_loss", TensorFloat32, []int{1})
	prog.DeclareOutput("generation_logits", TensorFloat32, []int{batchSize, cfg.VocabSize})
	return prog, nil
}

// BuildDistillationTeacherIRProgramFromConfig constructs a dropout-free logits
// program for fixed-teacher distillation. Unlike normal eval IR, masked
// objectives keep their masked attention/head semantics so teacher logits match
// the student objective being distilled.
func BuildDistillationTeacherIRProgramFromConfig(cfg *ArchConfig, objective string) (*Program, error) {
	if cfg == nil {
		return nil, fmt.Errorf("nil config")
	}
	objective = normalizeTrainingObjective(objective)
	if objective == ObjectiveCausal {
		return BuildEvalIRProgramFromConfig(cfg)
	}
	if objective != ObjectiveMLM && objective != ObjectiveMNTP && objective != ObjectiveHybridExample {
		return nil, fmt.Errorf("distillation teacher objective %q is not supported", objective)
	}
	buildCfg := cfg
	if objective == ObjectiveHybridExample {
		copied := *cfg
		copied.Training.Objective = ObjectiveHybrid
		copied.Training.HybridMixGranularity = HybridMixGranularityExample
		if strings.TrimSpace(copied.Training.HybridSecondaryObjective) == "" {
			copied.Training.HybridSecondaryObjective = ObjectiveMNTP
		}
		buildCfg = &copied
	}
	return buildIRProgramFromConfigWithState(buildCfg, TrainingProgramState{
		RecurrenceActive:       true,
		HeadUntied:             cfg.MTPUntieEnabled(),
		MTPAuxInactive:         true,
		DistillationInactive:   true,
		Data2VecInactive:       true,
		InvarianceInactive:     true,
		PLLMarginInactive:      true,
		ZLossInactive:          true,
		DropoutInactive:        true,
		Objective:              objective,
		SegmentMaskInactive:    true,
		ExampleFramingInactive: true,
	}, nil, false)
}

// TrainingProgramState selects training-time graph schedules that can switch
// without changing the weight layout.
type TrainingProgramState struct {
	RecurrenceActive       bool
	HeadUntied             bool
	MTPAuxInactive         bool
	DistillationInactive   bool
	Data2VecInactive       bool
	InvarianceInactive     bool
	PLLMarginInactive      bool
	ZLossInactive          bool
	DropoutInactive        bool
	Objective              string
	HiddenCaptureTopK      int
	HiddenCapturePrefix    string
	SegmentMaskInactive    bool
	ExampleFramingInactive bool
}

// BuildTrainingIRProgramFromConfig constructs a training program with MTP
// auxiliary losses enabled when configured unless state disables them.
func BuildTrainingIRProgramFromConfig(cfg *ArchConfig, state TrainingProgramState) (*Program, error) {
	if cfg == nil {
		return nil, fmt.Errorf("nil config")
	}
	if cfg.Training.MultiheadEnabled() {
		return buildMultiheadTrainingIRProgramFromConfig(cfg, state)
	}
	return buildIRProgramFromConfigWithState(cfg, state, cfg.MTP, cfg.Training.FirstByteMask)
}

// BuildTrainingIRProgramForRecurrencePhaseFromConfig constructs a training
// program for one explicit recurrence_phases entry.
func BuildTrainingIRProgramForRecurrencePhaseFromConfig(cfg *ArchConfig, phaseIdx int, state TrainingProgramState) (*Program, error) {
	if cfg == nil {
		return nil, fmt.Errorf("nil config")
	}
	if phaseIdx < 0 || phaseIdx >= len(cfg.RecurrencePhases) {
		return nil, fmt.Errorf("recurrence phase index %d out of range [0,%d)", phaseIdx, len(cfg.RecurrencePhases))
	}
	return buildIRProgramFromConfigWithStateAndOrder(cfg, state, cfg.MTP, cfg.RecurrencePhases[phaseIdx].Order, cfg.Training.FirstByteMask)
}

// BuildPreActivationIRProgramFromConfig constructs the recurrence-inactive
// training program for configs that delay recurrence activation. The program
// keeps the full recurrence weight layout, but emits only the first occurrence
// of each recurrence root in order.
func BuildPreActivationIRProgramFromConfig(cfg *ArchConfig) (*Program, error) {
	if cfg == nil {
		return nil, fmt.Errorf("nil config")
	}
	if len(cfg.RecurrencePhases) > 0 {
		return BuildTrainingIRProgramForRecurrencePhaseFromConfig(cfg, 0, TrainingProgramState{
			RecurrenceActive: true,
			HeadUntied:       cfg.MTPUntieEnabled(),
			MTPAuxInactive:   false,
		})
	}
	return BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{
		RecurrenceActive: false,
		HeadUntied:       cfg.MTPUntieEnabled(),
		MTPAuxInactive:   false,
	})
}

func buildIRProgramFromConfigWithState(cfg *ArchConfig, state TrainingProgramState, mtp *MTPSpec, firstByteMask bool) (*Program, error) {
	return buildIRProgramFromConfigWithStateAndOrder(cfg, state, mtp, nil, firstByteMask)
}

func buildIRProgramFromConfigWithStateAndOrder(cfg *ArchConfig, state TrainingProgramState, mtp *MTPSpec, phaseOrder []int, firstByteMask bool) (*Program, error) {
	batchSize := 1
	if cfg.Training.BatchTokens > 0 && cfg.SeqLen > 0 {
		batchSize = cfg.Training.BatchTokens / cfg.SeqLen
		if batchSize <= 0 {
			batchSize = 1
		}
	}

	var executionOrder []int
	if phaseOrder != nil {
		if len(phaseOrder) == 0 {
			return nil, fmt.Errorf("recurrence phase order must not be empty")
		}
		executionOrder = phaseOrder
	} else if !state.RecurrenceActive {
		order, err := uniqueRecurrenceExecutionOrder(cfg.Blocks, cfg.Recurrence)
		if err != nil {
			return nil, err
		}
		executionOrder = order
	}
	reserveHead := cfg.ReservesUntiedHeadWeight()
	useTiedHead := cfg.TieEmbeddings && !state.HeadUntied
	activeMTP := mtp
	if state.MTPAuxInactive {
		activeMTP = nil
	}
	objective := normalizeTrainingObjective(state.Objective)
	if objective == ObjectiveCausal && strings.TrimSpace(state.Objective) == "" {
		objective = cfg.Training.DefaultConcreteObjective()
	}
	distillation := cfg.Training.Distillation
	if state.DistillationInactive || !cfg.Training.DistillationActiveForConcreteObjective(objective) {
		distillation = nil
	}
	var data2vec *Data2VecSpec
	if cfg.Training.Data2VecActive() && !state.Data2VecInactive && state.HiddenCaptureTopK == 0 {
		data2vec = cfg.Training.Data2Vec
	}
	invariance := cfg.Training.Invariance
	if state.InvarianceInactive || !cfg.Training.InvarianceActive() {
		invariance = nil
	}
	pllMargin := cfg.Training.PLLMargin
	if state.PLLMarginInactive || !cfg.Training.PLLMarginActive() {
		pllMargin = nil
	}
	zLoss := cfg.Training.ZLoss
	if state.ZLossInactive {
		zLoss = 0
	}
	hiddenDropout := cfg.EffectiveHiddenDropout()
	attnDropout := cfg.EffectiveAttnDropout()
	embeddingDropout := cfg.EffectiveEmbeddingDropout()
	if state.DropoutInactive {
		hiddenDropout = 0
		attnDropout = 0
		embeddingDropout = 0
	}
	segmentAttentionMask := cfg.Training.AttentionSegmentMaskEnabled() && !state.SegmentMaskInactive
	framedCausalLoss := cfg.Training.ExampleFramingEnabled() && !state.ExampleFramingInactive
	wordStructural := cfg.Training.WordStructuralObjective
	if state.HiddenCaptureTopK > 0 {
		wordStructural = nil
	}

	return buildIRProgramWithDropoutNgramsOrderAndSmear(
		cfg.ModelDim,
		cfg.VocabSize,
		cfg.SeqLen,
		batchSize,
		cfg.EffectiveMLPMult(),
		cfg.TieEmbeddings,
		cfg.BlockScales,
		cfg.ResidMix,
		cfg.UNet,
		cfg.ParallelResidual,
		cfg.CharVocabSize,
		cfg.EffectiveCharDim(),
		cfg.EffectiveCharMaxPerToken(),
		cfg.BigramVocabSize,
		cfg.EffectiveBigramDim(),
		cfg.TrigramVocabSize,
		cfg.EffectiveTrigramDim(),
		cfg.LogitSoftcap,
		hiddenDropout,
		attnDropout,
		cfg.Blocks,
		cfg.Recurrence,
		executionOrder,
		activeMTP,
		reserveHead,
		useTiedHead,
		objective,
		cfg.Training.EffectiveObjective(),
		cfg.EffectiveMLMHead(),
		cfg.EffectiveLayerAggregation(),
		firstByteMask,
		zLoss,
		cfg.EffectivePositionalEmbedding(),
		cfg.EffectiveMaxPositions(),
		embeddingDropout,
		cfg.smearEmbeddingOptions(),
		cfg.Backout,
		distillation,
		data2vec,
		newData2VecHiddenCapture(state.HiddenCaptureTopK, len(cfg.Blocks), state.HiddenCapturePrefix),
		segmentAttentionMask,
		framedCausalLoss,
		cfg.EffectiveNormSpec(),
		cfg.EffectiveNormPlacement(),
		cfg.FFNInternalNorm,
		wordStructural,
		invariance,
		pllMargin,
	)
}

// BuildData2VecTeacherIRProgramFromConfig constructs a no-gradient teacher
// program that exposes top-K block hidden states for data2vec targets.
func BuildData2VecTeacherIRProgramFromConfig(cfg *ArchConfig, objective string) (*Program, error) {
	if cfg == nil {
		return nil, fmt.Errorf("nil config")
	}
	if !cfg.Training.Data2VecActive() {
		return nil, fmt.Errorf("training.data2vec is not active")
	}
	return buildIRProgramFromConfigWithState(cfg, TrainingProgramState{
		RecurrenceActive:     true,
		HeadUntied:           cfg.MTPUntieEnabled(),
		MTPAuxInactive:       true,
		DistillationInactive: true,
		Data2VecInactive:     true,
		InvarianceInactive:   true,
		PLLMarginInactive:    true,
		ZLossInactive:        true,
		DropoutInactive:      true,
		Objective:            objective,
		HiddenCaptureTopK:    cfg.Training.Data2Vec.TopKLayers,
		HiddenCapturePrefix:  "data2vec",
	}, nil, false)
}

// CountIRWeightsFromConfig returns the expected number of IR weight tensors.
func CountIRWeightsFromConfig(cfg *ArchConfig) (int, error) {
	if cfg == nil {
		return 0, fmt.Errorf("nil config")
	}

	metas, err := CollectWeightShapesFromConfig(cfg)
	if err != nil {
		return 0, err
	}
	return len(metas), nil
}

func convertBlockSpecs(specs []BlockSpec) []BlockSpec {
	if len(specs) == 0 {
		return nil
	}
	out := make([]BlockSpec, len(specs))
	copy(out, specs)
	return out
}
