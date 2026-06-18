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
	if len(cfg.RecurrencePhases) > 0 {
		return buildIRProgramFromConfigWithStateAndOrder(cfg, TrainingProgramState{
			RecurrenceActive:     true,
			HeadUntied:           cfg.MTPUntieEnabled(),
			Objective:            ObjectiveCausal,
			DistillationInactive: true,
			Data2VecInactive:     true,
			ZLossInactive:        true,
			DropoutInactive:      true,
		}, nil, cfg.RecurrencePhases[len(cfg.RecurrencePhases)-1].Order, false)
	}
	return buildIRProgramFromConfigWithState(cfg, TrainingProgramState{
		RecurrenceActive:     true,
		HeadUntied:           cfg.MTPUntieEnabled(),
		Objective:            ObjectiveCausal,
		DistillationInactive: true,
		Data2VecInactive:     true,
		ZLossInactive:        true,
		DropoutInactive:      true,
	}, nil, false)
}

// TrainingProgramState selects training-time graph schedules that can switch
// without changing the weight layout.
type TrainingProgramState struct {
	RecurrenceActive     bool
	HeadUntied           bool
	MTPAuxInactive       bool
	DistillationInactive bool
	Data2VecInactive     bool
	ZLossInactive        bool
	DropoutInactive      bool
	Objective            string
	HiddenCaptureTopK    int
	HiddenCapturePrefix  string
}

// BuildTrainingIRProgramFromConfig constructs a training program with MTP
// auxiliary losses enabled when configured unless state disables them.
func BuildTrainingIRProgramFromConfig(cfg *ArchConfig, state TrainingProgramState) (*Program, error) {
	if cfg == nil {
		return nil, fmt.Errorf("nil config")
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
	distillation := cfg.Training.Distillation
	if state.DistillationInactive {
		distillation = nil
	}
	var data2vec *Data2VecSpec
	if cfg.Training.Data2VecActive() && !state.Data2VecInactive && state.HiddenCaptureTopK == 0 {
		data2vec = cfg.Training.Data2Vec
	}
	zLoss := cfg.Training.ZLoss
	if state.ZLossInactive {
		zLoss = 0
	}
	objective := normalizeTrainingObjective(state.Objective)
	if objective == ObjectiveCausal && strings.TrimSpace(state.Objective) == "" {
		objective = cfg.Training.DefaultConcreteObjective()
	}
	hiddenDropout := cfg.EffectiveHiddenDropout()
	attnDropout := cfg.EffectiveAttnDropout()
	if state.DropoutInactive {
		hiddenDropout = 0
		attnDropout = 0
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
		firstByteMask,
		zLoss,
		cfg.smearEmbeddingOptions(),
		cfg.Backout,
		distillation,
		data2vec,
		newData2VecHiddenCapture(state.HiddenCaptureTopK, len(cfg.Blocks), state.HiddenCapturePrefix),
		cfg.EffectiveNormSpec(),
		cfg.EffectiveNormPlacement(),
		cfg.FFNInternalNorm,
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
