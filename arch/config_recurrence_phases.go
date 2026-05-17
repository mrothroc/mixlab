package arch

import (
	"bytes"
	"encoding/json"
	"fmt"
)

// RecurrencePhaseSpec defines one training phase with an explicit block
// execution order. Weight sharing remains controlled by ArchConfig.Recurrence.
type RecurrencePhaseSpec struct {
	Frac  float64 `json:"frac"`
	Order []int   `json:"order"`
}

// RecurrencePhaseActivationSpec is reserved for the alternative prefix-length
// schema. It is not implemented; configs using it are rejected clearly.
type RecurrencePhaseActivationSpec struct {
	Frac               float64 `json:"frac"`
	ExecutionPrefixLen int     `json:"execution_prefix_len"`
}

// UnmarshalJSON decodes ArchConfig while recording which recurrence-phase
// fields were present in the source JSON. The unexported *Set bools let
// validation distinguish an absent field from a zero-value one.
func (c *ArchConfig) UnmarshalJSON(data []byte) error {
	type archConfigAlias ArchConfig
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}

	var alias archConfigAlias
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&alias); err != nil {
		return err
	}
	*c = ArchConfig(alias)
	_, c.recurrencePhasesSet = raw["recurrence_phases"]
	_, c.executionOrderSet = raw["execution_order"]
	_, c.recurrencePhaseActivationsSet = raw["recurrence_phase_activations"]
	return nil
}

func validateRecurrencePhases(cfg *ArchConfig, source string) error {
	if cfg == nil {
		return nil
	}
	schema2Set := cfg.executionOrderSet || cfg.recurrencePhaseActivationsSet
	if !cfg.recurrencePhasesSet {
		if schema2Set {
			return fmt.Errorf("config %q recurrence_phase_activations/execution_order schema is not implemented; use recurrence_phases", source)
		}
		return nil
	}
	if len(cfg.RecurrencePhases) == 0 {
		return fmt.Errorf("config %q recurrence_phases must contain at least one phase", source)
	}
	if schema2Set {
		return fmt.Errorf("config %q cannot set recurrence_phases with execution_order or recurrence_phase_activations", source)
	}
	if cfg.Training.RecurrenceActivationFrac > 0 || cfg.Training.RecurrenceActivationStep > 0 {
		return fmt.Errorf("config %q cannot set recurrence_phases with training.recurrence_activation_frac or training.recurrence_activation_step", source)
	}
	if cfg.UNet {
		return fmt.Errorf("config %q recurrence_phases is not supported with unet", source)
	}
	totalSteps := cfg.Training.TotalSteps()
	if totalSteps <= 0 {
		return fmt.Errorf("config %q recurrence_phases requires positive total training steps", source)
	}
	refs, err := normalizeWeightRefs(cfg.Blocks, cfg.Recurrence)
	if err != nil {
		return fmt.Errorf("config %q recurrence_phases: %w", source, err)
	}
	plan, err := newParallelResidualPlan(cfg.Blocks, cfg.ParallelResidual)
	if err != nil {
		return fmt.Errorf("config %q recurrence_phases: %w", source, err)
	}

	prevFrac := -1.0
	prevStep := -1
	for phaseIdx, phase := range cfg.RecurrencePhases {
		if phase.Frac != phase.Frac || phase.Frac < 0 || phase.Frac >= 1 {
			return fmt.Errorf("config %q recurrence_phases[%d].frac=%g must be in [0,1)", source, phaseIdx, phase.Frac)
		}
		if phaseIdx == 0 {
			if phase.Frac != 0 {
				return fmt.Errorf("config %q recurrence_phases[0].frac=%g must be 0.0", source, phase.Frac)
			}
		} else if phase.Frac <= prevFrac {
			return fmt.Errorf("config %q recurrence_phases[%d].frac=%g must be greater than previous frac=%g", source, phaseIdx, phase.Frac, prevFrac)
		}
		startStep := int(phase.Frac * float64(totalSteps))
		if phaseIdx > 0 && startStep <= prevStep {
			return fmt.Errorf("config %q recurrence_phases[%d] starts at step %d, must be after previous phase start step %d", source, phaseIdx, startStep, prevStep)
		}
		if len(phase.Order) == 0 {
			return fmt.Errorf("config %q recurrence_phases[%d].order must not be empty", source, phaseIdx)
		}
		if err := validateRecurrencePhaseOrder(cfg, refs, plan, phaseIdx, phase.Order, source); err != nil {
			return err
		}
		prevFrac = phase.Frac
		prevStep = startStep
	}
	return nil
}

func validateRecurrencePhaseOrder(cfg *ArchConfig, refs []int, plan parallelResidualPlan, phaseIdx int, order []int, source string) error {
	seen := make(map[int]int, len(order))
	for pos, idx := range order {
		if idx < 0 || idx >= len(cfg.Blocks) {
			return fmt.Errorf("config %q recurrence_phases[%d].order[%d]=%d out of range [0,%d)", source, phaseIdx, pos, idx, len(cfg.Blocks))
		}
		if prev, ok := seen[idx]; ok {
			return fmt.Errorf("config %q recurrence_phases[%d].order repeats block position %d at entries %d and %d", source, phaseIdx, idx, prev, pos)
		}
		root := refs[idx]
		if root != idx {
			if _, ok := seen[root]; !ok {
				return fmt.Errorf("config %q recurrence_phases[%d].order[%d]=%d reuses weights from root block %d, which must appear earlier in the same phase", source, phaseIdx, pos, idx, root)
			}
		}
		block := cfg.Blocks[idx]
		if blockTypeKey(block) == "plain" && block.KVSource > 0 {
			src := block.KVSource - 1
			if _, ok := seen[src]; !ok {
				return fmt.Errorf("config %q recurrence_phases[%d].order[%d]=%d has kv_source=%d, whose source block %d must appear earlier in the same phase", source, phaseIdx, pos, idx, block.KVSource, src)
			}
		}
		if plan.startsAt(idx) {
			if pos+1 >= len(order) || order[pos+1] != idx+1 {
				return fmt.Errorf("config %q recurrence_phases[%d].order must keep parallel_residual pair [%d,%d] contiguous", source, phaseIdx, idx, idx+1)
			}
		}
		if plan.secondAt(idx) {
			if pos == 0 || order[pos-1] != idx-1 {
				return fmt.Errorf("config %q recurrence_phases[%d].order must keep parallel_residual pair [%d,%d] contiguous", source, phaseIdx, idx-1, idx)
			}
		}
		seen[idx] = pos
	}
	if cfg.Backout != nil {
		if _, ok := seen[cfg.Backout.SaveLayer]; !ok {
			return fmt.Errorf("config %q recurrence_phases[%d].order must include backout.save_layer=%d", source, phaseIdx, cfg.Backout.SaveLayer)
		}
	}
	return nil
}
