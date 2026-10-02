package arch

import (
	"fmt"
	"strings"
)

// validatePhaseBaseLR rejects phase schedules whose explicit group or per-weight
// rates would be scaled by an implicit base. Every group's rate is multiplied by
// phase_lr / training.lr, so with training.lr omitted an override silently runs
// at override * phase_lr / default_lr. Inherited rates follow the phase LR
// exactly, and an explicit zero base keeps the legacy no-scaling policy, so
// neither needs an explicit base.
func validatePhaseBaseLR(cfg *ArchConfig, source string) error {
	t := cfg.Training
	if len(t.Phases) == 0 || t.lrSet {
		return nil
	}
	var overrides []string
	for _, group := range []struct {
		name  string
		set   bool
		value float32
	}{
		{"training.embed_lr", t.embedLRSet, t.EmbedLR},
		{"training.matrix_lr", t.matrixLRSet, t.MatrixLR},
		{"training.scalar_lr", t.scalarLRSet, t.ScalarLR},
		{"training.head_lr", t.headLRSet, t.HeadLR},
	} {
		if group.set && group.value != 0 {
			overrides = append(overrides, group.name)
		}
	}
	for i, b := range cfg.Blocks {
		if b.StateLR != nil && *b.StateLR != 0 {
			overrides = append(overrides, fmt.Sprintf("blocks[%d].state_lr", i))
		}
		if EffectiveS4DSobolevTrainable(b) && effectiveS4DSobolevLearningRate(b) != 0 {
			overrides = append(overrides, fmt.Sprintf("blocks[%d].sobolev_filter learning rate", i))
		}
	}
	if len(overrides) == 0 {
		return nil
	}
	return fmt.Errorf("config %q sets training.phases with %s but no explicit training.lr; "+
		"each group's rate is scaled by phase_lr / training.lr, so set training.lr "+
		"(usually the first phase's lr, %g, to apply these rates as written in the first phase)",
		source, strings.Join(overrides, ", "), t.Phases[0].LR)
}
