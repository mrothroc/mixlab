package train

import (
	"fmt"
	"strings"

	"github.com/mrothroc/mixlab/gpu"
)

// Shared by the runtime and diagnostics, including the legacy zero-base policy.
func scheduledLRScale(base, scheduled float32) float32 {
	if base > 0 {
		return scheduled / base
	}
	return 1
}

func effectiveGroupRates(spec gpu.TrainerOptimizerSpec, scheduled float32) string {
	active := make([]bool, len(spec.Groups))
	for _, w := range spec.Weights {
		if !w.Frozen && w.GroupIndex >= 0 && w.GroupIndex < len(active) {
			active[w.GroupIndex] = true
		}
	}
	scale := scheduledLRScale(spec.DefaultBaseLR, scheduled)
	var rates []string
	for i, g := range spec.Groups {
		if active[i] {
			name := g.ReportName
			if name == "" {
				name = fmt.Sprintf("group_%d", i)
			}
			rates = append(rates, fmt.Sprintf("%s=%.6g", name, g.LR*scale))
		}
	}
	return fmt.Sprintf("effective optimizer rates (base_lr=%g scale=%.6g): %s", spec.DefaultBaseLR, scale, strings.Join(rates, " "))
}

func logPhaseRates(trainer GPUTrainer, sched trainingScheduler, step, startStep int, name string) {
	s, ok := sched.(phaseSchedule)
	if !ok || (step != startStep && s.phaseIndex[step] == s.phaseIndex[step-1]) {
		return
	}
	i := s.phaseIndex[step]
	p := s.phases[i]
	fmt.Printf("  [%s] entering %s (%d/%d) update=%d steps=%d scheduled_lr=%g\n",
		name, phaseDisplayLabel(p, i), i+1, len(s.phases), step+1, p.Steps, sched.At(step))
	if reporter, ok := trainer.(interface{ effectiveLearningRates(float32) string }); ok {
		fmt.Printf("  [%s] %s\n", name, reporter.effectiveLearningRates(sched.At(step)))
	}
}
