package train

import (
	"fmt"
	"sort"

	"github.com/mrothroc/mixlab/arch"
)

// initialRecurrencePhase returns the starting phase index given whether the
// schedule is active. 0 when scheduled, -1 otherwise.
func initialRecurrencePhase(scheduled bool) int {
	if scheduled {
		return 0
	}
	return -1
}

// recurrencePhaseIndexForStep returns the index of the recurrence phase that
// should be active at the given training step. Phases are identified by their
// integer start step in ascending order. Returns -1 when starts is empty.
func recurrencePhaseIndexForStep(starts []int, step int) int {
	if len(starts) == 0 {
		return -1
	}
	idx := sort.Search(len(starts), func(i int) bool {
		return starts[i] > step
	}) - 1
	if idx < 0 {
		return 0
	}
	return idx
}

// logRecurrencePhasesSchedule prints the active phase-0 IR state and the
// scheduled transition steps for subsequent phases.
func logRecurrencePhasesSchedule(name string, cfg *arch.ArchConfig, starts []int, initialProg *arch.Program) {
	fmt.Printf("  [%s] recurrence phase 0 active at step 0: order_len=%d, IR program: %d ops, %d weights\n",
		name, len(cfg.RecurrencePhases[0].Order), len(initialProg.Ops), initialProg.NumWeights)
	for i := 1; i < len(starts); i++ {
		fmt.Printf("  [%s] recurrence phase %d scheduled at step %d: order_len=%d\n",
			name, i, starts[i], len(cfg.RecurrencePhases[i].Order))
	}
}

// logRecurrencePhaseTransition prints the per-step transition message when the
// active recurrence phase changes mid-training.
func logRecurrencePhaseTransition(name string, cfg *arch.ArchConfig, prev, next, step int) {
	prevLen := len(cfg.RecurrencePhases[prev].Order)
	nextLen := len(cfg.RecurrencePhases[next].Order)
	fmt.Printf("  [%s] recurrence phase %d activated at step %d: order_len=%d (was %d)\n",
		name, next, step, nextLen, prevLen)
}
