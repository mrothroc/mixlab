package train

import (
	"fmt"
	"math"
	"strings"
)

// LRSchedule defines a cosine learning rate schedule with warmup and hold.
type LRSchedule struct {
	BaseLR             float32
	MinLR              float32
	Warmup             int
	Hold               int
	Warmdown           int
	MaxSteps           int
	ClampWarmdownToMin bool
}

type trainingScheduler interface {
	At(step int) float32
}

type phaseSchedule struct {
	lrs        []float32
	phaseIndex []int
	phases     []TrainingPhase
}

type trainingScheduleOptions struct {
	WarmupSteps    int
	WarmupStepsSet bool
	WarmupRatio    float64
	WarmupRatioSet bool
	HoldSteps      int
	HoldStepsSet   bool
}

// At returns the learning rate at the given step.
func (s LRSchedule) At(step int) float32 {
	baseAt := func(step int) float32 {
		if step < s.Warmup {
			if s.Warmup == 0 {
				return s.BaseLR
			}
			return s.BaseLR * float32(step) / float32(s.Warmup)
		}
		if step < s.Warmup+s.Hold {
			return s.BaseLR
		}
		decaySteps := s.MaxSteps - s.Warmup - s.Hold
		if decaySteps <= 0 {
			return s.BaseLR
		}
		progress := float64(step-s.Warmup-s.Hold) / float64(decaySteps)
		if progress > 1.0 {
			progress = 1.0
		}
		cosine := 0.5 * (1.0 + math.Cos(math.Pi*progress))
		return s.MinLR + (s.BaseLR-s.MinLR)*float32(cosine)
	}

	lr := baseAt(step)
	if s.Warmdown <= 0 || s.MaxSteps <= 0 {
		return lr
	}
	warmdownStart := s.MaxSteps - s.Warmdown
	if warmdownStart < 0 {
		warmdownStart = 0
	}
	if step < warmdownStart {
		return lr
	}
	startLR := baseAt(warmdownStart)
	targetLR := s.MinLR / 10
	if s.ClampWarmdownToMin && targetLR < s.MinLR {
		targetLR = s.MinLR
	}
	progress := float32(step-warmdownStart) / float32(s.Warmdown)
	if progress > 1 {
		progress = 1
	}
	return startLR + (targetLR-startLR)*progress
}

// At returns the per-step LR for a phase-based schedule.
func (s phaseSchedule) At(step int) float32 {
	if len(s.lrs) == 0 {
		return 0
	}
	if step < 0 {
		step = 0
	}
	if step >= len(s.lrs) {
		step = len(s.lrs) - 1
	}
	return s.lrs[step]
}

func (s phaseSchedule) PhaseAt(step int) TrainingPhase {
	if len(s.phases) == 0 {
		return TrainingPhase{}
	}
	if step < 0 {
		step = 0
	}
	if step >= len(s.phaseIndex) {
		step = len(s.phaseIndex) - 1
	}
	idx := s.phaseIndex[step]
	if idx < 0 || idx >= len(s.phases) {
		return TrainingPhase{}
	}
	return s.phases[idx]
}

// trainingSchedule constructs the standard LR schedule from base LR and total steps.
func trainingSchedule(lr float32, steps, warmdown int, minLRFraction float32) LRSchedule {
	return trainingScheduleWithOptions(lr, steps, warmdown, minLRFraction, trainingScheduleOptions{})
}

func trainingScheduleWithOptions(lr float32, steps, warmdown int, minLRFraction float32, opts trainingScheduleOptions) LRSchedule {
	if warmdown < 0 {
		warmdown = 0
	}
	if warmdown > steps {
		warmdown = steps
	}
	warmup := 100
	if opts.WarmupRatioSet {
		warmup = int(math.Round(float64(steps) * opts.WarmupRatio))
	} else if opts.WarmupStepsSet {
		warmup = opts.WarmupSteps
	}
	if warmup < 0 {
		warmup = 0
	}
	if steps < warmup {
		warmup = steps
	}
	hold := 200
	if opts.HoldStepsSet {
		hold = opts.HoldSteps
	}
	if hold < 0 {
		hold = 0
	}
	if steps < warmup+hold {
		hold = steps - warmup
		if hold < 0 {
			hold = 0
		}
	}
	minLR := lr * 0.1
	if minLRFraction > 0 {
		minLR = lr * minLRFraction
	}
	return LRSchedule{
		BaseLR:             lr,
		MinLR:              minLR,
		Warmup:             warmup,
		Hold:               hold,
		Warmdown:           warmdown,
		MaxSteps:           steps,
		ClampWarmdownToMin: minLRFraction > 0,
	}
}

func newPhaseSchedule(phases []TrainingPhase, warmdown int, minLRFraction float32) phaseSchedule {
	totalSteps := 0
	for _, phase := range phases {
		totalSteps += phase.Steps
	}
	sched := phaseSchedule{
		lrs:        make([]float32, totalSteps),
		phaseIndex: make([]int, totalSteps),
		phases:     append([]TrainingPhase(nil), phases...),
	}
	offset := 0
	for phaseIdx, phase := range phases {
		lr := float32(phase.LR)
		for i := 0; i < phase.Steps; i++ {
			sched.lrs[offset+i] = lr
			sched.phaseIndex[offset+i] = phaseIdx
		}
		offset += phase.Steps
	}
	if totalSteps == 0 || warmdown <= 0 || len(phases) == 0 {
		return sched
	}
	lastPhase := phases[len(phases)-1]
	lastPhaseWarmdown := warmdown
	if lastPhaseWarmdown > lastPhase.Steps {
		lastPhaseWarmdown = lastPhase.Steps
	}
	if lastPhaseWarmdown <= 0 {
		return sched
	}
	warmdownStart := totalSteps - lastPhaseWarmdown
	startLR := sched.lrs[warmdownStart]
	targetLR := float32(lastPhase.LR) * 0.01
	if minLRFraction > 0 {
		minLR := float32(lastPhase.LR) * minLRFraction
		if targetLR < minLR {
			targetLR = minLR
		}
	}
	for step := warmdownStart; step < totalSteps; step++ {
		progress := float32(step-warmdownStart) / float32(lastPhaseWarmdown)
		if progress > 1 {
			progress = 1
		}
		sched.lrs[step] = startLR + (targetLR-startLR)*progress
	}
	return sched
}

func buildTrainingScheduler(spec TrainingSpec) (trainingScheduler, int) {
	if len(spec.Phases) > 0 {
		totalSteps := spec.TotalSteps()
		return newPhaseSchedule(spec.Phases, spec.WarmdownSteps, spec.MinLRFraction), totalSteps
	}
	opts := trainingScheduleOptions{
		WarmupSteps:    spec.WarmupSteps,
		WarmupStepsSet: spec.WarmupStepsConfigured(),
		WarmupRatio:    spec.WarmupRatio,
		WarmupRatioSet: spec.WarmupRatioConfigured(),
		HoldSteps:      spec.HoldSteps,
		HoldStepsSet:   spec.HoldStepsConfigured(),
	}
	return trainingScheduleWithOptions(float32(spec.LR), spec.Steps, spec.WarmdownSteps, spec.MinLRFraction, opts), spec.Steps
}

func phaseDisplayLabel(phase TrainingPhase, index int) string {
	if strings.TrimSpace(phase.Label) != "" {
		return phase.Label
	}
	return fmt.Sprintf("phase-%d", index+1)
}
