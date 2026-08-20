package train

import (
	"fmt"
	"math"

	"github.com/mrothroc/mixlab/arch"
)

type metricTrainingScheduler interface {
	trainingScheduler
	Observe(metric float64) (newBobObservation, error)
}

type newBobObservation struct {
	Metric         float64
	PreviousMetric float64
	HavePrevious   bool
	Improvement    float64
	OldLR          float32
	NewLR          float32
	Annealed       bool
	PatientBefore  int
	PatientAfter   int
}

type newBobScheduleState struct {
	InitialLR            float32 `json:"initial_lr"`
	CurrentLR            float32 `json:"current_lr"`
	AnnealingFactor      float64 `json:"annealing_factor"`
	ImprovementThreshold float64 `json:"improvement_threshold"`
	Patient              int     `json:"patient"`
	CurrentPatient       int     `json:"current_patient"`
	Metric               string  `json:"metric"`
	PreviousMetric       float64 `json:"previous_metric,omitempty"`
	HavePrevious         bool    `json:"have_previous,omitempty"`
	Observations         int     `json:"observations"`
}

type newBobSchedule struct {
	state newBobScheduleState
}

func newNewBobSchedule(initialLR float32, spec *arch.NewBobSpec) *newBobSchedule {
	state := newBobScheduleState{InitialLR: initialLR, CurrentLR: initialLR}
	if spec != nil {
		state.AnnealingFactor = spec.AnnealingFactor
		state.ImprovementThreshold = spec.ImprovementThreshold
		state.Patient = spec.Patient
		state.CurrentPatient = spec.Patient
		state.Metric = spec.Metric
	}
	return &newBobSchedule{state: state}
}

func newNewBobScheduleFromState(state newBobScheduleState) (*newBobSchedule, error) {
	if err := validateNewBobScheduleState(state); err != nil {
		return nil, err
	}
	return &newBobSchedule{state: state}, nil
}

func (s *newBobSchedule) At(_ int) float32 {
	if s == nil {
		return 0
	}
	return s.state.CurrentLR
}

func (s *newBobSchedule) Observe(metric float64) (newBobObservation, error) {
	if s == nil {
		return newBobObservation{}, fmt.Errorf("nil NewBob scheduler")
	}
	if math.IsNaN(metric) || math.IsInf(metric, 0) || metric < 0 {
		return newBobObservation{}, fmt.Errorf("NewBob metric must be finite and >= 0, got %g", metric)
	}
	result := newBobObservation{
		Metric:        metric,
		OldLR:         s.state.CurrentLR,
		NewLR:         s.state.CurrentLR,
		PatientBefore: s.state.CurrentPatient,
		PatientAfter:  s.state.CurrentPatient,
	}
	if s.state.HavePrevious {
		result.HavePrevious = true
		result.PreviousMetric = s.state.PreviousMetric
		if s.state.PreviousMetric == 0 {
			result.Improvement = 0
		} else {
			result.Improvement = (s.state.PreviousMetric - metric) / s.state.PreviousMetric
		}
		if result.Improvement < s.state.ImprovementThreshold {
			if s.state.CurrentPatient == 0 {
				s.state.CurrentLR *= float32(s.state.AnnealingFactor)
				s.state.CurrentPatient = s.state.Patient
				result.Annealed = true
			} else {
				s.state.CurrentPatient--
			}
		}
	}
	s.state.PreviousMetric = metric
	s.state.HavePrevious = true
	s.state.Observations++
	result.NewLR = s.state.CurrentLR
	result.PatientAfter = s.state.CurrentPatient
	return result, nil
}

func (s *newBobSchedule) snapshot() newBobScheduleState {
	if s == nil {
		return newBobScheduleState{}
	}
	return s.state
}

func validateNewBobScheduleState(state newBobScheduleState) error {
	if state.InitialLR <= 0 || math.IsNaN(float64(state.InitialLR)) || math.IsInf(float64(state.InitialLR), 0) {
		return fmt.Errorf("checkpoint NewBob initial_lr=%g must be finite and > 0", state.InitialLR)
	}
	if state.CurrentLR <= 0 || math.IsNaN(float64(state.CurrentLR)) || math.IsInf(float64(state.CurrentLR), 0) || state.CurrentLR > state.InitialLR {
		return fmt.Errorf("checkpoint NewBob current_lr=%g must be finite, > 0, and <= initial_lr=%g", state.CurrentLR, state.InitialLR)
	}
	if state.AnnealingFactor <= 0 || state.AnnealingFactor > 1 || math.IsNaN(state.AnnealingFactor) || math.IsInf(state.AnnealingFactor, 0) {
		return fmt.Errorf("checkpoint NewBob annealing_factor=%g must be finite and in (0,1]", state.AnnealingFactor)
	}
	if state.ImprovementThreshold < 0 || math.IsNaN(state.ImprovementThreshold) || math.IsInf(state.ImprovementThreshold, 0) {
		return fmt.Errorf("checkpoint NewBob improvement_threshold=%g must be finite and >= 0", state.ImprovementThreshold)
	}
	if state.Patient < 0 || state.CurrentPatient < 0 || state.CurrentPatient > state.Patient {
		return fmt.Errorf("checkpoint NewBob patient state current=%d configured=%d is invalid", state.CurrentPatient, state.Patient)
	}
	if state.Observations < 0 || (state.HavePrevious && state.Observations == 0) || (!state.HavePrevious && state.Observations != 0) {
		return fmt.Errorf("checkpoint NewBob observation state count=%d have_previous=%t is invalid", state.Observations, state.HavePrevious)
	}
	if state.HavePrevious && (state.PreviousMetric < 0 || math.IsNaN(state.PreviousMetric) || math.IsInf(state.PreviousMetric, 0)) {
		return fmt.Errorf("checkpoint NewBob previous_metric=%g must be finite and >= 0", state.PreviousMetric)
	}
	switch state.Metric {
	case arch.NewBobMetricValLoss, arch.NewBobMetricValErrorRate:
	default:
		return fmt.Errorf("checkpoint NewBob metric=%q is unsupported", state.Metric)
	}
	return nil
}

func newBobClassificationMetric(spec *arch.NewBobSpec, metrics ClassificationMetrics) (float64, error) {
	if spec == nil {
		return 0, fmt.Errorf("NewBob metric selection requires training.newbob")
	}
	var value float64
	switch spec.Metric {
	case arch.NewBobMetricValLoss:
		value = metrics.Loss
	case arch.NewBobMetricValErrorRate:
		value = 1 - metrics.Accuracy
	default:
		return 0, fmt.Errorf("unsupported NewBob metric %q", spec.Metric)
	}
	if value < 0 || math.IsNaN(value) || math.IsInf(value, 0) {
		return 0, fmt.Errorf("NewBob metric %q produced invalid value %g", spec.Metric, value)
	}
	return value, nil
}
