package train

import (
	"encoding/json"
	"math"
	"reflect"
	"testing"

	"github.com/mrothroc/mixlab/arch"
)

func TestNewBobMatchesSpeechBrainDocstringFixture(t *testing.T) {
	scheduler := newNewBobSchedule(1, &arch.NewBobSpec{
		AnnealingFactor: 0.5, ImprovementThreshold: 0.0025,
		Patient: 0, Metric: arch.NewBobMetricValLoss,
	})
	metrics := []float64{10, 2, 2.5}
	wantLR := []float32{1, 1, 0.5}
	for i, metric := range metrics {
		observation, err := scheduler.Observe(metric)
		if err != nil {
			t.Fatal(err)
		}
		if observation.NewLR != wantLR[i] || scheduler.At(i) != wantLR[i] {
			t.Fatalf("observation %d metric=%g lr=%g At=%g want=%g", i, metric, observation.NewLR, scheduler.At(i), wantLR[i])
		}
	}
}

func TestNewBobPatientDelaysExactlyNNonImprovingObservations(t *testing.T) {
	scheduler := newNewBobSchedule(1, &arch.NewBobSpec{
		AnnealingFactor: 0.5, ImprovementThreshold: 0.1,
		Patient: 2, Metric: arch.NewBobMetricValLoss,
	})
	metrics := []float64{1, 1, 1, 1}
	wantLR := []float32{1, 1, 1, 0.5}
	wantPatient := []int{2, 1, 0, 2}
	for i, metric := range metrics {
		observation, err := scheduler.Observe(metric)
		if err != nil {
			t.Fatal(err)
		}
		if observation.NewLR != wantLR[i] || observation.PatientAfter != wantPatient[i] {
			t.Fatalf("observation %d=%+v want lr=%g patient=%d", i, observation, wantLR[i], wantPatient[i])
		}
	}
}

func TestNewBobFirstObservationAndZeroPreviousMetric(t *testing.T) {
	scheduler := newNewBobSchedule(2, &arch.NewBobSpec{
		AnnealingFactor: 0.9, ImprovementThreshold: 0.0025,
		Patient: 0, Metric: arch.NewBobMetricValErrorRate,
	})
	first, err := scheduler.Observe(0)
	if err != nil {
		t.Fatal(err)
	}
	if first.Annealed || first.HavePrevious || first.NewLR != 2 {
		t.Fatalf("first observation=%+v", first)
	}
	second, err := scheduler.Observe(0)
	if err != nil {
		t.Fatal(err)
	}
	if !second.Annealed || second.Improvement != 0 || math.Abs(float64(second.NewLR-1.8)) > 1e-6 {
		t.Fatalf("zero-previous observation=%+v", second)
	}
}

func TestNewBobScheduleResumeRoundTripContinuesState(t *testing.T) {
	spec := TrainingSpec{
		Steps: 100, LR: 1, LRSchedule: arch.LRScheduleNewBob,
		NewBob: &arch.NewBobSpec{
			AnnealingFactor: 0.5, ImprovementThreshold: 0.1,
			Patient: 1, Metric: arch.NewBobMetricValLoss,
		},
	}
	scheduler, total := buildTrainingScheduler(spec)
	metricScheduler := scheduler.(metricTrainingScheduler)
	for _, metric := range []float64{1, 1} {
		if _, err := metricScheduler.Observe(metric); err != nil {
			t.Fatal(err)
		}
	}
	saved, err := resumeScheduleFrom(spec, scheduler, total)
	if err != nil {
		t.Fatal(err)
	}
	blob, err := json.Marshal(saved)
	if err != nil {
		t.Fatal(err)
	}
	var decoded resumeSchedule
	if err := json.Unmarshal(blob, &decoded); err != nil {
		t.Fatal(err)
	}
	restored, restoredTotal, err := schedulerForResume(decoded, 120)
	if err != nil {
		t.Fatal(err)
	}
	if restoredTotal != 120 || restored.At(50) != scheduler.At(50) {
		t.Fatalf("restored total/lr=%d/%g want=120/%g", restoredTotal, restored.At(50), scheduler.At(50))
	}
	beforeState := scheduler.(*newBobSchedule).snapshot()
	afterState := restored.(*newBobSchedule).snapshot()
	if !reflect.DeepEqual(afterState, beforeState) {
		t.Fatalf("restored state=%+v want=%+v", afterState, beforeState)
	}
	want, err := metricScheduler.Observe(1)
	if err != nil {
		t.Fatal(err)
	}
	got, err := restored.(metricTrainingScheduler).Observe(1)
	if err != nil {
		t.Fatal(err)
	}
	if got.NewLR != want.NewLR || got.PatientAfter != want.PatientAfter || got.Annealed != want.Annealed {
		t.Fatalf("continued observation=%+v want=%+v", got, want)
	}
}

func TestNewBobClassificationMetricSelection(t *testing.T) {
	metrics := ClassificationMetrics{Loss: 0.75, Accuracy: 0.8}
	loss, err := newBobClassificationMetric(&arch.NewBobSpec{Metric: arch.NewBobMetricValLoss}, metrics)
	if err != nil || loss != 0.75 {
		t.Fatalf("loss metric=%g err=%v", loss, err)
	}
	errorRate, err := newBobClassificationMetric(&arch.NewBobSpec{Metric: arch.NewBobMetricValErrorRate}, metrics)
	if err != nil || math.Abs(errorRate-0.2) > 1e-12 {
		t.Fatalf("error-rate metric=%g err=%v", errorRate, err)
	}
}
