package train

import (
	"encoding/json"
	"fmt"
	"math"
	"testing"
)

func TestScheduleMinLRFractionFromConfig(t *testing.T) {
	for _, tc := range []struct {
		name, field string
		fraction    float32
		configured  bool
	}{
		{"omitted", "", 0.1, false},
		{"zero", `,"min_lr_fraction":0`, 0, true},
		{"small", `,"min_lr_fraction":0.001`, 0.001, true},
		{"ten_percent", `,"min_lr_fraction":0.1`, 0.1, true},
	} {
		for _, warmdown := range []int{0, 60} {
			t.Run(fmt.Sprintf("%s/warmdown%d", tc.name, warmdown), func(t *testing.T) {
				cfg, err := ParseArchConfig([]byte(fmt.Sprintf(`{
					"model_dim":16,"vocab_size":32,"seq_len":4,
					"blocks":[{"type":"plain","heads":2}],
					"training":{"lr":0.0001,"steps":300,"warmup_steps":0,"hold_steps":0,
					"warmdown_steps":%d%s}}`, warmdown, tc.field)), "floor")
				if err != nil {
					t.Fatal(err)
				}
				scheduler, steps := buildTrainingScheduler(cfg.Training)
				s := scheduler.(LRSchedule)
				wantMin := float32(0.0001) * tc.fraction
				if s.MinLR != wantMin || s.ClampWarmdownToMin != tc.configured || steps != 300 {
					t.Fatalf("schedule=%+v, want MinLR=%g", s, wantMin)
				}
				wantEnd := wantMin
				if warmdown > 0 && !tc.configured {
					wantEnd /= 10
				}
				for step := 0; step <= 310; step++ {
					progress := math.Min(float64(step)/300, 1)
					want := float64(wantMin) + float64(s.BaseLR-wantMin)*0.5*(1+math.Cos(math.Pi*progress))
					if warmdown > 0 && step >= 240 {
						start := float64(wantMin) + float64(s.BaseLR-wantMin)*0.5*(1+math.Cos(math.Pi*0.8))
						want = start + (float64(wantEnd)-start)*math.Min(float64(step-240)/60, 1)
					}
					if got := s.At(step); math.Abs(float64(got)-want) > 2e-11 {
						t.Fatalf("step %d: LR=%g want %g", step, got, want)
					}
				}
				if tc.name == "zero" && s.At(300) != 0 {
					t.Fatalf("explicit zero endpoint=%g", s.At(300))
				}
			})
		}
	}
}

func TestMinLRFractionResumeAndConfigRoundTrip(t *testing.T) {
	for _, phase := range []bool{false, true} {
		for _, floor := range []string{"", `,"min_lr_fraction":0`, `,"min_lr_fraction":0.1`} {
			t.Run(fmt.Sprintf("phases%v/%s", phase, floor), func(t *testing.T) {
				input := `{"steps":1000,"lr":0.001,"warmup_steps":25,"hold_steps":10,"warmdown_steps":100` + floor
				if phase {
					input += `,"phases":[{"steps":1000,"lr":0.001}]`
				} else {
					input += `,"lr_schedule_steps":1200`
				}
				input += `}`
				var spec TrainingSpec
				if err := json.Unmarshal([]byte(input), &spec); err != nil {
					t.Fatal(err)
				}
				sched, total := buildTrainingScheduler(spec)
				blob, err := json.Marshal(spec)
				if err != nil {
					t.Fatal(err)
				}
				var parsed TrainingSpec
				if err := json.Unmarshal(blob, &parsed); err != nil {
					t.Fatal(err)
				}
				roundTrip, _ := buildTrainingScheduler(parsed)
				saved, err := resumeScheduleFrom(spec, sched, total)
				if err != nil {
					t.Fatal(err)
				}
				blob, err = json.Marshal(saved)
				if err != nil {
					t.Fatal(err)
				}
				var loaded resumeSchedule
				if err := json.Unmarshal(blob, &loaded); err != nil {
					t.Fatal(err)
				}
				resumed, _, err := schedulerForResume(loaded, total)
				if err != nil {
					t.Fatal(err)
				}
				for step := 0; step <= total; step++ {
					if sched.At(step) != roundTrip.At(step) || sched.At(step) != resumed.At(step) {
						t.Fatalf("step %d: original=%g roundtrip=%g resumed=%g", step, sched.At(step), roundTrip.At(step), resumed.At(step))
					}
				}
				if phase && floor == `,"min_lr_fraction":0` {
					want := float32(0.001) * (1 - float32(99)/100)
					if math.Abs(float64(sched.At(999)-want)) > 1e-10 {
						t.Fatalf("phase warmdown must approach zero: last LR=%g want %g", sched.At(999), want)
					}
				}
			})
		}
	}
}

func TestMinLRFractionLegacyResume(t *testing.T) {
	// Older manifests have no presence flag. Replay their original schedules,
	// including the legacy 1% phase warmdown and 10% cosine floor.
	for _, input := range []string{
		`{"kind":"phases","original_total_steps":1000,"phases":[{"steps":1000,"lr":0.001}],"warmdown_steps":100}`,
		`{"kind":"cosine","original_total_steps":1000,"standard":{"base_lr":0.001,"min_lr":0.0001,"max_steps":1000}}`,
	} {
		var saved resumeSchedule
		if err := json.Unmarshal([]byte(input), &saved); err != nil {
			t.Fatal(err)
		}
		s, _, err := schedulerForResume(saved, 1000)
		if err != nil {
			t.Fatal(err)
		}
		var want trainingScheduler = LRSchedule{BaseLR: 0.001, MinLR: 0.0001, MaxSteps: 1000}
		if saved.Kind == "phases" {
			want = newPhaseSchedule(saved.Phases, 100, 0)
		}
		for step := 0; step <= 1010; step++ {
			if s.At(step) != want.At(step) {
				t.Fatalf("%s step %d: LR=%g want %g", saved.Kind, step, s.At(step), want.At(step))
			}
		}
	}
}
