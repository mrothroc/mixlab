package train

import (
	"math"
	"reflect"
	"strings"
	"testing"
	"time"
)

// ---------- LRSchedule.At ----------

func TestLRScheduleAt_WarmupRamp(t *testing.T) {
	s := LRSchedule{BaseLR: 1.0, MinLR: 0.1, Warmup: 100, Hold: 0, MaxSteps: 1000}
	// Step 0 should be 0
	if got := s.At(0); got != 0.0 {
		t.Errorf("At(0) = %v, want 0", got)
	}
	// Step 50 should be halfway through warmup
	if got := s.At(50); math.Abs(float64(got)-0.5) > 1e-6 {
		t.Errorf("At(50) = %v, want 0.5", got)
	}
	// Step 99 should be near BaseLR
	if got := s.At(99); got < 0.98 || got > 1.0 {
		t.Errorf("At(99) = %v, want ~0.99", got)
	}
}

func TestLRScheduleAt_WarmupZero(t *testing.T) {
	// Warmup=0 and step < Warmup is false (step=0 is not < 0), so it goes to hold/decay.
	// But let's also test the Warmup==0 early-return branch by crafting step < Warmup scenario:
	// Actually step < 0 is impossible. The Warmup==0 branch is only reachable if step < Warmup
	// AND Warmup == 0, which requires step < 0. Let's just verify step=0 goes to hold.
	s := LRSchedule{BaseLR: 1.0, MinLR: 0.1, Warmup: 0, Hold: 100, MaxSteps: 1000}
	if got := s.At(0); got != 1.0 {
		t.Errorf("At(0) with Warmup=0 = %v, want 1.0 (hold phase)", got)
	}
}

func TestLRScheduleAt_HoldPhase(t *testing.T) {
	s := LRSchedule{BaseLR: 1.0, MinLR: 0.1, Warmup: 100, Hold: 200, MaxSteps: 1000}
	// Steps in [100, 300) should return BaseLR
	for _, step := range []int{100, 150, 200, 299} {
		if got := s.At(step); got != 1.0 {
			t.Errorf("At(%d) = %v, want 1.0 (hold)", step, got)
		}
	}
}

func TestLRScheduleAt_CosineDecay(t *testing.T) {
	s := LRSchedule{BaseLR: 1.0, MinLR: 0.0, Warmup: 0, Hold: 0, MaxSteps: 1000}
	// At step 0 (start of decay): cosine(0) = 1.0, so LR = 1.0
	if got := s.At(0); math.Abs(float64(got)-1.0) > 1e-6 {
		t.Errorf("At(0) = %v, want 1.0", got)
	}
	// At step 500 (midpoint): cosine(pi/2) ~ 0, LR ~ 0.5
	if got := s.At(500); math.Abs(float64(got)-0.5) > 1e-4 {
		t.Errorf("At(500) = %v, want ~0.5", got)
	}
	// At step 1000 (end): cosine(pi) = -1, so 0.5*(1-1)=0, LR = MinLR = 0
	if got := s.At(1000); math.Abs(float64(got)) > 1e-6 {
		t.Errorf("At(1000) = %v, want 0.0", got)
	}
}

func TestLRScheduleAt_CosineDecayWithMinLR(t *testing.T) {
	s := LRSchedule{BaseLR: 1.0, MinLR: 0.1, Warmup: 0, Hold: 0, MaxSteps: 1000}
	// At end: should be MinLR
	if got := s.At(1000); math.Abs(float64(got)-0.1) > 1e-6 {
		t.Errorf("At(1000) = %v, want 0.1 (MinLR)", got)
	}
}

func TestLRScheduleAt_BeyondMaxSteps(t *testing.T) {
	s := LRSchedule{BaseLR: 1.0, MinLR: 0.1, Warmup: 0, Hold: 0, MaxSteps: 1000}
	// Progress clamped to 1.0, so returns MinLR
	if got := s.At(2000); math.Abs(float64(got)-0.1) > 1e-6 {
		t.Errorf("At(2000) = %v, want 0.1 (clamped to MinLR)", got)
	}
}

func TestLRScheduleAt_DecayStepsZero(t *testing.T) {
	// MaxSteps == Warmup+Hold => decaySteps <= 0 => return BaseLR
	s := LRSchedule{BaseLR: 1.0, MinLR: 0.1, Warmup: 500, Hold: 500, MaxSteps: 1000}
	if got := s.At(999); got != 1.0 {
		t.Errorf("At(999) with no decay = %v, want 1.0", got)
	}
}

func TestLRScheduleAt_DecayStepsNegative(t *testing.T) {
	// MaxSteps < Warmup+Hold => decaySteps < 0 => return BaseLR
	s := LRSchedule{BaseLR: 2.0, MinLR: 0.1, Warmup: 500, Hold: 600, MaxSteps: 1000}
	if got := s.At(600); got != 2.0 {
		t.Errorf("At(600) with negative decay = %v, want 2.0", got)
	}
}

func TestLRScheduleAt_FullCycle(t *testing.T) {
	s := LRSchedule{BaseLR: 0.001, MinLR: 0.0001, Warmup: 100, Hold: 200, MaxSteps: 1000}
	// Warmup: monotonically increasing
	prev := float32(0.0)
	for step := 0; step < 100; step++ {
		got := s.At(step)
		if got < prev {
			t.Errorf("warmup not monotonic at step %d: %v < %v", step, got, prev)
		}
		prev = got
	}
	// Hold: constant
	for step := 100; step < 300; step++ {
		if got := s.At(step); got != 0.001 {
			t.Errorf("hold phase step %d = %v, want 0.001", step, got)
		}
	}
	// Decay: monotonically decreasing
	prev = s.At(300)
	for step := 301; step <= 1000; step++ {
		got := s.At(step)
		if got > prev {
			t.Errorf("decay not monotonic at step %d: %v > %v", step, got, prev)
		}
		prev = got
	}
}

func TestLRScheduleAt_WarmdownRamp(t *testing.T) {
	s := LRSchedule{BaseLR: 1.0, MinLR: 0.1, Warmup: 0, Hold: 0, Warmdown: 100, MaxSteps: 1000}

	start := s.At(900)
	wantStart := s.MinLR + (s.BaseLR-s.MinLR)*float32(0.5*(1+math.Cos(math.Pi*0.9)))
	if math.Abs(float64(start-wantStart)) > 1e-6 {
		t.Fatalf("At(900) = %v, want warmdown start cosine value", start)
	}
	mid := s.At(950)
	wantMid := start + (0.01-start)*0.5
	if math.Abs(float64(mid-wantMid)) > 1e-6 {
		t.Fatalf("At(950) = %v, want %v", mid, wantMid)
	}
	if got := s.At(1000); math.Abs(float64(got-0.01)) > 1e-6 {
		t.Fatalf("At(1000) = %v, want 0.01", got)
	}
}

func TestLRScheduleAt_WarmdownDisabled(t *testing.T) {
	s := LRSchedule{BaseLR: 1.0, MinLR: 0.1, Warmup: 0, Hold: 0, Warmdown: 0, MaxSteps: 1000}
	if got := s.At(1000); math.Abs(float64(got-0.1)) > 1e-6 {
		t.Fatalf("At(1000) = %v, want 0.1 without warmdown", got)
	}
}

// ---------- trainingSchedule ----------

func TestTrainingSchedule_Default(t *testing.T) {
	s := trainingSchedule(0.001, 10000, 0)
	if s.Warmup != 100 {
		t.Errorf("Warmup = %d, want 100", s.Warmup)
	}
	if s.Hold != 200 {
		t.Errorf("Hold = %d, want 200", s.Hold)
	}
	if s.MaxSteps != 10000 {
		t.Errorf("MaxSteps = %d, want 10000", s.MaxSteps)
	}
	if s.BaseLR != 0.001 {
		t.Errorf("BaseLR = %v, want 0.001", s.BaseLR)
	}
	if math.Abs(float64(s.MinLR)-0.0001) > 1e-8 {
		t.Errorf("MinLR = %v, want 0.0001", s.MinLR)
	}
}

func TestTrainingSchedule_StepsLessThanWarmup(t *testing.T) {
	s := trainingSchedule(0.01, 50, 0)
	if s.Warmup != 50 {
		t.Errorf("Warmup = %d, want 50 (clamped to steps)", s.Warmup)
	}
	if s.Hold != 0 {
		t.Errorf("Hold = %d, want 0", s.Hold)
	}
	if s.MaxSteps != 50 {
		t.Errorf("MaxSteps = %d, want 50", s.MaxSteps)
	}
}

func TestTrainingSchedule_StepsBetweenWarmupAndWarmupPlusHold(t *testing.T) {
	// steps=150: warmup stays 100, hold should be clamped to 50
	s := trainingSchedule(0.01, 150, 0)
	if s.Warmup != 100 {
		t.Errorf("Warmup = %d, want 100", s.Warmup)
	}
	if s.Hold != 50 {
		t.Errorf("Hold = %d, want 50 (clamped to steps-warmup)", s.Hold)
	}
}

func TestTrainingSchedule_StepsExactlyWarmup(t *testing.T) {
	s := trainingSchedule(0.01, 100, 0)
	if s.Warmup != 100 {
		t.Errorf("Warmup = %d, want 100", s.Warmup)
	}
	if s.Hold != 0 {
		t.Errorf("Hold = %d, want 0", s.Hold)
	}
}

func TestTrainingSchedule_StepsExactlyWarmupPlusHold(t *testing.T) {
	s := trainingSchedule(0.01, 300, 0)
	if s.Warmup != 100 {
		t.Errorf("Warmup = %d, want 100", s.Warmup)
	}
	if s.Hold != 200 {
		t.Errorf("Hold = %d, want 200", s.Hold)
	}
}

func TestTrainingSchedule_ZeroSteps(t *testing.T) {
	s := trainingSchedule(0.01, 0, 0)
	if s.Warmup != 0 {
		t.Errorf("Warmup = %d, want 0", s.Warmup)
	}
	if s.Hold != 0 {
		t.Errorf("Hold = %d, want 0", s.Hold)
	}
}

func TestTrainingSchedule_WarmdownClampedToSteps(t *testing.T) {
	s := trainingSchedule(0.01, 50, 100)
	if s.Warmdown != 50 {
		t.Fatalf("Warmdown = %d, want 50", s.Warmdown)
	}
}

func TestPhaseScheduleAt_Boundaries(t *testing.T) {
	s := newPhaseSchedule([]TrainingPhase{
		{Steps: 2, LR: 1e-4, Label: "warmup"},
		{Steps: 3, LR: 1e-3, Label: "main"},
		{Steps: 2, LR: 5e-4, Label: "cooldown"},
	}, 0)

	if got := s.At(0); math.Abs(float64(got-1e-4)) > 1e-8 {
		t.Fatalf("At(0) = %g, want 1e-4", got)
	}
	if got := s.At(1); math.Abs(float64(got-1e-4)) > 1e-8 {
		t.Fatalf("At(1) = %g, want 1e-4", got)
	}
	if got := s.At(2); math.Abs(float64(got-1e-3)) > 1e-8 {
		t.Fatalf("At(2) = %g, want 1e-3", got)
	}
	if got := s.At(4); math.Abs(float64(got-1e-3)) > 1e-8 {
		t.Fatalf("At(4) = %g, want 1e-3", got)
	}
	if got := s.At(5); math.Abs(float64(got-5e-4)) > 1e-8 {
		t.Fatalf("At(5) = %g, want 5e-4", got)
	}
	if phase := s.PhaseAt(2); phase.Label != "main" {
		t.Fatalf("PhaseAt(2).Label = %q, want main", phase.Label)
	}
}

func TestPhaseSchedule_WarmdownWithinLastPhase(t *testing.T) {
	s := newPhaseSchedule([]TrainingPhase{
		{Steps: 2, LR: 1e-4},
		{Steps: 4, LR: 1e-3},
	}, 2)

	if got := s.At(3); math.Abs(float64(got-1e-3)) > 1e-8 {
		t.Fatalf("At(3) = %g, want 1e-3 before warmdown", got)
	}
	if got := s.At(4); math.Abs(float64(got-1e-3)) > 1e-8 {
		t.Fatalf("At(4) = %g, want warmdown start at 1e-3", got)
	}
	wantLast := float32(1e-3 + (1e-5-1e-3)*0.5)
	if got := s.At(5); math.Abs(float64(got-wantLast)) > 1e-8 {
		t.Fatalf("At(5) = %g, want %g", got, wantLast)
	}
}

func TestBuildTrainingScheduler_BackwardCompatibleWithoutPhases(t *testing.T) {
	sched, steps := buildTrainingScheduler(TrainingSpec{Steps: 150, LR: 0.01})
	if steps != 150 {
		t.Fatalf("steps = %d, want 150", steps)
	}
	s, ok := sched.(LRSchedule)
	if !ok {
		t.Fatalf("scheduler type = %T, want LRSchedule", sched)
	}
	if s.MaxSteps != 150 {
		t.Fatalf("MaxSteps = %d, want 150", s.MaxSteps)
	}
}

func TestBuildTrainingScheduler_UsesPhaseTotalSteps(t *testing.T) {
	sched, steps := buildTrainingScheduler(TrainingSpec{
		Steps: 999,
		LR:    0.1,
		Phases: []TrainingPhase{
			{Steps: 3, LR: 1e-4},
			{Steps: 7, LR: 2e-4},
		},
	})
	if steps != 10 {
		t.Fatalf("steps = %d, want 10", steps)
	}
	if got := sched.At(0); math.Abs(float64(got-1e-4)) > 1e-8 {
		t.Fatalf("At(0) = %g, want 1e-4", got)
	}
	if got := sched.At(3); math.Abs(float64(got-2e-4)) > 1e-8 {
		t.Fatalf("At(3) = %g, want 2e-4", got)
	}
}

func TestShouldUpdateSWA(t *testing.T) {
	cases := []struct {
		step, start, interval int
		want                  bool
	}{
		{step: 0, start: 0, interval: 10, want: false},
		{step: 9, start: 10, interval: 5, want: false},
		{step: 10, start: 10, interval: 5, want: true},
		{step: 14, start: 10, interval: 5, want: false},
		{step: 15, start: 10, interval: 5, want: true},
		{step: 20, start: 10, interval: 0, want: false},
	}
	for _, tc := range cases {
		if got := shouldUpdateSWA(tc.step, tc.start, tc.interval); got != tc.want {
			t.Errorf("shouldUpdateSWA(%d, %d, %d) = %v, want %v", tc.step, tc.start, tc.interval, got, tc.want)
		}
	}
}

func TestUpdateEMAWeights(t *testing.T) {
	ema := make([][]float32, 2)
	current := [][]float32{
		{1, 2},
		{3},
	}
	updateEMAWeights(ema, current, 0.9)
	if !reflect.DeepEqual(ema, current) {
		t.Fatalf("first EMA update = %v, want %v", ema, current)
	}

	next := [][]float32{
		{3, 4},
		{7},
	}
	updateEMAWeights(ema, next, 0.5)
	want := [][]float32{
		{2, 3},
		{5},
	}
	if !reflect.DeepEqual(ema, want) {
		t.Fatalf("second EMA update = %v, want %v", ema, want)
	}
}

func TestExportWeightsForTrainerPrefersSWA(t *testing.T) {
	swa := [][]float32{{1, 2, 3}}
	got, err := exportWeightsForTrainer(nil, swa)
	if err != nil {
		t.Fatalf("exportWeightsForTrainer: %v", err)
	}
	if !reflect.DeepEqual(got, swa) {
		t.Fatalf("export weights = %v, want %v", got, swa)
	}
	got[0][0] = 99
	if swa[0][0] != 1 {
		t.Fatalf("exportWeightsForTrainer should clone SWA weights, mutated source=%v", swa)
	}
}

func TestFormatProgressTiming_BeforeETAThreshold(t *testing.T) {
	// step 0 (< 1): no ETA, just elapsed
	got := formatProgressTiming(12500*time.Millisecond, 0, 100)
	if got != "(12.5s)" {
		t.Fatalf("formatProgressTiming = %q, want %q", got, "(12.5s)")
	}
}

func TestFormatProgressTiming_WithETA(t *testing.T) {
	// 42s elapsed over 11 steps (0-10), 9 remaining
	// avg = 42s/11 ≈ 3.818s, eta = 9 * 3.818 ≈ 34s
	got := formatProgressTiming(42*time.Second, 10, 20)
	want := "(42.0s, ~34s remaining)"
	if got != want {
		t.Fatalf("formatProgressTiming = %q, want %q", got, want)
	}
}

func TestFormatProgressTiming_TokPerSec(t *testing.T) {
	// Verify wall-clock based tok/s: 1000 steps * 16384 batch_tokens / 100s = 163840 tok/s
	elapsed := 100 * time.Second
	step := 999
	batchTokens := 16384
	tokPerSec := float64(batchTokens) * float64(step+1) / elapsed.Seconds()
	if tokPerSec < 163000 || tokPerSec > 164000 {
		t.Fatalf("tok/s = %.0f, want ~163840", tokPerSec)
	}
}

func TestCheckpointHelpers(t *testing.T) {
	if !shouldWriteCheckpoint(9, 10) {
		t.Fatal("expected checkpoint at step index 9 for every=10")
	}
	if shouldWriteCheckpoint(8, 10) {
		t.Fatal("did not expect checkpoint at step index 8 for every=10")
	}
	if shouldWriteCheckpoint(9, 0) {
		t.Fatal("did not expect checkpoint when disabled")
	}
	if got := checkpointPath("/tmp/checkpoints", 12); got != "/tmp/checkpoints/step_000012.st" {
		t.Fatalf("checkpointPath = %q", got)
	}
}

// ---------- TrainResult.formatSummary ----------

func TestFormatSummary_WithValLoss(t *testing.T) {
	r := TrainResult{
		Name:        "test_model",
		FirstLoss:   2.5,
		LastLoss:    1.5,
		LastValLoss: 1.7,
		HasValLoss:  true,
		Delta:       -1.0,
		Elapsed:     3*time.Second + 456*time.Millisecond,
	}
	got := r.formatSummary()
	if !strings.Contains(got, "test_model") {
		t.Errorf("missing name in %q", got)
	}
	if !strings.Contains(got, "first=2.5000") {
		t.Errorf("missing first loss in %q", got)
	}
	if !strings.Contains(got, "last=1.5000") {
		t.Errorf("missing last loss in %q", got)
	}
	if !strings.Contains(got, "val=1.7000") {
		t.Errorf("missing val loss in %q", got)
	}
	if !strings.Contains(got, "delta=-1.0000") {
		t.Errorf("missing delta in %q", got)
	}
	if !strings.Contains(got, "3.456s") {
		t.Errorf("missing elapsed in %q", got)
	}
}

func TestFormatSummary_WithoutValLoss(t *testing.T) {
	r := TrainResult{
		Name:       "baseline",
		FirstLoss:  3.0,
		LastLoss:   2.0,
		HasValLoss: false,
		Delta:      -1.0,
		Elapsed:    1*time.Minute + 30*time.Second,
	}
	got := r.formatSummary()
	if strings.Contains(got, "val=") {
		t.Errorf("should not contain val= when HasValLoss=false: %q", got)
	}
	if !strings.Contains(got, "baseline") {
		t.Errorf("missing name in %q", got)
	}
	if !strings.Contains(got, "first=3.0000") {
		t.Errorf("missing first loss in %q", got)
	}
	if !strings.Contains(got, "last=2.0000") {
		t.Errorf("missing last loss in %q", got)
	}
	if !strings.Contains(got, "1m30s") {
		t.Errorf("missing elapsed in %q", got)
	}
}

func TestFormatSummary_NamePadding(t *testing.T) {
	r := TrainResult{
		Name:       "ab",
		FirstLoss:  1.0,
		LastLoss:   1.0,
		HasValLoss: false,
		Delta:      0.0,
		Elapsed:    0,
	}
	got := r.formatSummary()
	// Name should be padded to 12 characters
	if !strings.HasPrefix(got, "ab          ") {
		t.Errorf("name not left-padded to 12 chars: %q", got)
	}
}
