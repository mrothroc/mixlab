package train

import "testing"

type distributedScheduleProbe bool

func (p distributedScheduleProbe) DistributedContextActiveGPU() bool {
	return bool(p)
}

func TestDistributedCollectiveOrderStable(t *testing.T) {
	transitions := []string{
		"steady",
		"phase",
		"objective",
		"program",
		"error",
	}
	for _, transition := range transitions {
		t.Run(transition, func(t *testing.T) {
			if trainerAllowsStepLookahead(distributedScheduleProbe(true), true) {
				t.Fatalf("distributed transition %q allowed submit-before-collect", transition)
			}
		})
	}
	if !trainerAllowsStepLookahead(distributedScheduleProbe(false), true) {
		t.Fatal("single-process trainer lost configured step lookahead")
	}
	if trainerAllowsStepLookahead(distributedScheduleProbe(false), false) {
		t.Fatal("disabled single-process step lookahead was enabled")
	}
}
