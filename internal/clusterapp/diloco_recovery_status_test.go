package clusterapp

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/coordinator"
)

func TestDiLoCoStatusReportsRecoveryAndTerminalFailure(t *testing.T) {
	s := coordinator.State{Phase: coordinator.Open, Round: 4, LearnerUnavailableAt: 100,
		Recovery: &coordinator.LearnerRecovery{WorkerID: "worker", Started: 105, PrivateHash: "private-state-hash"}}
	var out DiLoCoSubmissionStatus
	out.setRunState(s)
	if out.RecoveryWorker != "worker" || out.RecoveryStartedAt != 105 || out.LearnerUnavailableSince != 100 || out.Round != 4 {
		t.Fatal(out)
	}
	b, err := json.Marshal(out)
	if err != nil || strings.Contains(string(b), "private-state-hash") {
		t.Fatalf("status leaked recovery authority: %s %v", b, err)
	}
	s.Phase, s.Failure = coordinator.Failed, "learner recovery exhausted"
	out.setRunState(s)
	if out.Failure == "" || out.RunPhase != coordinator.Failed || out.RecoveryWorker != "" || out.LearnerUnavailableSince != 0 {
		t.Fatal(out)
	}
}
