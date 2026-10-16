package integration

import (
	"context"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/coordinator/learnerapi"
	private "github.com/mrothroc/mixlab/diloco/learner"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/trust"
)

func recoveryFenceRequests(t *testing.T, x *relayFixture, index int) []learnerapi.Request {
	t.Helper()
	a := x.s.Assignments[index]
	h, err := run.Digest(a)
	check(t, err)
	base := learnerapi.Request{Workload: x.bindings[index], Call: private.RelayRequest{Format: "mixlab_learner_relay_v1", Binding: a.Binding, WorkerID: a.WorkerID, Operation: "poll"}}
	start := base
	start.Call.Operation = "start"
	start.Call.Start = &private.Start{Format: "mixlab_learner_start_v1", Binding: a.Binding, WorkerID: a.WorkerID, AssignmentHash: h, PrivateHash: strings.Repeat("c", 64)}
	read := base
	read.Call.Operation = "artifact"
	read.Call.Artifact = &private.ArtifactRequest{Assignment: a, Blob: a.BaseModel, Operation: "read"}
	create := base
	create.Call.Operation = "artifact"
	create.Call.Artifact = &private.ArtifactRequest{Assignment: a, Blob: run.Blob{Kind: "delta_f32", Ref: artifactRef(x.body)}, Operation: "create"}
	return []learnerapi.Request{base, start, read, create}
}

func TestLearnerGatewayRecoveryFencesOldAttemptAndGrants(t *testing.T) {
	x := newLearnerRelay(t)
	ctx := context.Background()
	old := recoveryFenceRequests(t, x, 0)
	check(t, x.clients[0].Start(ctx, *old[1].Call.Start))
	runID := x.s.Plan.Cohort.RunID
	check(t, x.model.BeginLearnerRecovery(ctx, runID, x.bindings[0].Worker))
	s, err := x.model.Repository.Load(ctx, runID)
	check(t, err)
	intent := *s.Recovery
	for _, phase := range []string{"begin", "bound", "finished"} {
		t.Run(phase, func(t *testing.T) {
			for _, q := range old {
				if code := x.raw(t, x.identities[0], q); code != http.StatusForbidden {
					t.Fatalf("old %s accepted during %s: %d", q.Call.Operation, phase, code)
				}
			}
			for _, q := range recoveryFenceRequests(t, x, 1) {
				if code := x.raw(t, x.identities[1], q); code != http.StatusOK {
					t.Fatalf("unaffected %s rejected during %s: %d", q.Call.Operation, phase, code)
				}
			}
			if phase != "begin" {
				current := x.bindings[0]
				current.Attempt, current.Generation, current.Principal = intent.Attempt, current.Generation+1, id(t)
				identity := x.issue(t, current)
				for _, q := range old {
					q.Workload = current
					if code := x.raw(t, identity, q); code != http.StatusOK {
						t.Fatalf("new %s rejected during %s: %d", q.Call.Operation, phase, code)
					}
				}
			}
		})
		switch phase {
		case "begin":
			a := x.s.Assignments[0]
			check(t, x.model.BindAttempt(ctx, run.AttemptBinding{Format: "mixlab_learner_attempt_v1", Binding: a.Binding, WorkerID: a.WorkerID, PreviousAttemptID: intent.Previous, AttemptID: intent.Attempt, WorkloadGeneration: x.bindings[0].Generation + 1, Allocation: x.s.Plan.Slots[0].Allocation, NodeAuthorizationHash: strings.Repeat("b", 64), PrivateHash: intent.PrivateHash, PrecisionHash: a.PrecisionHash}, relayRecovery{}))
		case "bound":
			check(t, x.model.FinishLearnerRecovery(ctx, runID, intent.Attempt))
		}
	}
}

func TestLearnerGatewayWithholdsInflightArtifactGrantAfterBegin(t *testing.T) {
	x := newLearnerRelay(t)
	ctx := context.Background()
	requests := recoveryFenceRequests(t, x, 0)
	check(t, x.clients[0].Start(ctx, *requests[1].Call.Start))
	grants, err := learnerapi.OpenGrants(artifactPath(t))
	check(t, err)
	var begin sync.Once
	h, err := (learnerapi.Gateway{Grants: grants, Policy: x.g.Policy, Anchor: x.f.a, Trust: artifactTrust{x.f}, Coordinator: x.s.Plan.Coordinator, Model: x.model, Clock: x.f.clock, Timeout: 5 * time.Second, Sign: func(ctx context.Context, q trust.SignRequest) (trust.SignedProof, error) {
		// Authorization and grant construction already succeeded; Begin commits
		// before the gateway's final authorization check releases the response.
		var beginErr error
		begin.Do(func() { beginErr = x.model.BeginLearnerRecovery(ctx, x.s.Plan.Cohort.RunID, x.bindings[0].Worker) })
		if beginErr != nil {
			return trust.SignedProof{}, beginErr
		}
		return trust.SignPrincipalProof(x.f.a, x.f.server.Chain, x.f.server.Key, x.f.view, q, x.f.now)
	}}).Handler()
	check(t, err)
	x.url = serveCapabilities(t, x.g.Policy, h).URL
	if code := x.raw(t, x.identities[0], requests[2]); code != http.StatusForbidden {
		t.Fatal("inflight grant released after recovery began", code)
	}
	s, err := x.model.Repository.Load(ctx, x.s.Plan.Cohort.RunID)
	check(t, err)
	if s.Recovery == nil {
		t.Fatal("test did not reach signing/Begin interleaving")
	}
	for _, q := range recoveryFenceRequests(t, x, 1) {
		if code := x.raw(t, x.identities[1], q); code != http.StatusOK {
			t.Fatal("unaffected worker fenced", q.Call.Operation, code)
		}
	}
}
