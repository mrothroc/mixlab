package diloco_test

import (
	"encoding/json"
	"github.com/mrothroc/mixlab/diloco/authenticated"
	"github.com/mrothroc/mixlab/diloco/learner"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/workerjob"
	"math"
	"testing"
)

func TestRecoveryContractMatrix(t *testing.T) {
	p, c, a, s, u := fixture(t)
	marker := learner.Completion{Format: "mixlab_learner_completion_v1", AssignmentHash: digest(t, a), UpdateID: u.ID, PrivateHash: digest(t, s), UpdateHash: digest(t, u)}
	marker.SHA256, _ = marker.Digest()
	if e := marker.ValidateFor(s, u); e != nil {
		t.Fatal(e)
	}
	bad := marker
	bad.PrivateHash = hash
	bad.SHA256, _ = bad.Digest()
	if bad.ValidateFor(s, u) == nil {
		t.Fatal("accepted partial/mismatched completion")
	}
	retry := run.AttemptBinding{Format: "mixlab_learner_attempt_v1", Binding: c.Binding, WorkerID: "w0", PreviousAttemptID: "attempt0", AttemptID: "attempt1", WorkloadGeneration: 2, Allocation: p.Slots[0].Allocation, NodeAuthorizationHash: hash, PrivateHash: u.PrivateHash, PrecisionHash: u.PrecisionHash}
	if e := retry.ValidateFor(c, "attempt0", 1, u.PrivateHash); e != nil {
		t.Fatal(e)
	}
	for _, change := range []func(*run.AttemptBinding){func(v *run.AttemptBinding) { v.WorkloadGeneration = 1 }, func(v *run.AttemptBinding) { v.PrivateHash = hash }, func(v *run.AttemptBinding) { v.Allocation = p.Slots[1].Allocation }, func(v *run.AttemptBinding) { v.PrecisionHash = c.Registrations[1].PrecisionHash }, func(v *run.AttemptBinding) { v.PreviousAttemptID = "stale" }} {
		v := retry
		change(&v)
		if v.ValidateFor(c, "attempt0", 1, u.PrivateHash) == nil {
			t.Fatal("accepted unfenced/replaced recovery")
		}
	}
	first := learner.Submission{Format: "mixlab_update_submission_v1", Update: u, PrincipalID: "workload0", AttemptID: "attempt0", WorkloadGeneration: 1, Allocation: p.Slots[0].Allocation}
	next := first
	next.AttemptID = retry.AttemptID
	next.WorkloadGeneration = 2
	if e := next.Validate(); e != nil {
		t.Fatal(e)
	}
	if digest(t, first.Update) != digest(t, next.Update) {
		t.Fatal("retry changed semantic bytes")
	}
	round := run.Round{Format: "mixlab_distributed_round_v1", Binding: c.Binding, Contract: p.Contract, Model: p.BaseModel, OuterState: a.OuterState, Outer: p.Outer, EffectiveTokens: 160, CreatedAt: 1500}
	for _, reg := range c.Registrations {
		round.Accepted = append(round.Accepted, run.AcceptedUpdate{WorkerID: reg.Slot.Member.WorkerID, UpdateID: reg.Slot.Member.WorkerID + "-update", UpdateHash: hash, PrivateHash: hash, PrecisionHash: reg.PrecisionHash})
	}
	round.SHA256, _ = round.Digest()
	if e := round.ValidateFor(p, c); e != nil {
		t.Fatal(e)
	}
	round.Accepted = round.Accepted[:1]
	round.SHA256, _ = round.Digest()
	if round.ValidateFor(p, c) == nil {
		t.Fatal("partial cohort committed round")
	}
	for _, v := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		bad := u
		bad.DeltaL2 = v
		if bad.Validate() == nil {
			t.Fatal("nonfinite summary")
		}
		o := p.Outer
		o.Momentum = v
		if o.Validate() == nil {
			t.Fatal("nonfinite outer momentum")
		}
	}
}

func TestSignedEnvelopeBinding(t *testing.T) {
	p, c, _, _, _ := fixture(t)
	signed := authenticated.Commit{Body: c, Proof: trust.SignedProof{Request: trust.SignRequest{Version: trust.ProofVersion, Purpose: trust.RunCommit, Digest: c.Binding.CommitHash, Context: c.Binding.RunID, Audience: p.Coordinator.Audience}, Evidence: trust.TrustEvidence{Cluster: p.Coordinator.ClusterID, Principal: p.Coordinator.PrincipalID, Role: trust.Coordinator}, Signature: make([]byte, 64)}}
	if e := signed.Validate(); e != nil {
		t.Fatal(e)
	}
	for _, mutate := range []func(*authenticated.Commit){func(v *authenticated.Commit) { v.Proof.Request.Purpose = trust.RunPlan }, func(v *authenticated.Commit) { v.Proof.Request.Audience = "other" }, func(v *authenticated.Commit) { v.Proof.Request.Digest = hash }, func(v *authenticated.Commit) { v.Proof.Evidence.Role = trust.Node }, func(v *authenticated.Commit) { v.Proof.Signature = nil }} {
		v := signed
		mutate(&v)
		if v.Validate() == nil {
			t.Fatal("accepted wrong proof binding")
		}
	}
}

func TestLocalLearnerAssignmentBinding(t *testing.T) {
	p, _, _, _, _ := fixture(t)
	a := workerjob.LearnerAssignment{Version: workerjob.LearnerVersion, JobID: "job0", AttemptID: "attempt0", RunID: "run0", PlanHash: p.SHA256, Member: p.Slots[0].Member, Contract: p.Contract, Coordinator: p.Coordinator, Config: json.RawMessage(`{"model_dim":16}`), DatasetSelector: "train", TrainPattern: "/tmp/train.bin", RuntimeSeconds: 100, PrivateMaxBytes: 1024}
	a.Contract.ConfigHash = digest(t, a.Config)
	b, e := a.Binding()
	if e != nil {
		t.Fatal(e)
	}
	raw, e := json.Marshal(a)
	if e != nil {
		t.Fatal(e)
	}
	if _, e = workerjob.DecodeLearner(raw, b); e != nil {
		t.Fatal(e)
	}
	b.AttemptID = "other"
	if _, e = workerjob.DecodeLearner(raw, b); e == nil {
		t.Fatal("accepted changed hosting binding")
	}
	b, e = a.Binding()
	if e != nil {
		t.Fatal(e)
	}
	if _, e = workerjob.Decode(raw, b); e == nil {
		t.Fatal("existing DDP decoder accepted new learner schema")
	}
}
