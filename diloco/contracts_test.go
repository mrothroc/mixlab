package diloco_test

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/artifact"
	"github.com/mrothroc/mixlab/diloco/authenticated"
	"github.com/mrothroc/mixlab/diloco/learner"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/distributed"
	"github.com/mrothroc/mixlab/precision"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/workerjob"
)

const hash = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

func digest(t *testing.T, v any) string {
	t.Helper()
	h, e := run.Digest(v)
	if e != nil {
		t.Fatal(e)
	}
	return h
}
func blob(kind string) run.Blob {
	return run.Blob{Ref: artifact.Ref{SHA256: hash, Bytes: 64}, Kind: kind}
}

func fixture(t *testing.T) (run.Plan, run.Commit, run.Assignment, learner.PrivateState, learner.Update) {
	t.Helper()
	members := []distributed.LearnerMember{{WorkerID: "w0", ParticipantID: "node0", IslandID: "mac0", DatasetPartitionID: "p0"}, {WorkerID: "w1", ParticipantID: "node1", IslandID: "mac1", DatasetPartitionID: "p1"}}
	cohort, e := distributed.NewLearnerCohort("run0", 0, members)
	if e != nil {
		t.Fatal(e)
	}
	c := run.Contract{ConfigHash: hash, ProgramHash: hash, WeightLayoutHash: hash, InnerOptimizerHash: hash, DatasetHash: hash, BuildHash: hash}
	p := run.Plan{Format: run.PlanVersion, Cohort: cohort, Coordinator: run.Coordinator{ClusterID: "cluster", PrincipalID: "coordinator", Role: "coordinator", Audience: "coord0"}, Contract: c, BaseModel: blob("model_f32"), Outer: run.Outer{Kind: "nesterov", LR: 1, Momentum: 0, Aggregation: "uniform", InnerSteps: 5, TotalSteps: 20, Accumulation: 2}, ArtifactStoreID: "store", Endpoint: "https://coordinator.test:8443", CreatedAt: 1000, ExpiresAt: 2000}
	for i, m := range members {
		a := run.Allocation{Provider: "node_agent", NodeID: m.ParticipantID, LeaseID: m.WorkerID + "-lease", JobID: m.WorkerID + "-job"}
		a.RefHash = digest(t, a)
		p.Slots = append(p.Slots, run.Slot{Member: members[i], CapabilityHash: hash, Allocation: a})
	}
	p.SHA256, e = p.Digest()
	if e != nil {
		t.Fatal(e)
	}
	if e = p.Validate(); e != nil {
		t.Fatal(e)
	}
	commit := run.Commit{Format: run.CommitVersion, Binding: run.Binding{RunID: "run0", PlanHash: p.SHA256, Generation: 0, CohortHash: cohort.CohortHash}, Coordinator: p.Coordinator, CommittedAt: 1100}
	for i, s := range p.Slots {
		policy := precision.Policy{ComputeDType: "float32", Backend: "metal", TF32Allowed: i == 1}
		ph, e := policy.Digest()
		if e != nil {
			t.Fatal(e)
		}
		commit.Registrations = append(commit.Registrations, run.Registration{Format: run.RegistrationVersion, RunID: "run0", PlanHash: p.SHA256, Generation: 0, CohortHash: cohort.CohortHash, Slot: s, Contract: c, BaseModelHash: hash, PreparationHash: hash, JobManifestHash: hash, AttemptID: s.Member.WorkerID + "-attempt0", WorkloadGeneration: 1, Coordinator: p.Coordinator, Precision: policy, PrecisionHash: ph})
	}
	commit.Binding.CommitHash, e = commit.Digest()
	if e != nil {
		t.Fatal(e)
	}
	if e = commit.ValidateFor(p); e != nil {
		t.Fatal(e)
	}
	a := run.Assignment{Format: run.AssignmentVersion, ID: "assignment0", Binding: commit.Binding, Coordinator: p.Coordinator, WorkerID: "w0", Round: 0, BaseRound: 0, BaseModel: p.BaseModel, OuterState: blob("outer_state_f32"), Contract: c, PartitionID: "p0", InnerSteps: 5, PrecisionHash: commit.Registrations[0].PrecisionHash, IssuedAt: 1200, ExpiresAt: 1800}
	if e = a.ValidateFor(p, commit); e != nil {
		t.Fatal(e)
	}
	ref := blob("private_state").Ref
	state := learner.PrivateState{Format: "mixlab_learner_private_v1", Assignment: a, AssignmentHash: digest(t, a), UpdateID: "update0", Stage: "completed", Model: ref, Optimizer: ref, Sampler: ref, Schedule: ref, TrainingMetadata: ref, Replay: learner.Replay{Version: "mixlab_learner_rng_v1", RunID: "run0", WorkerID: "w0", PartitionID: "p0", Seed: 42, AttemptStep: 5}, Precision: commit.Registrations[0].Precision, Attempted: 5, Committed: 5, EffectiveTokens: 80, EffectiveExamples: 20}
	if e = state.Validate(); e != nil {
		t.Fatal(e)
	}
	u := learner.Update{Format: "mixlab_outer_update_v1", ID: "update0", AssignmentHash: digest(t, a), Binding: commit.Binding, BaseModelHash: hash, WorkerID: "w0", ParticipantID: "node0", IslandID: "mac0", Contract: c, Delta: blob("delta_f32"), DType: "float32", TensorCount: 8, PrecisionHash: a.PrecisionHash, PrivateHash: digest(t, state), Attempted: 5, Committed: 5, EffectiveTokens: 80, EffectiveExamples: 20, Loss: 2.5, DeltaL2: 1, DeltaMaxAbs: .1, StartedAt: 1300, FinishedAt: 1400}
	if e = u.ValidateFor(a); e != nil {
		t.Fatal(e)
	}
	return p, commit, a, state, u
}

func TestCanonicalContracts(t *testing.T) {
	p, c, a, s, u := fixture(t)
	completion := learner.Completion{Format: "mixlab_learner_completion_v1", AssignmentHash: digest(t, a), UpdateID: u.ID, PrivateHash: digest(t, s), UpdateHash: digest(t, u)}
	completion.SHA256, _ = completion.Digest()
	g := run.Grant{Format: "mixlab_artifact_access_grant_v1", ID: "grant0", Binding: c.Binding, Coordinator: p.Coordinator, PrincipalID: "workload0", ParticipantID: "node0", WorkerID: "w0", AssignmentID: a.ID, Operation: "read", Artifact: a.BaseModel, AccessClass: "global_base", IssuedAt: 1200, ExpiresAt: 1800, MaximumUses: 2, Nonce: "nonce0"}
	g.SHA256, _ = g.Digest()
	eval := run.EvaluationAssignment{Format: "mixlab_evaluation_assignment_v1", ID: "eval0", Binding: c.Binding, Coordinator: p.Coordinator, Model: p.BaseModel, ValidationHash: hash, Split: "validation", BatchTokens: 8, SequenceLength: 4, Batches: 2, WorkerID: "w0", Precision: s.Precision, PrecisionHash: a.PrecisionHash, IssuedAt: 1200, ExpiresAt: 1800}
	result := run.EvaluationResult{Format: "mixlab_evaluation_result_v1", Assignment: eval, AssignmentHash: digest(t, eval), Loss: 2, EffectiveTokens: 16, StartedAt: 1300, FinishedAt: 1400, PrivateBeforeHash: hash, PrivateAfterHash: hash}
	round := run.Round{Format: "mixlab_distributed_round_v1", Binding: c.Binding, Contract: p.Contract, Model: p.BaseModel, OuterState: a.OuterState, Outer: p.Outer, Accepted: []run.AcceptedUpdate{{WorkerID: "w0", UpdateID: "update0", UpdateHash: digest(t, u), PrivateHash: u.PrivateHash, PrecisionHash: u.PrecisionHash}}, EffectiveTokens: 80, CreatedAt: 1500}
	round.SHA256, _ = round.Digest()
	local := workerjob.LearnerAssignment{Version: workerjob.LearnerVersion, JobID: "job0", AttemptID: "attempt0", RunID: "run0", PlanHash: p.SHA256, Member: p.Slots[0].Member, Contract: p.Contract, Coordinator: p.Coordinator, Config: json.RawMessage(`{"model_dim":16}`), DatasetSelector: "train", TrainPattern: "/tmp/train.bin", RuntimeSeconds: 100, PrivateMaxBytes: 1024}
	local.Contract.ConfigHash = digest(t, local.Config)
	values := map[string]interface{ Validate() error }{
		"plan": &p, "registration": &c.Registrations[0], "commit": &c, "assignment": &a, "private": &s, "update": &u, "completion": &completion, "grant": &g, "evaluation-assignment": &eval, "evaluation-result": &result, "round": &round, "local-assignment": &local,
		"grant-use":  &run.GrantUse{Format: "mixlab_artifact_use_v1", GrantHash: g.SHA256, GrantID: g.ID, Operation: "read", IdempotencyID: "request0"},
		"attempt":    &run.AttemptBinding{Format: "mixlab_learner_attempt_v1", Binding: c.Binding, WorkerID: "w0", PreviousAttemptID: "attempt0", AttemptID: "attempt1", WorkloadGeneration: 2, Allocation: p.Slots[0].Allocation, NodeAuthorizationHash: hash, PrivateHash: u.PrivateHash, PrecisionHash: u.PrecisionHash},
		"submission": &learner.Submission{Format: "mixlab_update_submission_v1", Update: u, PrincipalID: "workload0", AttemptID: "attempt0", WorkloadGeneration: 1, Allocation: p.Slots[0].Allocation},
		"receipt":    &learner.Receipt{Format: "mixlab_update_receipt_v1", Binding: c.Binding, UpdateID: u.ID, UpdateHash: digest(t, u), PrivateHash: u.PrivateHash},
		"decision":   &run.EvaluationDecision{Format: "mixlab_evaluation_decision_v1", Binding: c.Binding, ResultHash: digest(t, result), Decision: "continue"},
	}
	// These fixtures exercise envelope structure only. Trust signature and
	// revocation verification is a mandatory adapter step in P4/P5.
	proof := func(purpose trust.Purpose, h string) trust.SignedProof {
		return trust.SignedProof{Request: trust.SignRequest{Version: trust.ProofVersion, Purpose: purpose, Digest: h, Context: "run0", Audience: p.Coordinator.Audience}, Evidence: trust.TrustEvidence{Cluster: p.Coordinator.ClusterID, Principal: p.Coordinator.PrincipalID, Role: trust.Coordinator}, Signature: make([]byte, 64)}
	}
	values["signed-plan"] = &authenticated.Plan{Body: p, Proof: proof(trust.RunPlan, p.SHA256)}
	values["signed-commit"] = &authenticated.Commit{Body: c, Proof: proof(trust.RunCommit, c.Binding.CommitHash)}
	values["signed-assignment"] = &authenticated.Assignment{Body: a, Proof: proof(trust.RoundGrant, digest(t, a))}
	values["signed-grant"] = &authenticated.Grant{Body: g, Proof: proof(trust.ArtifactGrant, g.SHA256)}
	for name, v := range values {
		t.Run(name, func(t *testing.T) {
			if e := v.Validate(); e != nil {
				t.Fatal(e)
			}
			b, e := json.Marshal(v)
			if e != nil {
				t.Fatal(e)
			}
			path := filepath.Join("testdata", name+".json")
			if os.Getenv("UPDATE_DILOCO_FIXTURES") == "1" {
				if e := os.MkdirAll("testdata", 0755); e != nil {
					t.Fatal(e)
				}
				if e := os.WriteFile(path, b, 0644); e != nil {
					t.Fatal(e)
				}
			}
			gold, e := os.ReadFile(path)
			if e != nil {
				t.Fatal(e)
			}
			if !bytes.Equal(gold, b) {
				t.Fatal("canonical fixture changed")
			}
			fresh := func() interface{ Validate() error } {
				return reflect.New(reflect.TypeOf(v).Elem()).Interface().(interface{ Validate() error })
			}
			if e := run.Decode(b, fresh()); e != nil {
				t.Fatal(e)
			}
			var object map[string]json.RawMessage
			if e := json.Unmarshal(b, &object); e != nil {
				t.Fatal(e)
			}
			key := "format"
			if name == "local-assignment" {
				key = "version"
			}
			if strings.HasPrefix(name, "signed-") {
				key = "body"
			}
			badVersion := bytes.Replace(b, object[key], []byte(`"future_v99"`), 1)
			unknown := append([]byte(`{"unknown":0,`), b[1:]...)
			duplicate := append([]byte(`{"`+key+`":`+string(object[key])+`,`), b[1:]...)
			missing := bytes.Replace(b, append([]byte(`"`+key+`":`), append(object[key], ',')...), nil, 1)
			for _, bad := range [][]byte{badVersion, unknown, duplicate, missing, []byte("null"), bytes.Repeat([]byte(" "), (1<<20)+1), append(append([]byte{}, b...), []byte(`{}`)...)} {
				if e := run.Decode(bad, fresh()); e == nil {
					t.Fatalf("accepted invalid wire %s", string(bad[:min(len(bad), 100)]))
				}
			}
			// Every declared top-level field is required, even when its value is
			// zero. In particular omission/null must not supply a zero round,
			// generation, momentum, or boolean policy implicitly.
			for field, raw := range object {
				if bytes.Equal(raw, []byte("null")) {
					continue
				}
				needle := append([]byte(`"`+field+`":`), raw...)
				nulled := bytes.Replace(b, needle, []byte(`"`+field+`":null`), 1)
				if e := run.Decode(nulled, fresh()); e == nil {
					t.Fatalf("accepted null %s", field)
				}
			}
		})
	}
}

func TestPlanAndPrecisionBindings(t *testing.T) {
	p, c, a, _, _ := fixture(t)
	// Unequal policies are intentional; exact per-worker provenance is mandatory.
	if c.Registrations[0].Precision == c.Registrations[1].Precision {
		t.Fatal("fixture needs different policies")
	}
	mutations := []func(*run.Registration){func(r *run.Registration) { r.PlanHash = hash }, func(r *run.Registration) { r.Slot.Member.WorkerID = "other" }, func(r *run.Registration) { r.Precision.Backend = "cuda" }, func(r *run.Registration) { r.Precision.TF32Allowed = !r.Precision.TF32Allowed }, func(r *run.Registration) { r.Contract.ProgramHash = strings.Repeat("b", 64) }, func(r *run.Registration) { r.Coordinator.Audience = "other" }, func(r *run.Registration) { r.Slot.Allocation.Provider = "external" }}
	for i, f := range mutations {
		r := c.Registrations[0]
		f(&r)
		if r.ValidateFor(p) == nil {
			t.Fatalf("accepted registration mutation %d", i)
		}
	}
	a.Binding.CommitHash = ""
	if a.Validate() == nil {
		t.Fatal("staged plan authorized a round")
	}
	c.CommittedAt = p.ExpiresAt
	c.Binding.CommitHash, _ = c.Digest()
	if c.ValidateFor(p) == nil {
		t.Fatal("expired admission committed")
	}
}

func TestPrivateReplayAndUpdateBindings(t *testing.T) {
	_, _, a, s, u := fixture(t)
	for _, mutate := range []func(*learner.Update){func(v *learner.Update) { v.Skipped = 1 }, func(v *learner.Update) { v.Attempted++ }, func(v *learner.Update) { v.Attempted++; v.Committed++ }, func(v *learner.Update) { v.Round++ }, func(v *learner.Update) { v.PrivateHash = "" }, func(v *learner.Update) { v.PrecisionHash = "" }, func(v *learner.Update) { v.EffectiveTokens = 0 }, func(v *learner.Update) { v.FinishedAt = a.ExpiresAt + 1 }} {
		v := u
		mutate(&v)
		if v.ValidateFor(a) == nil {
			t.Fatal("accepted invalid update")
		}
	}
	for _, mutate := range []func(*learner.PrivateState){func(v *learner.PrivateState) { v.Replay.AttemptStep++ }, func(v *learner.PrivateState) { v.Replay.Microstep = 1 }, func(v *learner.PrivateState) { v.Precision.TF32Allowed = true }, func(v *learner.PrivateState) { v.Sampler = artifact.Ref{} }, func(v *learner.PrivateState) { v.Stage = "start" }} {
		v := s
		mutate(&v)
		if v.Validate() == nil {
			t.Fatal("accepted invalid private state")
		}
	}
	start := s
	start.Stage = "start"
	start.Attempted = 0
	start.Committed = 0
	start.EffectiveTokens = 0
	start.EffectiveExamples = 0
	start.Replay.AttemptStep = 0
	if e := start.Validate(); e != nil {
		t.Fatal(e)
	}
	before, _ := start.Replay.Digest()
	encoded, _ := json.Marshal(start)
	var restored learner.PrivateState
	if e := learner.Decode(encoded, &restored); e != nil {
		t.Fatal(e)
	}
	after, _ := restored.Replay.Digest()
	if before != after {
		t.Fatal("replay changed across restore")
	}
	// Hosting attempts are intentionally outside all semantic/checkpoint records.
	for _, v := range []any{a, s, u} {
		b, _ := json.Marshal(v)
		if bytes.Contains(b, []byte("attempt_id")) || bytes.Contains(b, []byte("workload_generation")) {
			t.Fatal("replaceable process identity contaminates replay")
		}
	}
	other := start.Replay
	other.WorkerID = "w1"
	d, _ := other.Digest()
	if d == before {
		t.Fatal("workers share private stream")
	}
}
