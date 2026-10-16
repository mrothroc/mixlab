package admission

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/distributed"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/workload"
)

func evidenceCheck(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}
func evidenceJSON(t *testing.T, v any) []byte {
	t.Helper()
	b, err := json.Marshal(v)
	evidenceCheck(t, err)
	return b
}

type evidenceCase struct {
	kind  string
	value any
	call  func(context.Context, *Inbox, string, string, func() (any, error)) (any, error)
}

func evidenceFixture(t *testing.T) (*Inbox, Submission, []evidenceCase) {
	t.Helper()
	i, q := inboxFixture(t)
	q.Specification.Coordinator.PrincipalID = strings.Repeat("e", 32)
	evidenceCheck(t, i.Accept(context.Background(), q))
	spec := q.Specification
	p := run.Plan{Format: run.PlanVersion, Contract: spec.Contract, Coordinator: spec.Coordinator, BaseModel: spec.BaseModel, Outer: spec.Outer, ArtifactStoreID: spec.ArtifactStoreID, Endpoint: spec.Endpoint, CreatedAt: spec.CreatedAt, ExpiresAt: q.Deadline}
	var members []distributed.LearnerMember
	for n, c := range q.Launch.Selection.Selected {
		m := distributed.LearnerMember{WorkerID: fmt.Sprintf("%032x", n+10), ParticipantID: c.Capabilities.Node, IslandID: "metal", DatasetPartitionID: fmt.Sprint(n)}
		members = append(members, m)
		a := run.Allocation{Provider: "node_agent", NodeID: c.Capabilities.Node, LeaseID: fmt.Sprintf("%032x", n+20), JobID: fmt.Sprintf("%032x", n+30)}
		var err error
		a.RefHash, err = run.Digest(a)
		evidenceCheck(t, err)
		h, err := run.Digest(c.Capabilities)
		evidenceCheck(t, err)
		p.Slots = append(p.Slots, run.Slot{Member: m, Allocation: a, CapabilityHash: h})
	}
	var err error
	p.Cohort, err = distributed.NewLearnerCohort(q.Launch.Run, 1, members)
	evidenceCheck(t, err)
	p.SHA256, err = p.Digest()
	evidenceCheck(t, err)
	m := nodejob.LearnerManifest{Version: nodejob.LearnerVersion, Controller: q.Launch.Controller, Attempt: q.Launch.Attempt, WorkloadGeneration: 1, Plan: p, Slot: p.Slots[0], Config: spec.Config, DatasetSelector: "toy", Limits: q.Launch.Selection.Requirements.Limits}
	mq, err := m.SigningRequest()
	evidenceCheck(t, err)
	// Evidence storage checks canonical envelopes and intent bindings; the
	// downstream node/authority owns cryptographic verification of these proofs.
	manifest := nodejob.SignedLearner{Manifest: m, Proof: trust.SignedProof{Request: mq, Signature: []byte("original-manifest-proof")}}
	scope, err := m.WorkloadScope(strings.Repeat("f", 32))
	evidenceCheck(t, err)
	_, key, err := ed25519.GenerateKey(rand.Reader)
	evidenceCheck(t, err)
	request, err := workload.NewRequest(scope, key)
	evidenceCheck(t, err)
	rq, err := request.SigningRequest()
	evidenceCheck(t, err)
	signed := workload.SignedRequest{Request: request, Proof: trust.SignedProof{Request: rq, Evidence: trust.TrustEvidence{Principal: scope.Node, Role: trust.Node}, Signature: []byte("original-node-proof")}}
	gq, err := signed.GrantRequest()
	evidenceCheck(t, err)
	grant := workload.Grant{Request: signed, Proof: trust.SignedProof{Request: gq, Signature: []byte("original-grant-proof")}}
	return i, q, []evidenceCase{
		{"manifest", manifest, func(ctx context.Context, i *Inbox, run, job string, build func() (any, error)) (any, error) {
			return i.Manifest(ctx, run, job, func() (nodejob.SignedLearner, error) {
				v, err := build()
				if err != nil {
					return nodejob.SignedLearner{}, err
				}
				return v.(nodejob.SignedLearner), nil
			})
		}},
		{"workload-grant", grant, func(ctx context.Context, i *Inbox, run, job string, build func() (any, error)) (any, error) {
			return i.WorkloadGrant(ctx, run, job, signed, func() (workload.Grant, error) {
				v, err := build()
				if err != nil {
					return workload.Grant{}, err
				}
				return v.(workload.Grant), nil
			})
		}},
	}
}

func TestEvidenceReplayAcrossReopenedInbox(t *testing.T) {
	i, q, cases := evidenceFixture(t)
	job := fmt.Sprintf("%032x", 30)
	ctx := context.Background()
	for _, tc := range cases {
		t.Run(tc.kind, func(t *testing.T) {
			calls := 0
			first, err := tc.call(ctx, i, q.Launch.Run, job, func() (any, error) { calls++; return tc.value, nil })
			evidenceCheck(t, err)
			want := evidenceJSON(t, tc.value)
			if calls != 1 || !bytes.Equal(evidenceJSON(t, first), want) {
				t.Fatal("initial evidence changed")
			}
			name := q.Launch.Run + "-" + job + "-" + tc.kind
			stored, err := i.path.ReadFileLimit(name, 2<<20)
			evidenceCheck(t, err)
			if !bytes.Equal(stored, want) {
				t.Fatal("evidence not durable before return")
			}
			for range 3 {
				reopened, err := OpenInbox(i.path)
				evidenceCheck(t, err)
				replay, err := tc.call(ctx, reopened, q.Launch.Run, job, func() (any, error) { calls++; return nil, errors.New("signer unavailable after restart") })
				evidenceCheck(t, err)
				if calls != 1 || !bytes.Equal(evidenceJSON(t, replay), want) {
					t.Fatal("replay rebuilt or changed evidence")
				}
				after, err := i.path.ReadFileLimit(name, 2<<20)
				evidenceCheck(t, err)
				if !bytes.Equal(stored, after) {
					t.Fatal("replay rewrote durable proof")
				}
			}
		})
	}
}

func TestEvidenceRejectsCorruptReplayWithoutRebuilding(t *testing.T) {
	i, q, cases := evidenceFixture(t)
	job := fmt.Sprintf("%032x", 30)
	for _, tc := range cases {
		t.Run(tc.kind, func(t *testing.T) {
			valid := evidenceJSON(t, tc.value)
			for name, corrupt := range map[string][]byte{
				"truncated": valid[:len(valid)/2], "null": []byte("null"),
				"whitespace": append(bytes.Clone(valid), '\n'),
				"unknown":    append([]byte(`{"extra":true,`), valid[1:]...),
				"trailing":   append(bytes.Clone(valid), []byte(`{}`)...),
				"oversize":   bytes.Repeat([]byte(" "), (2<<20)+1),
			} {
				t.Run(name, func(t *testing.T) {
					file := q.Launch.Run + "-" + job + "-" + tc.kind
					evidenceCheck(t, i.path.WriteFile(file, corrupt))
					reopened, err := OpenInbox(i.path)
					evidenceCheck(t, err)
					calls := 0
					_, err = tc.call(context.Background(), reopened, q.Launch.Run, job, func() (any, error) { calls++; return tc.value, nil })
					if err == nil || calls != 0 {
						t.Fatalf("corrupt evidence repaired: err=%v builds=%d", err, calls)
					}
					after, err := i.path.ReadFileLimit(file, 3<<20)
					evidenceCheck(t, err)
					if !bytes.Equal(after, corrupt) {
						t.Fatal("corrupt evidence overwritten")
					}
				})
			}
		})
	}
}

func TestEvidenceRejectsMismatchedIntent(t *testing.T) {
	i, q, cases := evidenceFixture(t)
	job := fmt.Sprintf("%032x", 30)
	ctx := context.Background()
	other := q
	other.Launch.Run = strings.Repeat("9", 32)
	evidenceCheck(t, i.Accept(ctx, other))
	for _, tc := range cases {
		t.Run(tc.kind, func(t *testing.T) {
			for _, target := range []struct{ run, job string }{{other.Launch.Run, job}, {q.Launch.Run, strings.Repeat("8", 32)}} {
				_, err := tc.call(ctx, i, target.run, target.job, func() (any, error) { return tc.value, nil })
				if err == nil {
					t.Fatal("mismatched run/job accepted")
				}
				_, err = i.path.ReadFileLimit(target.run+"-"+target.job+"-"+tc.kind, 2<<20)
				if !errors.Is(err, os.ErrNotExist) {
					t.Fatal("invalid builder result published", err)
				}
			}
			bad := tc.value
			switch v := bad.(type) {
			case nodejob.SignedLearner:
				v.Proof.Request.Digest = strings.Repeat("0", 64)
				bad = v
			case workload.Grant:
				v.Proof.Request.Audience = strings.Repeat("0", 32)
				bad = v
			}
			_, err := tc.call(ctx, i, q.Launch.Run, job, func() (any, error) { return bad, nil })
			if err == nil {
				t.Fatal("mismatched proof published")
			}
			file := q.Launch.Run + "-" + job + "-" + tc.kind
			_, err = i.path.ReadFileLimit(file, 2<<20)
			if !errors.Is(err, os.ErrNotExist) {
				t.Fatal("invalid proof persisted", err)
			}
			evidenceCheck(t, i.path.WriteFile(file, evidenceJSON(t, bad)))
			reopened, err := OpenInbox(i.path)
			evidenceCheck(t, err)
			_, err = tc.call(ctx, reopened, q.Launch.Run, job, func() (any, error) { t.Error("rebuilt mismatched persisted proof"); return tc.value, nil })
			if err == nil {
				t.Fatal("mismatched persisted proof accepted")
			}
		})
	}
}

func TestEvidenceGrantReplayRequiresExactNodeRequest(t *testing.T) {
	i, q, cases := evidenceFixture(t)
	g := cases[1].value.(workload.Grant)
	ctx := context.Background()
	job := g.Request.Request.Scope.Job
	_, err := i.WorkloadGrant(ctx, q.Launch.Run, job, g.Request, func() (workload.Grant, error) { return g, nil })
	evidenceCheck(t, err)
	for _, change := range []func(*workload.SignedRequest){
		func(r *workload.SignedRequest) { r.Request.Scope.Attempt = strings.Repeat("7", 32) },
		func(r *workload.SignedRequest) { r.Proof.Signature = []byte("refreshed-node-proof") },
		func(r *workload.SignedRequest) { r.Request.CSR = []byte("different-key") },
	} {
		r := g.Request
		change(&r)
		reopened, err := OpenInbox(i.path)
		evidenceCheck(t, err)
		_, err = reopened.WorkloadGrant(ctx, q.Launch.Run, job, r, func() (workload.Grant, error) { t.Error("rebuilt grant for changed request"); return g, nil })
		if err == nil {
			t.Fatal("grant replay accepted changed request")
		}
	}
}

func TestEvidenceRequiresAcceptedRunAndCanonicalJob(t *testing.T) {
	i, q, cases := evidenceFixture(t)
	for _, tc := range cases {
		for _, target := range []struct{ run, job string }{{strings.Repeat("9", 32), fmt.Sprintf("%032x", 30)}, {q.Launch.Run, "../job"}, {q.Launch.Run, strings.Repeat("A", 32)}, {q.Launch.Run, "short"}} {
			_, err := tc.call(context.Background(), i, target.run, target.job, func() (any, error) { t.Error("invalid target reached builder"); return tc.value, nil })
			if err == nil {
				t.Fatal("invalid evidence target accepted")
			}
		}
		job := fmt.Sprintf("%032x", 30)
		failure := errors.New("builder unavailable")
		_, err := tc.call(context.Background(), i, q.Launch.Run, job, func() (any, error) { return nil, failure })
		if !errors.Is(err, failure) {
			t.Fatal(err)
		}
		_, err = tc.call(context.Background(), i, q.Launch.Run, job, func() (any, error) { return tc.value, nil })
		evidenceCheck(t, err)
	}
}

func TestEvidenceSnapshotLeaseFieldsAreOwnedCopies(t *testing.T) {
	x := setup(t)
	ctx := context.Background()
	for range 2 {
		_, err := x.s.Step(ctx, x.p)
		evidenceCheck(t, err)
	}
	first, err := x.s.Snapshot(ctx)
	evidenceCheck(t, err)
	if len(first.Nodes) != 2 || !first.Nodes[0].Touched || first.Nodes[0].Lease == nil || first.Nodes[1].Touched {
		t.Fatal("snapshot omitted reservation progress")
	}
	want := evidenceJSON(t, first.Nodes)
	first.Nodes[0].Reserve.IdempotencyKey = "changed"
	first.Nodes[0].Lease.Version = 999
	first.Nodes[0].Finalized, first.Nodes[0].Released = true, true
	again, err := x.s.Snapshot(ctx)
	evidenceCheck(t, err)
	if !bytes.Equal(evidenceJSON(t, again.Nodes), want) {
		t.Fatal("snapshot changed durable lease/intent state")
	}
}
