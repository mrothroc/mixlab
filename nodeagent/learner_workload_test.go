package nodeagent

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/json"
	"errors"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/distributed"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/workload"
)

// This fixture installs an already-accepted journal; cryptographic admission is
// exercised by trust/integration. These tests isolate store lifecycle locking.
func learnerCredentialFixture(t *testing.T) (*Store, trust.AuthenticatedPrincipal, Reserve, string) {
	t.Helper()
	s, actor := leaseFixture(t)
	q := reservation("1")
	l, err := s.Reserve(context.Background(), actor, q, leaseNow)
	if err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile("../diloco/testdata/plan.json")
	if err != nil {
		t.Fatal(err)
	}
	var p run.Plan
	if err = json.Unmarshal(b, &p); err != nil {
		t.Fatal(err)
	}
	m := distributed.LearnerMember{WorkerID: strings.Repeat("e", 32), ParticipantID: s.node, IslandID: "metal", DatasetPartitionID: "p0"}
	p.Cohort, err = distributed.NewLearnerCohort(q.Run, 0, []distributed.LearnerMember{m})
	if err != nil {
		t.Fatal(err)
	}
	a := run.Allocation{Provider: "node_agent", NodeID: s.node, LeaseID: l.ID, JobID: strings.Repeat("f", 32)}
	a.RefHash, err = run.Digest(a)
	if err != nil {
		t.Fatal(err)
	}
	p.Slots = []run.Slot{{Member: m, CapabilityHash: strings.Repeat("a", 64), Allocation: a}}
	p.Coordinator.ClusterID, p.Coordinator.PrincipalID = s.cluster, strings.Repeat("9", 32)
	p.CreatedAt, p.ExpiresAt = leaseNow.Unix(), leaseNow.Add(2*time.Minute).Unix()
	config := json.RawMessage(`{"model_dim":16}`)
	p.Contract.ConfigHash = nodejob.Hash(config)
	p.SHA256, err = p.Digest()
	if err != nil {
		t.Fatal(err)
	}
	manifest := nodejob.LearnerManifest{Version: nodejob.LearnerVersion, Controller: actor.Principal, Attempt: strings.Repeat("8", 32), WorkloadGeneration: 1, Plan: p, Slot: p.Slots[0], Config: config, DatasetSelector: "toy", Limits: nodejob.Limits{RuntimeSeconds: 600, CPUSeconds: 1200, MemoryBytes: 1 << 30, DiskBytes: 1 << 30, LogBytes: 1 << 20}}
	request, err := manifest.SigningRequest()
	if err != nil {
		t.Fatal(err)
	}
	j := LearnerJob{Version: 1, Signed: nodejob.SignedLearner{Manifest: manifest, Proof: trust.SignedProof{Request: request, Evidence: trust.TrustEvidence{Principal: actor.Principal}}}, Accepted: trust.AcceptedProof{Generation: 1, Digest: strings.Repeat("a", 64)}}
	if err = s.saveLearner(nil, j); err != nil {
		t.Fatal(err)
	}
	old, leases, err := s.load()
	if err != nil {
		t.Fatal(err)
	}
	live := findLease(&leases, l.ID)
	live.State, live.Job, live.JobKind = Prepared, a.JobID, "learner"
	live.Version++
	leases.NodeVersion++
	if err = s.save(old, leases); err != nil {
		t.Fatal(err)
	}
	return s, actor, q, a.JobID
}

func learnerCredentialRequest(_ context.Context, scope workload.Scope) (workload.SignedRequest, error) {
	_, key, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		return workload.SignedRequest{}, err
	}
	r, err := workload.NewRequest(scope, key)
	if err != nil {
		return workload.SignedRequest{}, err
	}
	q, err := r.SigningRequest()
	return workload.SignedRequest{Request: r, Proof: trust.SignedProof{Request: q, Evidence: trust.TrustEvidence{Principal: scope.Node, Role: trust.Node}}}, err
}

func TestLearnerWorkloadAbortAndCleanupWhileCredentialPortBlocked(t *testing.T) {
	for _, phase := range []string{"initialize", "request"} {
		t.Run(phase, func(t *testing.T) {
			s, actor, q, id := learnerCredentialFixture(t)
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			entered, resume := make(chan struct{}), make(chan struct{})
			var once sync.Once
			unblock := func() { once.Do(func() { close(resume) }) }
			defer unblock()
			block := func() { close(entered); <-resume }
			p := WorkloadPorts{Clock: func() time.Time { return leaseNow }, Initialize: func(context.Context, workload.Scope) error {
				if phase == "initialize" {
					block()
				}
				return nil
			}, Request: func(ctx context.Context, scope workload.Scope) (workload.SignedRequest, error) {
				if phase == "request" {
					block()
				}
				return learnerCredentialRequest(ctx, scope)
			}}
			done := make(chan error, 1)
			go func() { _, err := s.PrepareLearnerWorkload(ctx, actor, id, p); done <- err }()
			select {
			case <-entered:
			case <-ctx.Done():
				t.Fatal("credential port not entered")
			}
			other, err := Open(s.path, s.cluster, s.node)
			if err != nil {
				t.Fatal(err)
			}
			short, stop := context.WithTimeout(ctx, 100*time.Millisecond)
			abort, err := other.AbortReservation(short, actor, q, leaseNow)
			stop()
			if err != nil || abort.Lease == nil || abort.Lease.State != Releasing {
				t.Fatalf("credential port blocked abort: %+v %v", abort, err)
			}
			short, stop = context.WithTimeout(ctx, 30*time.Millisecond)
			err = other.CleanupLearner(short, id, func(context.Context, CleanupRequest) error {
				t.Error("cleanup raced an in-flight credential operation")
				return nil
			})
			stop()
			if !errors.Is(err, context.DeadlineExceeded) {
				t.Fatalf("cleanup did not wait on job lock: %v", err)
			}
			unblock()
			if err = <-done; err == nil {
				t.Fatal("published credential after reservation abort")
			}
			short, stop = context.WithTimeout(ctx, 30*time.Millisecond)
			err = other.CleanupLearner(short, id, func(ctx context.Context, _ CleanupRequest) error {
				<-ctx.Done()
				return ctx.Err()
			})
			stop()
			if !errors.Is(err, context.DeadlineExceeded) {
				t.Fatal(err)
			}
			abort, err = other.AbortReservation(ctx, actor, q, leaseNow)
			if err != nil || abort.Lease.State != Releasing {
				t.Fatal("timeout released uncertain credentials", err)
			}
			cleaned := 0
			err = other.CleanupLearner(ctx, id, func(ctx context.Context, request CleanupRequest) error {
				cleaned++
				if request.Job != id || request.Attempt != request.Workload.Attempt || request.WorkloadInitialized != (phase == "request") {
					t.Error("cleanup changed durable credential initialization evidence")
				}
				// Reenter a node-lock operation from the cleanup port.
				_, err := s.AbortReservation(ctx, actor, q, leaseNow)
				return err
			})
			if err != nil || cleaned != 1 {
				t.Fatalf("cleanup retry failed: calls=%d err=%v", cleaned, err)
			}
			v, err := other.Availability(leaseNow)
			if err != nil || !v.Available {
				t.Fatal("successful cleanup stranded allocation", err)
			}
			if _, err = other.PrepareLearnerWorkload(ctx, actor, id, p); err == nil {
				t.Fatal("preparation recreated released credentials")
			}
		})
	}
}

func TestLearnerWorkloadInitialBudgetAndExactRetry(t *testing.T) {
	for _, cap := range []string{"caller", "actor", "admission"} {
		t.Run(cap, func(t *testing.T) {
			s, actor, _, id := learnerCredentialFixture(t)
			ctx := context.Background()
			want := 2 * time.Minute
			switch cap {
			case "caller":
				var cancel context.CancelFunc
				want = 30 * time.Second
				ctx, cancel = context.WithTimeout(ctx, want)
				defer cancel()
			case "actor":
				want = 20 * time.Second
				actor.ExpiresAt = leaseNow.Add(want)
			}
			var scopes []workload.Scope
			calls := 0
			p := WorkloadPorts{Clock: func() time.Time { return leaseNow }, Initialize: func(ctx context.Context, scope workload.Scope) error {
				d, ok := ctx.Deadline()
				if !ok || time.Until(d) <= 5*time.Second || time.Until(d) > want {
					t.Errorf("incorrect credential budget: %v", time.Until(d))
				}
				scopes = append(scopes, scope)
				if len(scopes) == 1 {
					return context.DeadlineExceeded // ambiguous initialized intent
				}
				return nil
			}, Request: func(ctx context.Context, scope workload.Scope) (workload.SignedRequest, error) {
				calls++
				return learnerCredentialRequest(ctx, scope)
			}}
			if _, err := s.PrepareLearnerWorkload(ctx, actor, id, p); !errors.Is(err, context.DeadlineExceeded) {
				t.Fatal(err)
			}
			if calls != 0 {
				t.Fatal("request ran before durable initialization")
			}
			var err error
			s, err = Open(s.path, s.cluster, s.node)
			if err != nil {
				t.Fatal(err)
			}
			for range 2 {
				if _, err = s.PrepareLearnerWorkload(ctx, actor, id, p); err != nil {
					t.Fatal(err)
				}
			}
			if len(scopes) != 2 || scopes[0] != scopes[1] || calls != 2 {
				t.Fatal("exact retry replaced credential intent or reinitialized durable context")
			}
		})
	}
}

func TestLearnerWorkloadRechecksAuthorityAfterCredentialPorts(t *testing.T) {
	for _, phase := range []string{"initialize", "request"} {
		for _, fence := range []string{"actor_expired", "lease_expired", "caller_canceled"} {
			t.Run(phase+"/"+fence, func(t *testing.T) {
				s, actor, _, id := learnerCredentialFixture(t)
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				now := leaseNow
				invalidate := func() {
					switch fence {
					case "actor_expired":
						now = actor.ExpiresAt
					case "lease_expired":
						now = leaseNow.Add(time.Minute)
					case "caller_canceled":
						cancel()
					}
				}
				requests := 0
				p := WorkloadPorts{Clock: func() time.Time { return now }, Initialize: func(context.Context, workload.Scope) error {
					if phase == "initialize" {
						invalidate()
					}
					return nil
				}, Request: func(ctx context.Context, scope workload.Scope) (workload.SignedRequest, error) {
					requests++
					out, err := learnerCredentialRequest(ctx, scope)
					invalidate()
					return out, err
				}}
				out, err := s.PrepareLearnerWorkload(ctx, actor, id, p)
				if err == nil || out.Request.Version != "" {
					t.Fatal("credential port result escaped an invalidated authority")
				}
				_, j, err := s.loadLearner(id)
				if err != nil {
					t.Fatal(err)
				}
				if j.Workload == nil || j.Workload.Initialized != (phase == "request") || phase == "initialize" && requests != 0 {
					t.Fatal("advanced credential state after authority was invalidated")
				}
			})
		}
	}
}
