package integration

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/coordinator"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/workload"
)

type rolloverRecoveryNode struct {
	clusterapp.LearnerRecoveryNode
	change func(*clusterapp.NodeLearnerRetryPrepared)
}

type rolloverStartNode struct {
	clusterapp.LearnerRecoveryNode
	change func(*clusterapp.NodeLearnerRetryStart)
}

func (n rolloverStartNode) StartLearnerRetry(ctx context.Context, id string, q clusterapp.NodeLearnerRetryStart) error {
	n.change(&q)
	return n.LearnerRecoveryNode.StartLearnerRetry(ctx, id, q)
}

func advanceRecoverySnapshot(t *testing.T, f *recoveryFixture) {
	t.Helper()
	f.x.f.snapshot.Payload.Generation++
	f.x.f.signSnapshot(t)
}

func TestProductionLearnerRecoveryMirrorSnapshotRollover(t *testing.T) {
	for _, credentialOnly := range []bool{false, true} {
		for _, concurrent := range []bool{false, true} {
			t.Run(fmt.Sprintf("credential_only=%v/concurrent_refresh=%v", credentialOnly, concurrent), func(t *testing.T) {
				var f *recoveryFixture
				var refreshes atomic.Int32
				n := &recoveryNodeTrust{}
				n.refresh = func(context.Context) error {
					refreshes.Add(1)
					n.pin(f.x.f.view)
					if concurrent {
						return fmt.Errorf("in-flight snapshot superseded by concurrent maintenance")
					}
					return nil
				}
				f = newRecoveryFixtureWithNodeTrust(t, strings.Repeat("a", 64), true, n)
				before := f.state(t)
				f.stop(t)
				old := f.x.f.view
				n.pin(old)
				if !credentialOnly {
					advanceRecoverySnapshot(t, f)
				}
				starts := 0
				for range 4 {
					r := f.recovery(t)
					if credentialOnly {
						// The authority advances after granting, while both the node
						// and the acknowledgement signer still use the old view.
						issue := r.Issue
						r.Issue = func(ctx context.Context, g workload.Grant) (workload.Result, error) {
							advanceRecoverySnapshot(t, f)
							return issue(ctx, g)
						}
						r.Sign = func(_ context.Context, q trust.SignRequest) (trust.SignedProof, error) {
							return trust.SignPrincipalProof(f.x.f.a, f.x.coordinator.Chain, f.x.coordinator.Key, old, q, f.x.f.now)
						}
					}
					r.Node = func(run.Allocation) (clusterapp.LearnerRecoveryNode, error) {
						return rolloverStartNode{f.node, func(q *clusterapp.NodeLearnerRetryStart) {
							starts++
							ack := q.Acknowledgement.Proof.Evidence.Snapshot.Payload.Generation
							if q.Credential.Snapshot.Payload.Generation <= old.Generation() || credentialOnly && ack != old.Generation() || !credentialOnly && ack <= old.Generation() {
								t.Fatal("retry did not exercise intended acknowledgement/credential skew")
							}
						}}, nil
					}
					_, err := r.Tick(context.Background(), f.assignment.Binding.RunID)
					check(t, err)
					s := f.state(t)
					if s.Phase == coordinator.Failed {
						t.Fatal(s.Failure)
					}
					if s.Recovery == nil && len(s.Attempts) == 1 {
						x, err := f.x.store.ActiveExecution(context.Background())
						check(t, err)
						if refreshes.Load() != 1 || starts != 1 || n.current(f.x.f).Generation() != f.x.f.view.Generation() || !reflect.DeepEqual(before.Assignments, s.Assignments) || !reflect.DeepEqual(before.Plan.Slots, s.Plan.Slots) || x.Learner.Stopped || x.Workload.Generation != 2 || x.Learner.Assignment.AttemptID != s.Attempts[0].AttemptID || x.Learner.Assignment.ExpectedPrivateHash != s.Attempts[0].PrivateHash || f.x.initializations.Load() != 2 {
							t.Fatal("mirror rollover did not refresh once and preserve retry identity")
						}
						return
					}
				}
				t.Fatal("mirror rollover recovery did not converge")
			})
		}
	}
}

func TestProductionLearnerRecoveryMirrorRolloverWaitIsBounded(t *testing.T) {
	for _, mode := range []string{"unchanged", "unavailable", "no_refresh_port"} {
		t.Run(mode, func(t *testing.T) {
			var refreshes atomic.Int32
			n := &recoveryNodeTrust{}
			if mode != "no_refresh_port" {
				n.refresh = func(context.Context) error {
					refreshes.Add(1)
					if mode == "unavailable" {
						return io.ErrUnexpectedEOF
					}
					return nil
				}
			}
			f := newRecoveryFixtureWithNodeTrust(t, strings.Repeat("a", 64), true, n)
			renewRecoveryLeaseThroughBudget(t, f)
			f.stop(t)
			n.pin(f.x.f.view)
			advanceRecoverySnapshot(t, f)
			_, err := f.recovery(t).Tick(context.Background(), f.assignment.Binding.RunID)
			check(t, err)
			started := f.state(t).Recovery.Started
			for _, elapsed := range []int64{0, coordinator.LearnerRecoverySeconds - 1} {
				f.x.f.now = time.Unix(started+elapsed, 0)
				_, err = f.recovery(t).Tick(context.Background(), f.assignment.Binding.RunID)
				var remote *clusterapp.NodeHTTPError
				if !errors.As(err, &remote) || remote.Status != http.StatusServiceUnavailable {
					t.Fatalf("behind node returned %v, want HTTP 503", err)
				}
				s := f.state(t)
				x, err := f.x.store.ActiveExecution(context.Background())
				check(t, err)
				if s.Phase == coordinator.Failed || s.Recovery == nil || s.Recovery.Started != started || len(s.Attempts) != 1 || !x.Learner.Stopped || x.Workload.Generation != 1 || x.Learner.Recovery == nil || f.x.initializations.Load() != 2 {
					t.Fatal("trust wait renewed budget, failed permanently or launched replacement")
				}
			}
			before := refreshes.Load()
			f.x.f.now = time.Unix(started+coordinator.LearnerRecoverySeconds, 0)
			_, err = f.recovery(t).Tick(context.Background(), f.assignment.Binding.RunID)
			check(t, err)
			s := f.state(t)
			if s.Phase != coordinator.Failed || !strings.Contains(s.Failure, "budget exhausted") || refreshes.Load() != before {
				t.Fatal("node trust wait extended persisted recovery budget", s.Failure)
			}
			if mode == "no_refresh_port" && before != 0 || mode != "no_refresh_port" && before != 2 {
				t.Fatal("unexpected node refresh count", before)
			}
		})
	}
}

func TestProductionLearnerRecoveryMirrorRolloverFailsClosed(t *testing.T) {
	for _, mode := range []string{"signature", "generation", "binding", "revoked", "concurrent_revoked", "credential_signature", "credential_binding"} {
		t.Run(mode, func(t *testing.T) {
			var f *recoveryFixture
			var refreshes atomic.Int32
			n := &recoveryNodeTrust{}
			n.refresh = func(context.Context) error {
				refreshes.Add(1)
				payload := f.x.f.snapshot.Payload
				payload.Generation++
				payload.Revocations = []trust.Revocation{{Kind: "principal", ID: f.state(t).Plan.Coordinator.PrincipalID, Mode: "compromise", Reason: "test", FirstGeneration: payload.Generation}}
				signed, err := trust.SignSnapshot(f.x.f.a, payload, f.x.f.snapshotCert, f.x.f.snapshotKey, f.x.f.now)
				if err != nil {
					return err
				}
				view, err := trust.VerifySnapshot(f.x.f.a, signed, f.x.f.now)
				if err != nil {
					return err
				}
				n.pin(view)
				if mode == "concurrent_revoked" {
					return fmt.Errorf("in-flight snapshot superseded by concurrent revocation")
				}
				return nil
			}
			f = newRecoveryFixtureWithNodeTrust(t, strings.Repeat("a", 64), true, n)
			f.stop(t)
			n.pin(f.x.f.view)
			advanceRecoverySnapshot(t, f)
			r := f.recovery(t)
			r.Node = func(run.Allocation) (clusterapp.LearnerRecoveryNode, error) {
				return rolloverStartNode{f.node, func(q *clusterapp.NodeLearnerRetryStart) {
					switch mode {
					case "signature":
						q.Acknowledgement.Proof.Signature[0] ^= 1
					case "generation":
						q.Acknowledgement.Proof.Evidence.Snapshot.Payload.Generation++
					case "binding":
						q.Acknowledgement.Body.PrivateHash = strings.Repeat("f", 64)
					case "credential_signature":
						q.Credential.Snapshot.Signature[0] ^= 1
					case "credential_binding":
						q.Credential.Binding.Attempt = id(t)
					}
				}}, nil
			}
			_, err := r.Tick(context.Background(), f.assignment.Binding.RunID)
			check(t, err)
			_, err = r.Tick(context.Background(), f.assignment.Binding.RunID)
			var remote *clusterapp.NodeHTTPError
			if !errors.As(err, &remote) || remote.Status != http.StatusConflict || f.state(t).Phase != coordinator.Failed {
				t.Fatal("invalid retry did not fail closed with HTTP 409", err)
			}
			x, err := f.x.store.ActiveExecution(context.Background())
			check(t, err)
			if !x.Learner.Stopped || x.Workload.Generation != 1 || len(x.Learner.History) != 0 || x.Learner.Recovery == nil {
				t.Fatal("rejected retry changed node hosting permission")
			}
			want := int32(0)
			if mode == "revoked" || mode == "concurrent_revoked" {
				want = 1
			}
			if refreshes.Load() != want {
				t.Fatalf("refreshes=%d, want %d", refreshes.Load(), want)
			}
		})
	}
}

func (n rolloverRecoveryNode) AuthorizeLearnerRetry(ctx context.Context, id string, q clusterapp.NodeLearnerRetryRequest) (clusterapp.NodeLearnerRetryPrepared, error) {
	p, err := n.LearnerRecoveryNode.AuthorizeLearnerRetry(ctx, id, q)
	if err == nil && n.change != nil {
		n.change(&p)
	}
	return p, err
}

func TestProductionLearnerRecoverySnapshotRollover(t *testing.T) {
	for _, concurrent := range []bool{false, true} {
		t.Run(fmt.Sprintf("concurrent_refresh=%v", concurrent), func(t *testing.T) {
			testRecoverySnapshotRollover(t, concurrent)
		})
	}
}

func testRecoverySnapshotRollover(t *testing.T, concurrent bool) {
	f := newRecoveryFixture(t)
	f.stop(t)
	old := f.state(t)
	current := f.x.f.view
	// The node observes the new authority generation first. The coordinator
	// retains its independently cached, still-fresh previous generation.
	f.x.f.snapshot.Payload.Generation++
	f.x.f.signSnapshot(t)
	refreshes := 0
	for tick := 0; tick < 4; tick++ {
		r := f.recovery(t)
		r.Current = func(context.Context) (trust.VerifiedSnapshot, error) { return current, nil }
		r.Refresh = func(ctx context.Context) error {
			if _, ok := ctx.Deadline(); !ok {
				t.Fatal("recovery refresh has no deadline")
			}
			refreshes++
			current = f.x.f.view
			if concurrent {
				return fmt.Errorf("in-flight snapshot superseded by concurrent maintenance")
			}
			return nil
		}
		_, err := r.Tick(context.Background(), f.assignment.Binding.RunID)
		check(t, err)
		s := f.state(t)
		if s.Phase == coordinator.Failed {
			t.Fatal(s.Failure)
		}
		if s.Recovery == nil && len(s.Attempts) == 1 {
			if refreshes != 1 || current.Generation() != f.x.f.view.Generation() || !reflect.DeepEqual(old.Assignments, s.Assignments) || !reflect.DeepEqual(old.Plan.Slots, s.Plan.Slots) {
				t.Fatal("rollover changed allocation or did not refresh exactly once")
			}
			return
		}
	}
	t.Fatal("rollover recovery did not converge")
}

func TestProductionLearnerRecoveryRolloverWaitIsBounded(t *testing.T) {
	for _, mode := range []string{"unchanged", "unavailable", "no_refresh_port"} {
		t.Run(mode, func(t *testing.T) {
			f := newRecoveryFixture(t)
			f.stop(t)
			current := f.x.f.view
			f.x.f.snapshot.Payload.Generation++
			f.x.f.signSnapshot(t)
			r := f.recovery(t)
			r.Current = func(context.Context) (trust.VerifiedSnapshot, error) { return current, nil }
			refreshes := 0
			if mode != "no_refresh_port" {
				r.Refresh = func(context.Context) error {
					refreshes++
					if mode == "unavailable" {
						return io.ErrUnexpectedEOF
					}
					return nil
				}
			}
			_, err := r.Tick(context.Background(), f.assignment.Binding.RunID)
			check(t, err)
			started := f.state(t).Recovery.Started
			for range 2 {
				_, err = r.Tick(context.Background(), f.assignment.Binding.RunID)
				if err == nil {
					t.Fatal("behind coordinator accepted evidence")
				}
				s := f.state(t)
				if s.Phase == coordinator.Failed || s.Recovery == nil || s.Recovery.Started != started || len(s.Attempts) != 0 {
					t.Fatal("refresh wait failed permanently, reset budget or bound attempt", s)
				}
			}
			before := refreshes
			f.x.f.now = f.x.f.now.Add(time.Duration(coordinator.LearnerRecoverySeconds) * time.Second)
			_, err = r.Tick(context.Background(), f.assignment.Binding.RunID)
			check(t, err)
			s := f.state(t)
			if s.Phase != coordinator.Failed || !strings.Contains(s.Failure, "budget exhausted") || refreshes != before {
				t.Fatal("trust refresh extended recovery deadline", s.Failure)
			}
		})
	}
}

func TestProductionLearnerRecoveryRolloverFailsClosed(t *testing.T) {
	for _, mode := range []string{"signature", "generation", "binding", "revoked", "concurrent_revoked", "stale_evidence"} {
		t.Run(mode, func(t *testing.T) {
			f := newRecoveryFixture(t)
			f.stop(t)
			current := f.x.f.view
			f.x.f.snapshot.Payload.Generation++
			if mode == "stale_evidence" {
				f.x.f.snapshot.Payload.ExpiresAt = f.x.f.now.Add(time.Second).Unix()
			}
			f.x.f.signSnapshot(t)
			refreshes := 0
			r := f.recovery(t)
			r.Current = func(context.Context) (trust.VerifiedSnapshot, error) { return current, nil }
			r.Refresh = func(context.Context) error {
				refreshes++
				payload := f.x.f.snapshot.Payload
				payload.Generation++
				payload.Revocations = []trust.Revocation{{Kind: "principal", ID: f.x.prepare.Signed.Manifest.Slot.Allocation.NodeID, Mode: "compromise", Reason: "test", FirstGeneration: payload.Generation}}
				signed, err := trust.SignSnapshot(f.x.f.a, payload, f.x.f.snapshotCert, f.x.f.snapshotKey, f.x.f.now)
				check(t, err)
				current, err = trust.VerifySnapshot(f.x.f.a, signed, f.x.f.now)
				if err == nil && mode == "concurrent_revoked" {
					return fmt.Errorf("in-flight snapshot superseded by concurrent revocation")
				}
				return err
			}
			r.Node = func(run.Allocation) (clusterapp.LearnerRecoveryNode, error) {
				return rolloverRecoveryNode{LearnerRecoveryNode: f.node, change: func(p *clusterapp.NodeLearnerRetryPrepared) {
					switch mode {
					case "signature":
						p.Evidence.Proof.Signature[0] ^= 1
					case "generation":
						p.Evidence.Proof.Evidence.Snapshot.Payload.Generation++
					case "binding":
						p.Evidence.Proof.Request.Context = "wrong-run"
					case "stale_evidence":
						// Keep the real signed proof unchanged while its short-lived
						// snapshot expires and the coordinator advances past it.
						f.x.f.mu.Lock()
						f.x.f.now = f.x.f.now.Add(2 * time.Second)
						f.x.f.snapshot.Payload.Generation++
						f.x.f.snapshot.Payload.IssuedAt = f.x.f.now.Unix()
						f.x.f.snapshot.Payload.ExpiresAt = f.x.f.now.Add(trust.SnapshotLifetime).Unix()
						f.x.f.signSnapshot(t)
						current = f.x.f.view
						f.x.f.mu.Unlock()
					}
				}}, nil
			}
			_, err := r.Tick(context.Background(), f.assignment.Binding.RunID)
			check(t, err)
			_, err = r.Tick(context.Background(), f.assignment.Binding.RunID)
			if err == nil || f.state(t).Phase != coordinator.Failed || len(f.state(t).Attempts) != 0 {
				t.Fatal("invalid recovery evidence did not fail closed", err)
			}
			want := 0
			if mode == "revoked" || mode == "concurrent_revoked" {
				want = 1
			}
			if refreshes != want {
				t.Fatalf("refreshes=%d, want %d", refreshes, want)
			}
		})
	}
}
