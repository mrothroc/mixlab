//go:build darwin || linux

package clusterapp

import (
	"context"
	"crypto/x509"
	"errors"
	"path/filepath"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/workload"
	"github.com/mrothroc/mixlab/workerprobe"
)

func preparationAuthorizationFixture(t *testing.T) (learnerRuntimeFixture, *atomic.Int64) {
	t.Helper()
	f := newLearnerRuntimeFixtureMode(t, false)
	ctx := context.Background()
	var unix atomic.Int64
	unix.Store(testNow.Unix())
	f.remote.runtime.Clock = func() time.Time { return time.Unix(unix.Load(), 0) }
	var err error
	f.ports, err = NodeWorkloadPorts(f.remote.runtime.Principal, f.root, f.remote.runtime.Clock)
	check(t, err)
	m := &f.prepare.Signed.Manifest
	m.Plan.ExpiresAt = testNow.Add(30 * time.Minute).Unix()
	m.Plan.SHA256, err = m.Plan.Digest()
	check(t, err)
	request, err := m.SigningRequest()
	check(t, err)
	identity, signer, err := f.controller.Active(testNow)
	check(t, err)
	view, err := f.authority.Current(ctx, testNow)
	check(t, err)
	f.prepare.Signed.Proof, err = trust.SignPrincipalProof(f.authority.Anchor, identity.Chain, signer, view, request, testNow)
	check(t, err)
	lease, err := f.remote.runtime.Store.LeaseStatus(ctx, f.actor, m.Slot.Allocation.LeaseID, testNow)
	check(t, err)
	lease, err = f.remote.runtime.Store.RenewLease(ctx, f.actor, nodeagent.LeaseCommand{Lease: lease.ID, ExpectedVersion: lease.Version, IdempotencyKey: strings.Repeat("8", 32), TTLSeconds: 1800}, testNow)
	check(t, err)
	f.prepare.ExpectedLeaseVersion = lease.Version
	return f, &unix
}

func TestLearnerPreparationRefreshesOnlyAuthorization(t *testing.T) {
	f, unix := preparationAuthorizationFixture(t)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	var numerical, probes, datasets, initializes, requests, authorizations atomic.Int32
	app := preparationApp(t, f, context.Background(), func(context.Context, []byte) (workerprobe.Plan, error) {
		numerical.Add(1)
		return preparationNumerical(f), nil
	})
	probe, dataset := app.options.Preparation.Probe, app.options.Preparation.DatasetID
	app.options.Preparation.Probe = func(ctx context.Context) (workerprobe.Report, error) {
		probes.Add(1)
		return probe(ctx)
	}
	app.options.Preparation.DatasetID = func(ctx context.Context, path string) (string, error) {
		datasets.Add(1)
		return dataset(ctx, path)
	}
	initialize, request, authorize := app.options.Workload.Initialize, app.options.Workload.Request, app.options.Workload.AuthorizeRequest
	if authorize == nil {
		t.Fatal("production workload ports lack request authorization")
	}
	app.options.Workload.Initialize = func(ctx context.Context, scope workload.Scope) error {
		initializes.Add(1)
		return initialize(ctx, scope)
	}
	app.options.Workload.Request = func(ctx context.Context, scope workload.Scope) (workload.SignedRequest, error) {
		requests.Add(1)
		return request(ctx, scope)
	}
	entered := make(chan context.Context, 1)
	release := make(chan struct{})
	var stale workload.RequestAuthorization
	app.options.Workload.AuthorizeRequest = func(ctx context.Context, scope workload.Scope, original workload.SignedRequest) (workload.RequestAuthorization, error) {
		switch authorizations.Add(1) {
		case 2:
			return stale, nil
		case 3:
			entered <- ctx
			select {
			case <-release:
			case <-ctx.Done():
				return workload.RequestAuthorization{}, ctx.Err()
			}
		}
		return authorize(ctx, scope, original)
	}
	first, err := app.PrepareLearner(ctx, f.actor, f.prepare)
	check(t, err)
	stale = first.Authorization
	again, err := app.PrepareLearner(ctx, f.actor, f.prepare)
	check(t, err)
	if !reflect.DeepEqual(first, again) || authorizations.Load() != 1 {
		t.Fatal("fresh completed authorization was not reused")
	}

	next := testNow.Add(trust.SnapshotLifetime + time.Second)
	view, err := f.authority.Current(ctx, next)
	check(t, err)
	snapshot, err := f.authority.Snapshots.Load(next)
	check(t, err)
	check(t, refreshPrincipal(ctx, f.remote.runtime.Principal, snapshot, next))
	check(t, refreshPrincipal(ctx, f.controller, snapshot, next))
	identity, _, err := f.controller.Active(next)
	check(t, err)
	f.actor, err = trust.AuthenticatePrincipal(f.authority.Anchor, view, identity.Chain, next)
	check(t, err)
	unix.Store(next.Unix())
	if err := workload.VerifyRequestAuthorization(f.authority.Anchor, view, first.Request, stale, next); err == nil {
		t.Fatal("fixture did not expire the cached authorization")
	}
	// A structurally valid but expired port result must never escape as success.
	bad, err := app.PrepareLearner(ctx, f.actor, f.prepare)
	if err == nil || !reflect.DeepEqual(bad, NodeLearnerPrepared{}) || authorizations.Load() != 2 {
		t.Fatalf("expired replacement escaped or retried without a new call: err=%v calls=%d", err, authorizations.Load())
	}

	caller, cancelCaller := context.WithCancel(ctx)
	defer cancelCaller()
	firstWaiter := make(chan error, 1)
	go func() { _, err := app.PrepareLearner(caller, f.actor, f.prepare); firstWaiter <- err }()
	var operation context.Context
	select {
	case operation = <-entered:
	case <-ctx.Done():
		t.Fatal("authorization refresh did not start", ctx.Err())
	}
	deadline, bounded := operation.Deadline()
	if !bounded || time.Until(deadline) <= time.Minute {
		t.Fatal("authorization refresh inherited a short request deadline")
	}
	cancelCaller()
	select {
	case err := <-firstWaiter:
		if !errors.Is(err, context.Canceled) || operation.Err() != nil {
			t.Fatalf("caller cancellation changed authorization ownership: caller=%v operation=%v", err, operation.Err())
		}
	case <-ctx.Done():
		t.Fatal("canceled waiter did not return", ctx.Err())
	}
	type result struct {
		out NodeLearnerPrepared
		err error
	}
	results := make(chan result, 2)
	for range 2 {
		go func() {
			out, err := app.PrepareLearner(ctx, f.actor, f.prepare)
			results <- result{out, err}
		}()
	}
	close(release)
	var refreshed NodeLearnerPrepared
	for i := range 2 {
		select {
		case got := <-results:
			check(t, got.err)
			check(t, workload.VerifyRequestAuthorization(f.authority.Anchor, view, got.out.Request, got.out.Authorization, next))
			if !reflect.DeepEqual(first.Request, got.out.Request) || !reflect.DeepEqual(first.Registration, got.out.Registration) || reflect.DeepEqual(stale, got.out.Authorization) {
				t.Fatal("refresh replaced stable preparation or retained expired authorization")
			}
			if i != 0 && !reflect.DeepEqual(refreshed, got.out) {
				t.Fatal("concurrent waiters received different authorization results")
			}
			refreshed = got.out
		case <-ctx.Done():
			t.Fatal("authorization waiters did not complete", ctx.Err())
		}
	}
	if numerical.Load() != 1 || probes.Load() != 1 || datasets.Load() != 1 || initializes.Load() != 1 || requests.Load() != 1 || authorizations.Load() != 3 {
		t.Fatalf("authorization refresh replayed work: numerical=%d probes=%d datasets=%d initializes=%d requests=%d authorizations=%d", numerical.Load(), probes.Load(), datasets.Load(), initializes.Load(), requests.Load(), authorizations.Load())
	}
}

func TestLearnerPreparationRequiresValidAuthorizationPort(t *testing.T) {
	for _, mode := range []string{"missing", "forged"} {
		t.Run(mode, func(t *testing.T) {
			f := newLearnerRuntimeFixtureMode(t, false)
			app := preparationApp(t, f, context.Background(), func(context.Context, []byte) (workerprobe.Plan, error) { return preparationNumerical(f), nil })
			authorize := app.options.Workload.AuthorizeRequest
			if mode == "missing" {
				app.options.Workload.AuthorizeRequest = nil
			} else {
				if authorize == nil {
					t.Fatal("production workload ports lack request authorization")
				}
				app.options.Workload.AuthorizeRequest = func(ctx context.Context, scope workload.Scope, original workload.SignedRequest) (workload.RequestAuthorization, error) {
					auth, err := authorize(ctx, scope, original)
					if err == nil {
						auth.Proof.Signature = append([]byte(nil), auth.Proof.Signature...)
						auth.Proof.Signature[0] ^= 1
					}
					return auth, err
				}
			}
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			out, err := app.PrepareLearner(ctx, f.actor, f.prepare)
			if err == nil || !reflect.DeepEqual(out, NodeLearnerPrepared{}) {
				t.Fatalf("invalid authorization fell back to legacy preparation: %v", err)
			}
		})
	}
}

func TestLearnerPreparationDoesNotRefreshPermanentAuthorizationFailure(t *testing.T) {
	for _, mode := range []string{"tampered", "expired_snapshot_and_compromised_signer"} {
		t.Run(mode, func(t *testing.T) {
			f, unix := preparationAuthorizationFixture(t)
			ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
			defer cancel()
			app := preparationApp(t, f, context.Background(), func(context.Context, []byte) (workerprobe.Plan, error) { return preparationNumerical(f), nil })
			authorize := app.options.Workload.AuthorizeRequest
			if authorize == nil {
				t.Fatal("production workload ports lack request authorization")
			}
			var calls atomic.Int32
			app.options.Workload.AuthorizeRequest = func(ctx context.Context, scope workload.Scope, original workload.SignedRequest) (workload.RequestAuthorization, error) {
				if calls.Add(1) != 1 {
					return workload.RequestAuthorization{}, errors.New("permanent failure reached refresh port")
				}
				return authorize(ctx, scope, original)
			}
			first, err := app.PrepareLearner(ctx, f.actor, f.prepare)
			check(t, err)
			if mode == "tampered" {
				app.prepareMu.Lock()
				app.preparing.out.Authorization.Proof.Signature[0] ^= 1
				app.prepareMu.Unlock()
			} else {
				next := testNow.Add(trust.SnapshotLifetime + time.Second)
				snapshot, err := f.authority.Snapshots.Revoke(ctx, "certificate", first.Authorization.Proof.Evidence.Serial, "compromise", "test", next)
				check(t, err)
				view, err := trust.VerifySnapshot(f.authority.Anchor, snapshot, next)
				check(t, err)
				// Exercise the cached-signer check independently of NodeCurrentTrust,
				// which also rejects a revoked local node before serving a response.
				app.options.CurrentTrust = func(context.Context, time.Time) (trust.Anchor, trust.VerifiedSnapshot, error) {
					return f.authority.Anchor, view, nil
				}
				unix.Store(next.Unix())
			}
			out, err := app.PrepareLearner(ctx, f.actor, f.prepare)
			if err == nil || !reflect.DeepEqual(out, NodeLearnerPrepared{}) || calls.Load() != 1 {
				t.Fatalf("permanent authorization failure was refreshed: err=%v calls=%d", err, calls.Load())
			}
		})
	}
}

func TestLearnerPreparationRefreshesAfterNodeCertificateRenewal(t *testing.T) {
	for _, mode := range []string{"expired_certificate", "prospectively_retired_certificate", "failed_replacement_then_compromise"} {
		t.Run(mode, func(t *testing.T) {
			f := newLearnerRuntimeFixtureMode(t, false)
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()
			node := f.remote.runtime.Principal
			originalIdentity, _, err := node.Active(testNow)
			check(t, err)
			leaf, err := x509.ParseCertificate(originalIdentity.Chain[0])
			check(t, err)
			start := leaf.NotAfter.Add(-5 * time.Minute)
			_, err = f.authority.Current(ctx, start)
			check(t, err)
			check(t, RenewLocalPrincipal(ctx, f.authority, f.authority.Principal, start))
			check(t, RenewLocalPrincipal(ctx, f.authority, f.controller, start))
			snapshot, err := f.authority.Snapshots.Load(start)
			check(t, err)
			check(t, refreshPrincipal(ctx, node, snapshot, start))
			view, err := f.authority.Current(ctx, start)
			check(t, err)
			controller, signer, err := f.controller.Active(start)
			check(t, err)
			f.actor, err = trust.AuthenticatePrincipal(f.authority.Anchor, view, controller.Chain, start)
			check(t, err)

			// Start a new admission near the existing node certificate's expiry.
			// The original fixture's old lease is not revived or extended in place.
			path, err := statehome.Resolve(statehome.Options{ExactDir: filepath.Join(f.root.Dir(), "late-agent")}, statehome.Context{Kind: statehome.Agent})
			check(t, err)
			check(t, path.Ensure())
			s, err := nodeagent.Initialize(ctx, path, f.authority.Anchor.Cluster(), originalIdentity.Principal, 1)
			check(t, err)
			m := &f.prepare.Signed.Manifest
			check(t, s.InstallProfile(ctx, nodeagent.Profile{Version: nodeagent.ProfileVersion, Node: originalIdentity.Principal, DisplayName: "renewal-test", Generation: 1, Probe: f.probe, ProbeObservedAt: start.Unix(), Limits: m.Limits, Datasets: []nodeagent.LocalDataset{{Dataset: nodeagent.Dataset{Selector: m.DatasetSelector, ID: m.Plan.Contract.DatasetHash}, TrainPattern: "/synthetic/train.bin"}}}))
			caps, err := s.Capabilities(ctx, f.actor, start)
			check(t, err)
			lease, err := s.Reserve(ctx, f.actor, nodeagent.Reserve{IdempotencyKey: strings.Repeat("9", 32), ExpectedNodeVersion: 1, CapabilityGeneration: 1, Run: m.Plan.Cohort.RunID, TTLSeconds: 1800}, start)
			check(t, err)
			allocation := run.Allocation{Provider: "node_agent", NodeID: originalIdentity.Principal, LeaseID: lease.ID, JobID: m.Slot.Allocation.JobID}
			allocation.RefHash, err = run.Digest(allocation)
			check(t, err)
			m.Slot.Allocation = allocation
			m.Slot.CapabilityHash, err = run.Digest(caps)
			check(t, err)
			m.Plan.Slots[0] = m.Slot
			m.Plan.CreatedAt, m.Plan.ExpiresAt = start.Unix(), start.Add(30*time.Minute).Unix()
			m.Plan.SHA256, err = m.Plan.Digest()
			check(t, err)
			q, err := m.SigningRequest()
			check(t, err)
			f.prepare.Signed.Proof, err = trust.SignPrincipalProof(f.authority.Anchor, controller.Chain, signer, view, q, start)
			check(t, err)
			f.prepare.ExpectedLeaseVersion = lease.Version
			var unix atomic.Int64
			unix.Store(start.Unix())
			f.remote.runtime.Store = s
			f.remote.runtime.Clock = func() time.Time { return time.Unix(unix.Load(), 0) }
			f.ports, err = NodeWorkloadPorts(node, f.root, f.remote.runtime.Clock)
			check(t, err)
			var numerical, authorizations atomic.Int32
			app := preparationApp(t, f, context.Background(), func(context.Context, []byte) (workerprobe.Plan, error) {
				numerical.Add(1)
				return preparationNumerical(f), nil
			})
			authorize := app.options.Workload.AuthorizeRequest
			if authorize == nil {
				t.Fatal("production workload ports lack request authorization")
			}
			transient := errors.New("temporary replacement signing failure")
			app.options.Workload.AuthorizeRequest = func(ctx context.Context, scope workload.Scope, original workload.SignedRequest) (workload.RequestAuthorization, error) {
				if authorizations.Add(1) == 2 && mode == "failed_replacement_then_compromise" {
					return workload.RequestAuthorization{}, transient
				}
				return authorize(ctx, scope, original)
			}
			first, err := app.PrepareLearner(ctx, f.actor, f.prepare)
			check(t, err)
			app.prepareMu.Lock()
			originalReceipt := app.preparing.authorization
			app.prepareMu.Unlock()
			check(t, RenewLocalPrincipal(ctx, f.authority, node, start.Add(time.Minute)))
			next := leaf.NotAfter.Add(time.Second)
			if mode == "prospectively_retired_certificate" {
				_, err = f.authority.Snapshots.Revoke(ctx, "certificate", first.Authorization.Proof.Evidence.Serial, "prospective", "renewed", next)
				check(t, err)
			}
			view, err = f.authority.Current(ctx, next)
			check(t, err)
			snapshot, err = f.authority.Snapshots.Load(next)
			check(t, err)
			check(t, refreshPrincipal(ctx, node, snapshot, next))
			check(t, refreshPrincipal(ctx, f.controller, snapshot, next))
			unix.Store(next.Unix())
			if next.Unix() >= first.Authorization.Proof.Evidence.Snapshot.Payload.ExpiresAt {
				t.Fatal("fixture expired the snapshot rather than only the old certificate")
			}
			if err := workload.VerifyRequestAuthorization(f.authority.Anchor, view, first.Request, first.Authorization, next); err == nil {
				t.Fatal("old certificate unexpectedly remained usable")
			}
			fresh, err := app.PrepareLearner(ctx, f.actor, f.prepare)
			if mode == "failed_replacement_then_compromise" {
				if !errors.Is(err, transient) || !reflect.DeepEqual(fresh, NodeLearnerPrepared{}) || authorizations.Load() != 2 {
					t.Fatalf("replacement did not stop on its transient failure: %v", err)
				}
				app.prepareMu.Lock()
				prior := app.preparing.prepared
				app.prepareMu.Unlock()
				originalDigest, err := run.Digest(first)
				check(t, err)
				if prior == nil || prior.authorization != originalReceipt || prior.resultDigest != originalDigest {
					t.Fatal("failed replacement lost the owner's original receipt or result digest")
				}
				// The retry seed must not alias a previously returned response.
				first.Authorization.Proof.Signature[0] ^= 1
				check(t, prior.verifyDigest())
				snapshot, err = f.authority.Snapshots.Revoke(ctx, "certificate", first.Authorization.Proof.Evidence.Serial, "compromise", "old signer compromised", next)
				check(t, err)
				check(t, refreshPrincipal(ctx, node, snapshot, next))
				_, _, err = NodeCurrentTrust(node)(ctx, next)
				check(t, err)
				for range 2 {
					out, err := app.PrepareLearner(ctx, f.actor, f.prepare)
					if err == nil || !reflect.DeepEqual(out, NodeLearnerPrepared{}) || authorizations.Load() != 2 || numerical.Load() != 1 {
						t.Fatalf("failed replacement bypassed compromised history: err=%v authorizations=%d numerical=%d", err, authorizations.Load(), numerical.Load())
					}
				}
				return
			}
			check(t, err)
			check(t, workload.VerifyRequestAuthorization(f.authority.Anchor, view, fresh.Request, fresh.Authorization, next))
			if fresh.Authorization.Proof.Evidence.Serial == first.Authorization.Proof.Evidence.Serial || !reflect.DeepEqual(first.Request, fresh.Request) || !reflect.DeepEqual(first.Registration, fresh.Registration) || numerical.Load() != 1 || authorizations.Load() != 2 {
				t.Fatal("renewal did not refresh only the exact request's authorization")
			}

			snapshot, err = f.authority.Snapshots.Revoke(ctx, "principal", originalIdentity.Principal, "prospective", "deny current node", next)
			check(t, err)
			check(t, refreshPrincipal(ctx, node, snapshot, next))
			out, err := app.PrepareLearner(ctx, f.actor, f.prepare)
			if err == nil || !reflect.DeepEqual(out, NodeLearnerPrepared{}) || authorizations.Load() != 2 {
				t.Fatal("historical acceptance bypassed current node revocation")
			}
		})
	}
}
