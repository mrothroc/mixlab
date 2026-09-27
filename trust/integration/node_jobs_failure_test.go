package integration

import (
	"context"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/workerhost"
)

func TestNodeJobsRevokedTransportCancelsAndReleases(t *testing.T) {
	for _, start := range []bool{false, true} {
		t.Run(map[bool]string{false: "prepared", true: "running"}[start], func(t *testing.T) {
			x := newNodeJobsFixture(t)
			host, err := workerhost.NewAttemptStore(runtimePath(t, statehome.Worker))
			check(t, err)
			runner := &runtimeTestRunner{finish: make(chan struct{}), started: make(chan struct{})}
			ready, done := make(chan struct{}), make(chan error, 1)
			go func() {
				done <- clusterapp.RunNodeExecution(x.ctx, x.store, clusterapp.NodeExecutionPorts{
					Open: func(context.Context, string, string) (*workerhost.AttemptStore, workerhost.Runner, error) {
						return host, runner, nil
					},
					Authorize: x.app.Authorize, Cleanup: x.app.Cleanup, Clock: x.f.clock, Ready: func() { close(ready) },
				})
			}()
			t.Cleanup(func() { x.cancel(); check(t, <-done) })
			<-ready
			p, m := x.prepare(t)
			q := x.transportRequest(t, p, m)
			j, err := x.app.Transport(x.ctx, x.actor, m.Job, q)
			check(t, err)
			if start {
				_, err = x.app.Start(x.ctx, x.actor, nodeagent.StartCommand{IdempotencyKey: id(t), Job: m.Job, ExpectedVersion: j.Version})
				check(t, err)
				select {
				case <-runner.started:
				case <-time.After(5 * time.Second):
					t.Fatal("worker not started")
				}
			}
			f := x.f
			f.mu.Lock()
			f.snapshot.Payload.Generation++
			f.snapshot.Payload.Revocations = []trust.Revocation{{Kind: "principal", ID: q.Signed.Plan.Members[1].Binding.Principal, Mode: "compromise", Reason: "test", FirstGeneration: 2}}
			f.signSnapshot(t)
			f.mu.Unlock()
			deadline := time.Now().Add(5 * time.Second)
			for {
				available, err := x.store.Availability(f.clock())
				check(t, err)
				if available.Available {
					break
				}
				if time.Now().After(deadline) {
					t.Fatal("revoked relay left the key/lease live")
				}
				time.Sleep(10 * time.Millisecond)
			}
			out, err := x.store.JobStatus(x.actor, m.Job, f.clock())
			check(t, err)
			if out.State != nodeagent.JobCanceled {
				t.Fatal("revocation did not cancel the attempt", out)
			}
			if !start && runner.calls.Load() != 0 {
				t.Fatal("prepared-only attempt launched after revocation")
			}
		})
	}
}

func TestNodeExecutionTrustUsesCommittedHistoryAndFreshSnapshot(t *testing.T) {
	f := newTLSFixture(t)
	s, _, _, _ := capabilityNode(t, f)
	actor, err := trust.AuthenticatePrincipal(f.a, f.view, f.client.Chain, f.now)
	check(t, err)
	c, err := s.Capabilities(context.Background(), actor, f.now)
	check(t, err)
	_, _ = prepareRuntimeJob(t, f, s, c.Node)
	check(t, s.CheckExecutionTrust(context.Background(), f.a, f.view, f.now))
	if err := s.CheckExecutionTrust(context.Background(), f.a, f.view, f.now.Add(trust.SnapshotLifetime+time.Second)); err == nil {
		t.Fatal("stale trust authorized committed job")
	}
	f.snapshot.Payload.Generation++
	f.snapshot.Payload.Revocations = []trust.Revocation{{Kind: "principal", ID: f.clientID, Mode: "compromise", Reason: "test", FirstGeneration: 2}}
	f.signSnapshot(t)
	if err := s.CheckExecutionTrust(context.Background(), f.a, f.view, f.now); err == nil {
		t.Fatal("compromised controller history authorized")
	}
}
