package integration

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/distributed"
	"github.com/mrothroc/mixlab/grouptransport"
	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/workerhost"
	"github.com/mrothroc/mixlab/workerhost/contract"
	"github.com/mrothroc/mixlab/workerjob"
)

type runtimeTestRunner struct {
	calls   atomic.Int32
	finish  chan struct{}
	started chan struct{}
}

func (r *runtimeTestRunner) Run(ctx context.Context, _ contract.Approved, started func(int) error) error {
	r.calls.Add(1)
	if err := started(1234); err != nil {
		return err
	}
	close(r.started)
	select {
	case <-r.finish:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}
func (*runtimeTestRunner) Reconcile(context.Context, workerhost.Attempt) (bool, error) {
	return true, nil
}

func runtimePath(t *testing.T, kind statehome.Kind) statehome.Path {
	t.Helper()
	dir, err := filepath.EvalSymlinks(t.TempDir())
	check(t, err)
	check(t, os.Chmod(dir, 0700))
	p, err := statehome.Resolve(statehome.Options{ExactDir: dir}, statehome.Context{Kind: kind})
	check(t, err)
	return p
}

func prepareRuntimeJob(t *testing.T, f *tlsFixture, s *nodeagent.Store, node string) (trust.AuthenticatedPrincipal, nodeagent.Job) {
	t.Helper()
	ctx := context.Background()
	actor, err := trust.AuthenticatePrincipal(f.a, f.view, f.client.Chain, f.now)
	check(t, err)
	lease, err := s.Reserve(ctx, actor, nodeagent.Reserve{IdempotencyKey: id(t), ExpectedNodeVersion: 1, CapabilityGeneration: 1, Run: id(t), TTLSeconds: 300}, f.now)
	check(t, err)
	signed := signedJobFixture(t, f, node, lease)
	signed.Manifest.DatasetSelector = "toy"
	request, err := signed.Manifest.SigningRequest()
	check(t, err)
	signed.Proof, err = trust.SignPrincipalProof(f.a, f.client.Chain, f.client.Key, f.view, request, f.now)
	check(t, err)
	accepted, err := nodejob.Accept(f.a, f.view, actor, node, signed, f.now)
	check(t, err)
	j, err := s.PrepareJob(ctx, actor, accepted, lease.Version, f.now)
	check(t, err)
	m := signed.Manifest
	sp := signedTransportFixture(t, f, m)
	tp, err := grouptransport.Accept(f.a, f.view, actor, m, sp.Plan.Members[m.Rank].Chain, sp, f.now)
	check(t, err)
	j, err = s.ActivateTransport(ctx, actor, tp, j.Version, f.now)
	check(t, err)
	v, err := distributed.NewLocalGroupView(m.Membership, m.Members[m.Rank].MemberID, m.Rank, m.Attempt)
	check(t, err)
	a := workerjob.Assignment{Version: workerjob.Version, JobID: m.Job, AttemptID: m.Attempt, BuildID: m.BuildID, View: v, Config: m.Config, DatasetSelector: m.DatasetSelector, TrainPattern: "/unused/train.bin", DatasetSHA256: m.DatasetID, ProgramSHA256: m.ProgramHash, RuntimeSeconds: m.Limits.RuntimeSeconds, RingAddresses: [][]string{{"127.0.0.1:33000"}, {"127.0.0.1:33001"}}}
	a.WeightLayoutSHA256, a.OptimizerSHA256 = m.WeightLayoutHash, m.OptimizerHash
	_, err = s.StartJob(ctx, actor, nodeagent.StartCommand{IdempotencyKey: id(t), Job: j.ID, ExpectedVersion: j.Version}, a, f.now)
	check(t, err)
	j, err = s.JobStatus(actor, j.ID, f.now)
	check(t, err)
	return actor, j
}

func TestNodeLocalAssignmentUsesCatalogAndCommittedTransport(t *testing.T) {
	f := newTLSFixture(t)
	store, _, _, _ := capabilityNode(t, f)
	actor, err := trust.AuthenticatePrincipal(f.a, f.view, f.client.Chain, f.now)
	check(t, err)
	capabilities, err := store.Capabilities(context.Background(), actor, f.now)
	check(t, err)
	_, job := prepareRuntimeJob(t, f, store, capabilities.Node)
	a, err := store.ResolveAssignment(context.Background(), job.ID)
	check(t, err)
	if a.TrainPattern != "/private/local/train.bin" || a.RingAddresses[0][0] != "127.0.0.1:33000" || a.JobID != job.ID {
		t.Fatal("assignment did not resolve local policy", a)
	}
	a.Config[0] = '['
	again, err := store.ResolveAssignment(context.Background(), job.ID)
	check(t, err)
	if again.Config[0] != '{' {
		t.Fatal("caller mutated durable assignment")
	}
}

func TestNodeRuntimeExecutionAndCleanup(t *testing.T) {
	for _, mode := range []string{"success", "cancel", "expire", "revoked", "restart", "cleanup-failure"} {
		t.Run(mode, func(t *testing.T) {
			f := newTLSFixture(t)
			node := id(t)
			s, err := nodeagent.Initialize(context.Background(), runtimePath(t, statehome.Agent), f.a.Cluster(), node, 1)
			check(t, err)
			host, err := workerhost.NewAttemptStore(runtimePath(t, statehome.Worker))
			check(t, err)
			runner := &runtimeTestRunner{finish: make(chan struct{}), started: make(chan struct{})}
			var now atomic.Int64
			now.Store(f.now.Unix())
			var revoked, cleaned atomic.Bool
			ready := make(chan struct{})
			ports := clusterapp.NodeExecutionPorts{
				Open: func(context.Context, string, string) (*workerhost.AttemptStore, workerhost.Runner, error) {
					return host, runner, nil
				},
				Clock: func() time.Time { return time.Unix(now.Load(), 0) },
				Ready: func() { close(ready) },
				Authorize: func(context.Context, nodeagent.LocalExecution) error {
					if revoked.Load() {
						return errors.New("revoked")
					}
					return nil
				},
				Cleanup: func(ctx context.Context, q nodeagent.CleanupRequest) error {
					x, err := s.ActiveExecution(ctx)
					if err != nil {
						return err
					}
					if x == nil || x.Outcome == nil || !x.Outcome.NoChild || x.Job.ID != q.Job {
						return errors.New("cleanup preceded hosting confirmation")
					}
					if mode == "cleanup-failure" {
						return errors.New("key deletion failed")
					}
					cleaned.Store(true)
					return nil
				},
			}
			var actor trust.AuthenticatedPrincipal
			var job nodeagent.Job
			if mode == "restart" {
				_, job = prepareRuntimeJob(t, f, s, node)
			}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			done := make(chan error, 1)
			go func() { done <- clusterapp.RunNodeExecution(ctx, s, ports) }()
			select {
			case <-ready:
			case err := <-done:
				t.Fatal("runtime initialization failed", err)
			case <-time.After(5 * time.Second):
				t.Fatal("runtime not ready")
			}
			if mode != "restart" {
				actor, job = prepareRuntimeJob(t, f, s, node)
				select {
				case <-runner.started:
				case <-time.After(5 * time.Second):
					t.Fatal("worker not started")
				}
				job, err = s.JobStatus(actor, job.ID, f.now)
				check(t, err)
				if job.State != nodeagent.JobRunning {
					t.Fatal("start not applied", job)
				}
				switch mode {
				case "success", "cleanup-failure":
					close(runner.finish)
				case "cancel":
					_, err = s.CancelJob(ctx, actor, job.ID, job.Version, f.now)
					check(t, err)
				case "expire":
					now.Store(f.now.Add(301 * time.Second).Unix())
				case "revoked":
					revoked.Store(true)
				}
			}
			if mode == "cleanup-failure" {
				select {
				case err := <-done:
					if err == nil {
						t.Fatal("cleanup failure hidden")
					}
				case <-time.After(5 * time.Second):
					t.Fatal("cleanup failure did not stop runtime")
				}
				a, err := s.Availability(f.now)
				check(t, err)
				if a.Available || cleaned.Load() {
					t.Fatal("failed cleanup freed node")
				}
				return
			}
			deadline := time.Now().Add(5 * time.Second)
			for {
				a, err := s.Availability(f.now)
				check(t, err)
				if a.Available {
					break
				}
				if time.Now().After(deadline) {
					t.Fatal("node never released")
				}
				time.Sleep(10 * time.Millisecond)
			}
			if !cleaned.Load() {
				t.Fatal("lease released before key cleanup")
			}
			if mode == "restart" && runner.calls.Load() != 0 {
				t.Fatal("restarted interrupted job")
			}
			if mode != "restart" && runner.calls.Load() != 1 {
				t.Fatal("duplicate execution")
			}
			cancel()
			select {
			case err := <-done:
				check(t, err)
			case <-time.After(5 * time.Second):
				t.Fatal("runtime failed to stop")
			}
		})
	}
}
