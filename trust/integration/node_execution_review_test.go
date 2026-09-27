package integration

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/distributed"
	"github.com/mrothroc/mixlab/grouptransport"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/workerhost"
	"github.com/mrothroc/mixlab/workerjob"
)

func TestNodeStartAdmissionCannotSurviveCancellationOrDeadline(t *testing.T) {
	for _, scenario := range []string{"activate-expired", "start-expired", "retry-expired", "retry-canceled"} {
		t.Run(scenario, func(t *testing.T) {
			ctx := context.Background()
			f := newTLSFixture(t)
			node := id(t)
			dir, err := filepath.EvalSymlinks(t.TempDir())
			check(t, err)
			check(t, os.Chmod(dir, 0700))
			p, err := statehome.Resolve(statehome.Options{ExactDir: dir}, statehome.Context{Kind: statehome.Agent})
			check(t, err)
			s, err := nodeagent.Initialize(ctx, p, f.a.Cluster(), node, 1)
			check(t, err)
			actor, err := trust.AuthenticatePrincipal(f.a, f.view, f.client.Chain, f.now)
			check(t, err)
			lease, err := s.Reserve(ctx, actor, nodeagent.Reserve{IdempotencyKey: id(t), ExpectedNodeVersion: 1, CapabilityGeneration: 1, Run: id(t), TTLSeconds: 300}, f.now)
			check(t, err)
			signed := signedJobFixture(t, f, node, lease)
			signed.Manifest.Expires = f.now.Add(5 * time.Second).Unix()
			req, err := signed.Manifest.SigningRequest()
			check(t, err)
			signed.Proof, err = trust.SignPrincipalProof(f.a, f.client.Chain, f.client.Key, f.view, req, f.now)
			check(t, err)
			accepted, err := nodejob.Accept(f.a, f.view, actor, node, signed, f.now)
			check(t, err)
			j, err := s.PrepareJob(ctx, actor, accepted, lease.Version, f.now)
			check(t, err)
			m := signed.Manifest
			sp := signedTransportFixture(t, f, m)
			tp, err := grouptransport.Accept(f.a, f.view, actor, m, sp.Plan.Members[m.Rank].Chain, sp, f.now)
			check(t, err)
			expired := f.now.Add(6 * time.Second)
			if scenario == "activate-expired" {
				if _, err := s.ActivateTransport(ctx, actor, tp, j.Version, expired); err == nil {
					t.Fatal("activated expired manifest")
				}
				return
			}
			j, err = s.ActivateTransport(ctx, actor, tp, j.Version, f.now)
			check(t, err)
			view, err := distributed.NewLocalGroupView(m.Membership, m.Members[m.Rank].MemberID, m.Rank, m.Attempt)
			check(t, err)
			a := workerjob.Assignment{Version: workerjob.Version, JobID: m.Job, AttemptID: m.Attempt, BuildID: m.BuildID, View: view, Config: m.Config, DatasetSelector: m.DatasetSelector, TrainPattern: "/unused/train.bin", DatasetSHA256: m.DatasetID, ProgramSHA256: m.ProgramHash, RuntimeSeconds: m.Limits.RuntimeSeconds, RingAddresses: [][]string{{"127.0.0.1:33000"}, {"127.0.0.1:33001"}}}
			a.WeightLayoutSHA256, a.OptimizerSHA256 = m.WeightLayoutHash, m.OptimizerHash
			q := nodeagent.StartCommand{IdempotencyKey: id(t), Job: j.ID, ExpectedVersion: j.Version}
			if scenario == "start-expired" {
				if _, err := s.StartJob(ctx, actor, q, a, expired); err == nil {
					t.Fatal("started expired manifest")
				}
				return
			}
			_, err = s.StartJob(ctx, actor, q, a, f.now)
			check(t, err)
			if scenario == "retry-expired" {
				if _, err := s.StartJob(ctx, actor, q, a, expired); err == nil {
					t.Fatal("expired approval replayed")
				}
				return
			}
			j, err = s.JobStatus(actor, j.ID, f.now)
			check(t, err)
			j, err = s.CancelJob(ctx, actor, j.ID, j.Version, f.now)
			check(t, err)
			if _, err := s.StartJob(ctx, actor, q, a, f.now); err == nil {
				t.Fatal("canceled approval replayed")
			}
			approval, err := s.CancellationApproval(ctx, j.ID)
			check(t, err)
			workerDir, err := filepath.EvalSymlinks(t.TempDir())
			check(t, err)
			check(t, os.Chmod(workerDir, 0700))
			wp, err := statehome.Resolve(statehome.Options{ExactDir: workerDir}, statehome.Context{Kind: statehome.Worker})
			check(t, err)
			host, err := workerhost.NewAttemptStore(wp)
			check(t, err)
			outcome, err := host.FenceApproved(ctx, approval)
			check(t, err)
			_, err = s.ApplyOutcome(ctx, outcome)
			check(t, err)
			cleanupCtx, cancel := context.WithTimeout(ctx, time.Second)
			defer cancel()
			check(t, s.CompleteCleanup(cleanupCtx, j.ID, func(ctx context.Context, _ nodeagent.CleanupRequest) error {
				// Re-entering the node store must not deadlock behind cleanup.
				return s.Expire(ctx, f.now)
			}))
			availability, err := s.Availability(f.now)
			check(t, err)
			if !availability.Available {
				t.Fatal("canceled start kept lease")
			}
		})
	}
}
