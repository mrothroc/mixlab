package clusterapp

import (
	"context"
	"errors"
	"reflect"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/admission"
	"github.com/mrothroc/mixlab/coordinator"
	coordinatorlocal "github.com/mrothroc/mixlab/coordinator/local"
	"github.com/mrothroc/mixlab/coordinator/nodeleases"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/trust/principal"
)

type terminalLeaseNodes struct {
	leases                    map[string]nodeagent.Lease
	statusErr, releaseErr     error
	reads, renewals, releases int
}

func (n *terminalLeaseNodes) Status(_ context.Context, a run.Allocation) (nodeagent.Lease, error) {
	n.reads++
	return n.leases[a.LeaseID], n.statusErr
}

func (n *terminalLeaseNodes) Renew(context.Context, run.Allocation, nodeagent.LeaseCommand) (nodeagent.Lease, error) {
	n.renewals++
	return nodeagent.Lease{}, errors.New("unexpected renewal")
}

func (n *terminalLeaseNodes) Release(_ context.Context, a run.Allocation, q nodeagent.LeaseCommand) (nodeagent.Lease, error) {
	n.releases++
	l := n.leases[a.LeaseID]
	if q.ExpectedVersion != l.Version || q.Lease != l.ID || q.TTLSeconds != 0 {
		return nodeagent.Lease{}, errors.New("release changed allocation or version")
	}
	if l.State != nodeagent.Releasing {
		l.State = nodeagent.Releasing
		l.Version++
		n.leases[a.LeaseID] = l
	}
	// Even a successful release response is only an intent, not physical death.
	return l, n.releaseErr
}

func lifecycleModels(f *admissionLeaseFixture, repo *recoveryTestRepository, nodes *terminalLeaseNodes) *DiLoCoModels {
	artifacts := coordinatorlocal.Artifacts{}
	clock := serviceClock{func() time.Time { return time.Unix(f.now.Load(), 0) }}
	model := coordinator.Model{Repository: repo, Artifacts: artifacts, Clock: clock, Evaluator: remoteEvaluation{}}
	keeper := nodeleases.Keeper{Repository: repo, Nodes: nodes, Clock: clock, RunID: f.q.Launch.Run, TTLSeconds: 300}
	model.Allocations = &keeper
	keeper.Fail = model.Fail
	models := &DiLoCoModels{Inbox: &admission.Inbox{}, Principal: &principal.Store{}, Repository: repo, Artifacts: artifacts, Clock: clock.now}
	models.cache.Store(f.q.Launch.Run, composedRun{model, keeper})
	return models
}

func lifecycleLeases(f *admissionLeaseFixture) *terminalLeaseNodes {
	nodes := &terminalLeaseNodes{leases: map[string]nodeagent.Lease{}}
	for i, slot := range f.plan.Slots {
		f.handedOff(i, true)
		nodes.leases[slot.Allocation.LeaseID] = f.nodes[slot.Allocation.NodeID].lease
	}
	return nodes
}

func TestDiLoCoModelsTerminalReleaseConfirmation(t *testing.T) {
	for _, phase := range []coordinator.Phase{coordinator.Failed, coordinator.Complete} {
		t.Run(string(phase), func(t *testing.T) {
			_, f := admissionLeaseSetup(t, 2)
			f.committed(t)
			f.now.Store(1100)
			repo := &recoveryTestRepository{state: f.state}
			repo.state.Phase = phase
			if phase == coordinator.Complete {
				repo.state.Round = 1
				for i := range repo.state.PrivateHashes {
					repo.state.PrivateHashes[i] = f.plan.Contract.ConfigHash
				}
			} else {
				repo.state.Failure = "terminal fixture"
			}
			check(t, repo.state.Validate())
			before := repo.state
			nodes := lifecycleLeases(f)
			models := lifecycleModels(f, repo, nodes)
			recoverCalls := 0
			models.Recover = func(context.Context, admission.Submission, coordinator.Model) (bool, error) {
				recoverCalls++
				return false, errors.New("terminal recovery must not run")
			}
			tick := func() error {
				return models.Tick(context.Background(), f.q, admission.Snapshot{Done: true, Committed: true})
			}
			// An unavailable node cannot be treated as released.
			nodes.statusErr = context.DeadlineExceeded
			if err := tick(); !errors.Is(err, context.DeadlineExceeded) {
				t.Fatal("cleanup outage hidden", err)
			}
			nodes.statusErr = nil
			// Lost release ACK must be retried, even though the node accepted intent.
			nodes.releaseErr = context.DeadlineExceeded
			if err := tick(); !errors.Is(err, context.DeadlineExceeded) {
				t.Fatal("ambiguous release hidden", err)
			}
			nodes.releaseErr = nil
			check(t, tick())
			releases := nodes.releases
			// Confirming only one allocation is insufficient.
			first := f.plan.Slots[0].Allocation.LeaseID
			lease := nodes.leases[first]
			lease.State = nodeagent.Released
			nodes.leases[first] = lease
			check(t, tick())
			if nodes.releases <= releases {
				t.Fatal("skipped second allocation cleanup")
			}
			for id, lease := range nodes.leases {
				lease.State = nodeagent.Released
				nodes.leases[id] = lease
			}
			check(t, tick())
			reads, releases := nodes.reads, nodes.releases
			nodes.statusErr = errors.New("node went offline after confirmed release")
			for range 3 {
				check(t, tick())
			}
			if nodes.reads != reads || nodes.releases != releases || nodes.renewals != 0 || recoverCalls != 0 || !reflect.DeepEqual(before, repo.state) {
				t.Fatal("terminal run kept polling or changed state")
			}
			// Restart loses only the optimization: re-observe release, never assume it.
			models = lifecycleModels(f, repo, nodes)
			if err := tick(); err == nil || nodes.reads == reads {
				t.Fatal("restart assumed unobserved cleanup")
			}
			nodes.statusErr = nil
			check(t, tick())
			reads = nodes.reads
			check(t, tick())
			if nodes.reads != reads {
				t.Fatal("restart did not quiesce confirmed cleanup")
			}
		})
	}
}

func TestDiLoCoModelsKeeperFailureSkipsRecoveryButReleases(t *testing.T) {
	_, f := admissionLeaseSetup(t, 2)
	f.committed(t)
	f.now.Store(1100)
	repo := &recoveryTestRepository{state: f.state}
	nodes := lifecycleLeases(f)
	id := f.plan.Slots[0].Allocation.LeaseID
	lease := nodes.leases[id]
	lease.Expires = f.now.Load()
	nodes.leases[id] = lease
	models := lifecycleModels(f, repo, nodes)
	models.Recover = func(context.Context, admission.Submission, coordinator.Model) (bool, error) {
		t.Fatal("keeper failed run but polled recovery")
		return false, nil
	}
	check(t, models.Tick(context.Background(), f.q, admission.Snapshot{}))
	if repo.state.Phase != coordinator.Failed {
		t.Fatal("expired lease did not fail run")
	}
	check(t, models.Tick(context.Background(), f.q, admission.Snapshot{}))
	if nodes.releases != 2 || nodes.renewals != 0 {
		t.Fatal("terminal transition skipped allocation release")
	}
}
