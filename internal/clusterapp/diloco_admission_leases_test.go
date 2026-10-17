package clusterapp

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/admission"
	"github.com/mrothroc/mixlab/coordinator"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/recruitment"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/workercontrol"
	"github.com/mrothroc/mixlab/workerprobe"
)

type admissionLeaseFixture struct {
	mu       sync.Mutex
	now      atomic.Int64
	q        admission.Submission
	v        admission.Snapshot
	plan     run.Plan
	nodes    map[string]*admissionLeaseNode
	state    coordinator.State
	renewals []nodeagent.LeaseCommand
	roles    []trust.Role
	active   atomic.Int32
	maximum  atomic.Int32
	block    bool
}
type admissionLeaseNode struct {
	lease                     nodeagent.Lease
	reserved                  *nodeagent.Lease
	request                   nodeagent.Reserve
	lost, reject, unavailable bool
	reserves, reads           int
}
type admissionLeaseTestClient struct {
	f    *admissionLeaseFixture
	node string
	role trust.Role
}
type admissionLeaseRepository struct {
	coordinator.Repository
	f *admissionLeaseFixture
}

func (r admissionLeaseRepository) Load(context.Context, string) (coordinator.State, error) {
	return r.f.state, nil
}

func (c admissionLeaseTestClient) enter(ctx context.Context) error {
	deadline, ok := ctx.Deadline()
	if !ok || time.Until(deadline) > admissionLeaseCallBudget+time.Millisecond {
		return errors.New("unbounded lease RPC")
	}
	if c.f.block {
		n := c.f.active.Add(1)
		defer c.f.active.Add(-1)
		for old := c.f.maximum.Load(); n > old; old = c.f.maximum.Load() {
			if c.f.maximum.CompareAndSwap(old, n) {
				break
			}
		}
		<-ctx.Done()
		return ctx.Err()
	}
	return ctx.Err()
}
func (c admissionLeaseTestClient) Reserve(ctx context.Context, q nodeagent.Reserve) (nodeagent.Lease, error) {
	if err := c.enter(ctx); err != nil {
		return nodeagent.Lease{}, err
	}
	c.f.mu.Lock()
	defer c.f.mu.Unlock()
	n := c.f.nodes[c.node]
	n.reserves++
	if c.role != trust.Controller || q != n.request {
		return nodeagent.Lease{}, errors.New("reservation intent changed")
	}
	if n.reserved == nil {
		original := n.lease
		n.reserved = &original
	}
	return *n.reserved, nil
}
func (c admissionLeaseTestClient) LeaseStatus(ctx context.Context, id string) (nodeagent.Lease, error) {
	if err := c.enter(ctx); err != nil {
		return nodeagent.Lease{}, err
	}
	c.f.mu.Lock()
	defer c.f.mu.Unlock()
	n := c.f.nodes[c.node]
	n.reads++
	if n.unavailable || id != n.lease.ID {
		return nodeagent.Lease{}, errors.New("node unavailable")
	}
	if (n.lease.Coordinator != "") != (c.role == trust.Coordinator) {
		return nodeagent.Lease{}, errors.New("wrong lease owner")
	}
	return n.lease, nil
}
func (c admissionLeaseTestClient) Renew(ctx context.Context, q nodeagent.LeaseCommand) (nodeagent.Lease, error) {
	if err := c.enter(ctx); err != nil {
		return nodeagent.Lease{}, err
	}
	c.f.mu.Lock()
	defer c.f.mu.Unlock()
	n := c.f.nodes[c.node]
	l := n.lease
	now := c.f.now.Load()
	if n.reject {
		return nodeagent.Lease{}, errors.New("renew rejected")
	}
	if q.Lease != l.ID || q.ExpectedVersion != l.Version || now >= l.Expires || (l.Coordinator != "") != (c.role == trust.Coordinator) || l.State == nodeagent.Releasing || l.State == nodeagent.Released {
		return nodeagent.Lease{}, errors.New("stale or foreign renewal")
	}
	l.Expires = now + int64(q.TTLSeconds)
	if l.JobKind == "learner" && l.Coordinator == "" {
		l.Expires = min(l.Expires, c.f.v.Deadline)
	}
	if l.WorkloadDeadline != 0 {
		l.Expires = min(l.Expires, l.WorkloadDeadline)
	}
	if l.Expires <= n.lease.Expires {
		return nodeagent.Lease{}, errors.New("renewal did not extend")
	}
	l.RenewBy = now + (l.Expires-now)/2
	l.Version++
	n.lease = l
	c.f.renewals = append(c.f.renewals, q)
	c.f.roles = append(c.f.roles, c.role)
	if n.lost {
		n.lost = false
		return nodeagent.Lease{}, errors.New("lost renewal ACK")
	}
	return l, nil
}

func admissionLeaseSetup(t *testing.T, count int) (*DiLoCoAdmissionLeases, *admissionLeaseFixture) {
	t.Helper()
	f := &admissionLeaseFixture{nodes: map[string]*admissionLeaseNode{}}
	f.now.Store(1000)
	b, err := os.ReadFile("../../diloco/testdata/plan.json")
	check(t, err)
	var template run.Plan
	check(t, json.Unmarshal(b, &template))
	id := func(n int) string { return fmt.Sprintf("%032x", n) }
	config := json.RawMessage(`{"model_dim":16}`)
	template.Contract.ConfigHash = nodejob.Hash(config)
	template.Coordinator = run.Coordinator{ClusterID: id(1), PrincipalID: id(2), Role: "coordinator", Audience: "learner-run"}
	limits := nodejob.Limits{RuntimeSeconds: 3600, CPUSeconds: 3600, MemoryBytes: 1 << 30, DiskBytes: 1 << 30, LogBytes: 1 << 20}
	requirements := recruitment.Requirements{Cluster: id(1), Count: count, BuildID: template.Contract.BuildHash, MLXVersion: "test", DeviceKind: "metal", DType: "fp32", CustomOps: []string{}, DatasetSelector: "toy", DatasetID: template.Contract.DatasetHash, Limits: limits, ProbeMaxAge: time.Minute}
	f.q = admission.Submission{Owner: id(3), Deadline: 2800, Specification: admission.Specification{Contract: template.Contract, BaseModel: template.BaseModel, Outer: template.Outer, Coordinator: template.Coordinator, ArtifactStoreID: template.ArtifactStoreID, Endpoint: template.Endpoint, Config: config, CreatedAt: 1000}}
	f.q.Launch = recruitment.LaunchPlan{Cluster: id(1), Controller: id(4), Run: id(5), Attempt: id(6), TTLSeconds: 300, Selection: recruitment.Selection{Policy: recruitment.SelectionPolicy, Requirements: requirements}}
	var leases []nodeagent.Lease
	for index := range count {
		node := id(index + 100)
		probe := workerprobe.Report{Version: workerprobe.Version, BuildID: template.Contract.BuildHash, BuildVersion: "test", WorkerProtocol: workercontrol.Version, OS: "darwin", Arch: "arm64", Available: true, MLXVersion: "test", MLXSupported: true, DeviceKind: "metal", DeviceName: "synthetic", Backends: []string{"ring"}, DTypes: []string{"fp32"}, CustomOps: []string{}}
		caps := nodeagent.Capabilities{Version: nodeagent.CapabilityVersion, Node: node, DisplayName: "lease-test", Generation: 1, ObservedAt: 1000, ProbeObservedAt: 1000, Limits: limits, ManifestVersions: []string{nodejob.LearnerVersion}, Datasets: []nodeagent.Dataset{{Selector: "toy", ID: template.Contract.DatasetHash}}, Availability: nodeagent.Availability{NodeVersion: 1, CapabilityGeneration: 1, Available: true}, Recruitable: true, Probe: probe}
		f.q.Launch.Selection.Selected = append(f.q.Launch.Selection.Selected, recruitment.Candidate{Endpoint: fmt.Sprintf("node%d.local:7445", index), Capabilities: caps})
		h, err := run.Digest(struct{ Run, Attempt, Node string }{f.q.Launch.Run, f.q.Launch.Attempt, node})
		check(t, err)
		request := nodeagent.Reserve{IdempotencyKey: h[:32], ExpectedNodeVersion: 1, CapabilityGeneration: 1, Run: f.q.Launch.Run, TTLSeconds: 300}
		l := nodeagent.Lease{Format: nodeagent.LeaseVersion, ID: id(index + 200), Node: node, Controller: f.q.Launch.Controller, Run: f.q.Launch.Run, CapabilityGeneration: 1, Version: 1, State: nodeagent.Reserved, Created: 1000, Expires: 1300, RenewBy: 1150}
		leases = append(leases, l)
		copy := l
		f.nodes[node] = &admissionLeaseNode{lease: l, request: request, reserved: &copy}
		f.v.Nodes = append(f.v.Nodes, admission.NodeSnapshot{Reserve: request, Touched: true, Lease: &copy})
	}
	f.v.Request, f.v.Specification, f.v.Deadline = f.q.Launch, f.q.Specification, f.q.Deadline
	check(t, f.q.Validate())
	f.plan, err = buildDiLoCoPlan(f.q, leases)
	check(t, err)
	k := &DiLoCoAdmissionLeases{Repository: admissionLeaseRepository{f: f}, Clock: func() time.Time { return time.Unix(f.now.Load(), 0) }, Client: func(role trust.Role, c recruitment.Candidate) (AdmissionLeaseClient, error) {
		return admissionLeaseTestClient{f, c.Capabilities.Node, role}, nil
	}}
	return k, f
}
func (f *admissionLeaseFixture) prepared(index int) {
	f.v.Plan = &f.plan
	n := f.nodes[f.q.Launch.Selection.Selected[index].Capabilities.Node]
	n.lease.State = nodeagent.Prepared
	n.lease.JobKind = "learner"
	n.lease.Job = f.plan.Slots[index].Allocation.JobID
	n.lease.Version++
}
func (f *admissionLeaseFixture) committed(t *testing.T) {
	t.Helper()
	f.v.Plan = &f.plan
	p := f.plan
	b, err := os.ReadFile("../../diloco/testdata/registration.json")
	check(t, err)
	var template run.Registration
	check(t, json.Unmarshal(b, &template))
	c := run.Commit{Format: run.CommitVersion, Binding: run.Binding{RunID: p.Cohort.RunID, PlanHash: p.SHA256, Generation: p.Cohort.Generation, CohortHash: p.Cohort.CohortHash}, Coordinator: p.Coordinator, CommittedAt: 1100}
	for index, slot := range p.Slots {
		f.prepared(index)
		r := template
		r.RunID = p.Cohort.RunID
		r.PlanHash = p.SHA256
		r.Generation = p.Cohort.Generation
		r.CohortHash = p.Cohort.CohortHash
		r.Slot = slot
		r.Contract = p.Contract
		r.Coordinator = p.Coordinator
		c.Registrations = append(c.Registrations, r)
	}
	c.Binding.CommitHash, err = c.Digest()
	check(t, err)
	check(t, c.ValidateFor(p))
	f.v.Commit = &c
	f.v.Committed = true
	momentum := p.BaseModel
	momentum.Kind = "outer_state_f32"
	f.state = coordinator.State{Revision: 1, Phase: coordinator.Ready, Plan: p, Commit: &c, Base: p.BaseModel, Momentum: momentum, PrivateHashes: make([]string, len(p.Slots)), Policy: coordinator.EvaluationPolicy{WorkerID: p.Slots[0].Member.WorkerID, ValidationHash: p.Contract.DatasetHash, Split: "validation", BatchTokens: 4, SequenceLength: 4, Batches: 1, TimeoutSeconds: 60}}
	check(t, f.state.Validate())
}
func (f *admissionLeaseFixture) handedOff(index int, acknowledged bool) {
	n := f.nodes[f.q.Launch.Selection.Selected[index].Capabilities.Node]
	n.lease.Coordinator = f.plan.Coordinator.PrincipalID
	n.lease.RunCommitHash = f.v.Commit.Binding.CommitHash
	n.lease.WorkloadDeadline = 6400
	n.lease.State = nodeagent.Running
	n.lease.Version++
	f.v.Nodes[index].Finalized = acknowledged
}

func TestAdmissionLeasesPrevent64NodeTickStarvation(t *testing.T) {
	k, f := admissionLeaseSetup(t, 64)
	ctx := context.Background()
	for index := range f.v.Nodes {
		f.v.Nodes[index].Touched = false
		f.v.Nodes[index].Lease = nil
	}
	for effect := range 5*64 + 5 {
		check(t, k.Maintain(ctx, f.q, f.v))
		switch {
		case effect < 128:
			index := effect / 2
			if effect%2 == 0 {
				f.v.Nodes[index].Touched = true
			} else {
				n := f.nodes[f.q.Launch.Selection.Selected[index].Capabilities.Node]
				l := *n.reserved
				f.v.Nodes[index].Lease = &l
			}
		case effect < 128+3*64:
			if (effect-128)%3 == 0 {
				f.prepared((effect - 128) / 3)
			}
		}
		f.now.Add(1)
	}
	if f.now.Load() <= 1300 || len(f.renewals) == 0 {
		t.Fatal("test did not cross original TTL")
	}
	for _, n := range f.nodes {
		if n.lease.Expires <= f.now.Load() {
			t.Fatal("one-effect-per-tick starved original allocation")
		}
	}
}

func TestAdmissionLeasesCurrentVersionAndLostACK(t *testing.T) {
	k, f := admissionLeaseSetup(t, 2)
	n := f.nodes[f.q.Launch.Selection.Selected[0].Capabilities.Node]
	n.lease.Version = 9
	n.lost = true
	f.now.Store(1150)
	check(t, k.Maintain(context.Background(), f.q, f.v))
	commands := f.commands(n.lease.ID)
	if len(commands) != 1 || commands[0].ExpectedVersion != 9 || n.reads != 2 {
		t.Fatalf("renew did not resolve lost ACK against current version: %+v reads=%d", f.renewals, n.reads)
	}
	check(t, k.Maintain(context.Background(), f.q, f.v))
	if len(f.commands(n.lease.ID)) != 1 {
		t.Fatal("retry duplicated accepted renewal")
	}
	first := commands[0].IdempotencyKey
	f.now.Store(1300)
	check(t, k.Maintain(context.Background(), f.q, f.v))
	commands = f.commands(n.lease.ID)
	if len(commands) != 2 || commands[1].ExpectedVersion != 10 || commands[1].IdempotencyKey == first {
		t.Fatal("next renewal reused stale command")
	}
}

func TestAdmissionLeasesPartialHandoffUsesLiveOwner(t *testing.T) {
	k, f := admissionLeaseSetup(t, 3)
	f.committed(t)
	f.handedOff(0, true)
	f.handedOff(1, false)
	f.now.Store(1150)
	before, err := json.Marshal(f.v)
	check(t, err)
	check(t, k.Maintain(context.Background(), f.q, f.v))
	count := map[trust.Role]int{}
	for _, role := range f.roles {
		count[role]++
	}
	if count[trust.Coordinator] != 2 || count[trust.Controller] != 1 {
		t.Fatal("renewal used saga flag rather than live owner", count)
	}
	after, err := json.Marshal(f.v)
	check(t, err)
	if !bytes.Equal(before, after) || f.state.Phase != coordinator.Ready || f.v.Done {
		t.Fatal("maintenance changed admission or opened rounds")
	}
}

func TestAdmissionLeasesFailedNodeDoesNotStarveOthers(t *testing.T) {
	k, f := admissionLeaseSetup(t, 3)
	f.committed(t)
	f.handedOff(1, true)
	f.now.Store(1150)
	f.nodes[f.q.Launch.Selection.Selected[0].Capabilities.Node].unavailable = true
	if err := k.Maintain(context.Background(), f.q, f.v); err == nil {
		t.Fatal("unavailable node hidden")
	}
	if len(f.renewals) != 2 {
		t.Fatal("unavailable first node starved healthy leases", len(f.renewals))
	}
}

func TestAdmissionLeasesRejectExpiredTerminalAndForeign(t *testing.T) {
	for _, problem := range []string{"expired", "releasing", "released", "wrong_run", "wrong_node", "wrong_controller", "wrong_job", "wrong_lease", "rollback", "wrong_coordinator", "wrong_commit", "failed_run", "expired_admission"} {
		t.Run(problem, func(t *testing.T) {
			k, f := admissionLeaseSetup(t, 2)
			f.prepared(0)
			f.now.Store(1150)
			n := f.nodes[f.q.Launch.Selection.Selected[0].Capabilities.Node]
			switch problem {
			case "expired":
				n.lease.Expires = 1150
			case "releasing":
				n.lease.State = nodeagent.Releasing
			case "released":
				n.lease.State = nodeagent.Released
			case "wrong_run":
				n.lease.Run = strings.Repeat("f", 32)
			case "wrong_node":
				n.lease.Node = strings.Repeat("f", 32)
			case "wrong_controller":
				n.lease.Controller = strings.Repeat("f", 32)
			case "wrong_job":
				n.lease.Job = strings.Repeat("f", 32)
			case "wrong_lease":
				f.v.Nodes[0].Lease.ID = strings.Repeat("f", 32)
			case "rollback":
				f.v.Nodes[0].Lease.Version = 99
			case "wrong_coordinator":
				f.committed(t)
				f.handedOff(0, true)
				n.lease.Coordinator = strings.Repeat("f", 32)
			case "wrong_commit":
				f.committed(t)
				f.handedOff(0, true)
				n.lease.RunCommitHash = strings.Repeat("f", 64)
			case "failed_run":
				f.committed(t)
				f.state.Phase = coordinator.Failed
				f.state.Failure = "terminal"
			case "expired_admission":
				f.now.Store(f.v.Deadline)
			}
			if err := k.Maintain(context.Background(), f.q, f.v); err == nil {
				t.Fatal("invalid lease accepted")
			}
			if len(f.commands(n.lease.ID)) != 0 {
				t.Fatal("invalid lease renewed")
			}
		})
	}
}

func TestAdmissionLeasesDeadlineCapAndCompensation(t *testing.T) {
	k, f := admissionLeaseSetup(t, 2)
	f.prepared(0)
	n := f.nodes[f.q.Launch.Selection.Selected[0].Capabilities.Node]
	f.now.Store(2650)
	for _, node := range f.nodes {
		node.lease.Expires = 2700
		node.lease.RenewBy = 2600
	}
	check(t, k.Maintain(context.Background(), f.q, f.v))
	if n.lease.Expires != f.v.Deadline || f.commands(n.lease.ID)[0].TTLSeconds != 150 {
		t.Fatal("renewal exceeded admission deadline")
	}
	f.now.Store(2750)
	check(t, k.Maintain(context.Background(), f.q, f.v))
	if len(f.commands(n.lease.ID)) != 1 {
		t.Fatal("attempted impossible renewal at deadline cap")
	}
	f.v.Aborting = true
	f.now.Store(3000)
	check(t, k.Maintain(context.Background(), f.q, f.v))
	if len(f.commands(n.lease.ID)) != 1 {
		t.Fatal("compensation renewed allocations")
	}
}

func TestAdmissionLeasesBoundedConcurrencyAndCancellation(t *testing.T) {
	k, f := admissionLeaseSetup(t, 64)
	f.block = true
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	started := time.Now()
	if err := k.Maintain(ctx, f.q, f.v); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatal("lost cancellation", err)
	}
	if f.maximum.Load() > admissionLeaseConcurrency || f.maximum.Load() == 0 || f.active.Load() != 0 {
		t.Fatal("unbounded or unjoined RPCs", f.maximum.Load(), f.active.Load())
	}
	if time.Since(started) > time.Second {
		t.Fatal("maintenance exceeded caller budget")
	}
}

func TestAdmissionLeasesAmbiguousReserveReplaysIntent(t *testing.T) {
	k, f := admissionLeaseSetup(t, 2)
	f.v.Nodes[0].Lease = nil
	f.now.Store(1150)
	n := f.nodes[f.q.Launch.Selection.Selected[0].Capabilities.Node]
	check(t, k.Maintain(context.Background(), f.q, f.v))
	check(t, k.Maintain(context.Background(), f.q, f.v))
	if n.reserves != 2 || len(f.commands(n.lease.ID)) != 1 || f.v.Nodes[0].Lease != nil {
		t.Fatal("ambiguous reserve replaced intent or bypassed live status")
	}
	f.v.Nodes[0].Reserve.IdempotencyKey = strings.Repeat("f", 32)
	if err := k.Maintain(context.Background(), f.q, f.v); !errors.Is(err, ErrAdmissionLeaseLost) {
		t.Fatal("changed reserve intent accepted", err)
	}
	if n.reserves != 2 {
		t.Fatal("changed intent sent to node")
	}
}

func (f *admissionLeaseFixture) commands(id string) []nodeagent.LeaseCommand {
	var out []nodeagent.LeaseCommand
	for _, q := range f.renewals {
		if q.Lease == id {
			out = append(out, q)
		}
	}
	return out
}
