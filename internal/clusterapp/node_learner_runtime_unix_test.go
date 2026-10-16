//go:build darwin || linux

package clusterapp

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	private "github.com/mrothroc/mixlab/diloco/learner"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/distributed"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/enrollment"
	"github.com/mrothroc/mixlab/trust/enrollment/enrollee"
	"github.com/mrothroc/mixlab/trust/principal"
	"github.com/mrothroc/mixlab/trust/workload"
	wc "github.com/mrothroc/mixlab/workercontrol"
	"github.com/mrothroc/mixlab/workerprobe"
)

// All identity material is generated in test-owned file stores. The returned
// relay deliberately has no workload credential directory or process runtime:
// precommit polling and fencing must complete before either is consulted.
func learnerRuntimeRemote(t *testing.T) *nodeLearnerRemote {
	t.Helper()
	return newLearnerRuntimeFixture(t).remote
}

type learnerRuntimeFixture struct {
	remote      *nodeLearnerRemote
	authority   *Authority
	controller  *principal.Store
	coordinator *principal.Store
	ports       nodeagent.WorkloadPorts
}

func newLearnerRuntimeFixture(t *testing.T) learnerRuntimeFixture {
	t.Helper()
	ctx := context.Background()
	p := initialized(t)
	check(t, InitializeAuthority(ctx, p, testNow))
	a, err := OpenAuthority(ctx, p, testNow)
	check(t, err)
	t.Cleanup(func() { check(t, a.Close()) })
	v, err := a.Current(ctx, testNow)
	check(t, err)
	i, err := a.Enrollment.Invite(ctx, enrollment.NodeEnrollment, "https://authority.example:7443", "authority", time.Minute, v, testNow)
	check(t, err)
	defer i.Clear()
	stage, final := enrolleePaths(t, p)
	candidate, err := enrollee.Prepare(ctx, stage, final, a.Anchor, trust.Node, "file", testNow)
	check(t, err)
	defer func() { check(t, candidate.Close()) }()
	k, envelope, err := candidate.Keys()
	check(t, err)
	q, err := enrollment.NewRequest(i, k, envelope, testNow)
	check(t, err)
	result, err := a.Enrollment.Consume(ctx, q, i.Secret, v, testNow)
	check(t, err)
	check(t, candidate.CompleteProvisioned(ctx, i, q, result, testNow))
	node, err := principal.Open(final, testNow)
	check(t, err)
	t.Cleanup(func() { check(t, node.Close()) })
	resolve := func(name string, kind statehome.Kind) statehome.Path {
		path, err := statehome.Resolve(statehome.Options{ExactDir: filepath.Join(filepath.Dir(p.Dir()), name)}, statehome.Context{Kind: kind})
		check(t, err)
		return path
	}
	controller, err := principal.Open(resolve("controller", statehome.Principal), testNow)
	check(t, err)
	t.Cleanup(func() { check(t, controller.Close()) })
	coordinator, err := principal.Open(resolve("coordinator", statehome.Principal), testNow)
	check(t, err)
	t.Cleanup(func() { check(t, coordinator.Close()) })
	coordinatorIdentity, _, err := coordinator.Active(testNow)
	check(t, err)
	coordinatorActor, err := trust.AuthenticatePrincipal(a.Anchor, v, coordinatorIdentity.Chain, testNow)
	check(t, err)
	identity, signer, err := controller.Active(testNow)
	check(t, err)
	actor, err := trust.AuthenticatePrincipal(a.Anchor, v, identity.Chain, testNow)
	check(t, err)
	id := func() string { value, err := controllerRandomID(); check(t, err); return value }
	storePath := resolve("learner-agent", statehome.Agent)
	check(t, storePath.Ensure())
	s, err := nodeagent.Initialize(ctx, storePath, a.Anchor.Cluster(), result.Approval.Principal, 1)
	check(t, err)
	hash := strings.Repeat("a", 64)
	limits := nodejob.Limits{RuntimeSeconds: 60, CPUSeconds: 120, MemoryBytes: 1 << 30, DiskBytes: 1 << 30, LogBytes: 1 << 20}
	probe := workerprobe.Report{Version: workerprobe.Version, BuildID: hash, BuildVersion: "test", WorkerProtocol: wc.Version, OS: "darwin", Arch: "arm64", Available: true, MLXVersion: "test", MLXSupported: true, DeviceKind: "metal", DeviceName: "synthetic", Backends: []string{}, DTypes: []string{"fp32"}, CustomOps: []string{}}
	check(t, s.InstallProfile(ctx, nodeagent.Profile{Version: nodeagent.ProfileVersion, Node: result.Approval.Principal, DisplayName: "runtime-test", Generation: 1, Probe: probe, ProbeObservedAt: testNow.Unix(), Limits: limits, Datasets: []nodeagent.LocalDataset{{Dataset: nodeagent.Dataset{Selector: "toy", ID: hash}, TrainPattern: "/synthetic/train.bin"}}}))
	caps, err := s.Capabilities(ctx, actor, testNow)
	check(t, err)
	lease, err := s.Reserve(ctx, actor, nodeagent.Reserve{IdempotencyKey: id(), ExpectedNodeVersion: 1, CapabilityGeneration: 1, Run: id(), TTLSeconds: 300}, testNow)
	check(t, err)
	var plan run.Plan
	b, err := os.ReadFile("../../diloco/testdata/plan.json")
	check(t, err)
	check(t, json.Unmarshal(b, &plan))
	member := distributed.LearnerMember{WorkerID: id(), ParticipantID: caps.Node, IslandID: "metal", DatasetPartitionID: "partition0"}
	plan.Cohort, err = distributed.NewLearnerCohort(lease.Run, 1, []distributed.LearnerMember{member})
	check(t, err)
	allocation := run.Allocation{Provider: "node_agent", NodeID: caps.Node, LeaseID: lease.ID, JobID: id()}
	allocation.RefHash, err = run.Digest(allocation)
	check(t, err)
	capHash, err := run.Digest(caps)
	check(t, err)
	plan.Slots = []run.Slot{{Member: member, CapabilityHash: capHash, Allocation: allocation}}
	plan.Coordinator = run.Coordinator{ClusterID: a.Anchor.Cluster(), PrincipalID: coordinatorActor.Principal, Role: "coordinator", Audience: "learner-runtime"}
	plan.CreatedAt, plan.ExpiresAt = testNow.Unix(), testNow.Add(time.Minute).Unix()
	config := json.RawMessage(`{"model_dim":16}`)
	plan.Contract.ConfigHash = nodejob.Hash(config)
	plan.SHA256, err = plan.Digest()
	check(t, err)
	m := nodejob.LearnerManifest{Version: nodejob.LearnerVersion, Controller: actor.Principal, Attempt: id(), WorkloadGeneration: 1, Plan: plan, Slot: plan.Slots[0], Config: config, DatasetSelector: "toy", Limits: limits}
	request, err := m.SigningRequest()
	check(t, err)
	proof, err := trust.SignPrincipalProof(a.Anchor, identity.Chain, signer, v, request, testNow)
	check(t, err)
	accepted, err := nodejob.AcceptLearner(a.Anchor, v, actor, caps.Node, nodejob.SignedLearner{Manifest: m, Proof: proof}, testNow)
	check(t, err)
	_, err = s.PrepareLearner(ctx, actor, accepted, lease.Version, testNow)
	check(t, err)
	root := resolve("learner-credentials", statehome.Agent)
	check(t, root.Ensure())
	clock := func() time.Time { return testNow }
	ports, err := NodeWorkloadPorts(node, root, clock)
	check(t, err)
	workloadRequest, err := s.PrepareLearnerWorkload(ctx, actor, allocation.JobID, ports)
	check(t, err)
	grantQ, err := workloadRequest.GrantRequest()
	check(t, err)
	proof, err = trust.SignPrincipalProof(a.Anchor, identity.Chain, signer, v, grantQ, testNow)
	check(t, err)
	grant := workload.Grant{Request: workloadRequest, Proof: proof}
	credential, err := a.Workloads.Issue(ctx, identity.Chain, grant, v, testNow)
	check(t, err)
	check(t, s.ActivateLearnerCredential(ctx, actor, allocation.JobID, a.Anchor, grant, credential, testNow))
	assignment, err := s.StartLearner(ctx, actor, allocation.JobID, id(), testNow)
	check(t, err)
	binding, err := assignment.Binding()
	check(t, err)
	return learnerRuntimeFixture{remote: &nodeLearnerRemote{runtime: NodeLearnerRuntime{Store: s, Principal: node, Anchor: a.Anchor, Clock: clock}, binding: binding}, authority: a, controller: controller, coordinator: coordinator, ports: ports}
}

func TestNodeLearnerRuntimePrecommitPollReturnsPromptly(t *testing.T) {
	r := learnerRuntimeRemote(t)
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	before, err := r.runtime.Store.ActiveExecution(ctx)
	check(t, err)
	if before.Learner.Commit != nil {
		t.Fatal("fixture is already committed")
	}
	start := time.Now()
	for range 3 {
		out, err := r.Poll(ctx)
		check(t, err)
		if !reflect.DeepEqual(out, private.RelayResponse{Format: "mixlab_learner_reply_v1"}) {
			t.Fatalf("unexpected precommit reply: %+v", out)
		}
	}
	if elapsed := time.Since(start); elapsed >= time.Second {
		t.Fatalf("precommit poll waited for commit: %s", elapsed)
	}
	for name, call := range learnerRemoteCommands(r) {
		t.Run(name, func(t *testing.T) {
			if err := call(ctx); err == nil || !strings.Contains(err.Error(), "not committed") {
				t.Fatalf("precommit command: %v", err)
			}
		})
	}
	after, err := r.runtime.Store.ActiveExecution(ctx)
	check(t, err)
	if !reflect.DeepEqual(before, after) {
		t.Fatal("precommit RPC changed durable execution")
	}
}

// A reader/writer that would reveal accidental artifact access before admission.
type learnerForbiddenIO struct{}

func (learnerForbiddenIO) Read([]byte) (int, error)  { panic("read artifact before admission") }
func (learnerForbiddenIO) Write([]byte) (int, error) { panic("write artifact before admission") }

func learnerRemoteCommands(r *nodeLearnerRemote) map[string]func(context.Context) error {
	return map[string]func(context.Context) error{
		"call": func(ctx context.Context) error { _, err := r.Call(ctx, private.RelayRequest{}); return err },
		"read": func(ctx context.Context) error {
			return r.Read(ctx, run.Assignment{}, run.Blob{}, learnerForbiddenIO{})
		},
		"write": func(ctx context.Context) error {
			return r.Write(ctx, run.Assignment{}, run.Blob{}, learnerForbiddenIO{})
		},
		"evaluation": func(ctx context.Context) error {
			return r.ReadEvaluation(ctx, run.EvaluationAssignment{}, learnerForbiddenIO{})
		},
	}
}

func TestNodeLearnerRuntimeRechecksAttemptFence(t *testing.T) {
	r := learnerRuntimeRemote(t)
	ctx := context.Background()
	for name, mutate := range map[string]func(*wc.Binding){
		"job":        func(b *wc.Binding) { b.JobID = strings.Repeat("f", 32) },
		"attempt":    func(b *wc.Binding) { b.AttemptID = strings.Repeat("f", 32) },
		"binary":     func(b *wc.Binding) { b.BinaryID = "other" },
		"build":      func(b *wc.Binding) { b.BuildID = strings.Repeat("f", 64) },
		"assignment": func(b *wc.Binding) { b.AssignmentDigest[0] ^= 1 },
	} {
		t.Run(name, func(t *testing.T) {
			stale := *r
			mutate(&stale.binding)
			if _, err := stale.Poll(ctx); err == nil || !strings.Contains(err.Error(), "another attempt") {
				t.Fatalf("stale poll: %v", err)
			}
			for operation, call := range learnerRemoteCommands(&stale) {
				if err := call(ctx); err == nil || !strings.Contains(err.Error(), "another attempt") {
					t.Fatalf("%s escaped attempt fence: %v", operation, err)
				}
			}
		})
	}
	_, err := r.Poll(ctx)
	check(t, err)
	check(t, r.runtime.Store.ConfirmLearnerStopped(ctx, r.binding))
	if _, err := r.Poll(ctx); err == nil || !strings.Contains(err.Error(), "no longer live") {
		t.Fatalf("cached relay survived stopped child: %v", err)
	}
	for operation, call := range learnerRemoteCommands(r) {
		if err := call(ctx); err == nil || !strings.Contains(err.Error(), "no longer live") {
			t.Fatalf("%s survived stopped child: %v", operation, err)
		}
	}
}

func TestNodeLearnerRuntimeRechecksReleaseAndTrust(t *testing.T) {
	for _, reason := range []string{"release", "admission_expired", "trust_expired", "pin_changed", "canceled"} {
		t.Run(reason, func(t *testing.T) {
			r := learnerRuntimeRemote(t)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			_, err := r.Poll(ctx)
			check(t, err)
			switch reason {
			case "release":
				check(t, r.runtime.Store.InterruptActive(ctx, "agent_shutdown"))
			case "admission_expired":
				r.runtime.Clock = func() time.Time { return testNow.Add(time.Minute) }
			case "trust_expired":
				r.runtime.Clock = func() time.Time { return testNow.Add(trust.SnapshotLifetime + time.Second) }
			case "pin_changed":
				r.runtime.Anchor = trust.Anchor{}
			case "canceled":
				cancel()
			}
			if _, err := r.Poll(ctx); err == nil {
				t.Fatal("poll retained stale authority")
			}
			for operation, call := range learnerRemoteCommands(r) {
				if err := call(ctx); err == nil {
					t.Fatalf("%s retained stale authority", operation)
				}
			}
		})
	}
}

func TestNodeLearnerRuntimeOpenRequiresLocalApproval(t *testing.T) {
	for _, r := range []NodeLearnerRuntime{{}, {Clock: func() time.Time { return testNow }}} {
		host, runner, err := r.Open(context.Background(), nodeagent.LocalExecution{})
		if err == nil || host != nil || runner != nil {
			t.Fatal("opened unapproved learner")
		}
	}
}
