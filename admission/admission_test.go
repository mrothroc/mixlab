package admission

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/coordinator"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/distributed"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/recruitment"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/workercontrol"
	"github.com/mrothroc/mixlab/workerprobe"
)

type fixture struct {
	s            *Saga
	p            Ports
	plan         run.Plan
	regs         []run.Registration
	now          int64
	fail         string
	lost         bool
	leases       map[string]nodeagent.Lease
	started      map[string]bool
	finalized    map[string]bool
	commit       *run.Commit
	stage, abort bool
}

func setup(t *testing.T) *fixture {
	t.Helper()
	x := &fixture{now: 1200, leases: map[string]nodeagent.Lease{}, started: map[string]bool{}, finalized: map[string]bool{}}
	b, err := os.ReadFile("../diloco/testdata/plan.json")
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(b, &x.plan); err != nil {
		t.Fatal(err)
	}
	var original run.Commit
	b, err = os.ReadFile("../diloco/testdata/commit.json")
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(b, &original); err != nil {
		t.Fatal(err)
	}
	x.regs = original.Registrations
	lim := nodejob.Limits{RuntimeSeconds: 600, CPUSeconds: 1200, MemoryBytes: 1 << 30, DiskBytes: 1 << 30, LogBytes: 1 << 20}
	req := recruitment.Requirements{Cluster: strings.Repeat("a", 32), Count: 2, BuildID: x.plan.Contract.BuildHash, MLXVersion: "0.32.1", DeviceKind: "metal", DType: "fp32", CustomOps: []string{}, DatasetSelector: "toy", DatasetID: x.plan.Contract.DatasetHash, Limits: lim, ProbeMaxAge: time.Minute}
	lp := recruitment.LaunchPlan{Cluster: req.Cluster, Controller: strings.Repeat("b", 32), Run: x.plan.Cohort.RunID, Attempt: strings.Repeat("c", 32), TTLSeconds: 300, Selection: recruitment.Selection{Policy: recruitment.SelectionPolicy, Requirements: req}}
	for i := range 2 {
		nodeID := strings.Repeat(fmt.Sprint(i+1), 32)
		p := workerprobe.Report{Version: workerprobe.Version, BuildID: req.BuildID, BuildVersion: "test", WorkerProtocol: workercontrol.Version, OS: "darwin", Arch: "arm64", Available: true, MLXVersion: req.MLXVersion, MLXSupported: true, DeviceKind: "metal", DeviceName: "test", Backends: []string{"ring"}, DTypes: []string{"fp32"}, CustomOps: []string{}}
		c := nodeagent.Capabilities{Version: nodeagent.CapabilityVersion, Node: nodeID, DisplayName: "test", Generation: 1, ObservedAt: x.now, ProbeObservedAt: x.now, Limits: lim, ManifestVersions: []string{nodejob.LearnerVersion}, Datasets: []nodeagent.Dataset{{Selector: "toy", ID: req.DatasetID}}, Availability: nodeagent.Availability{NodeVersion: 1, CapabilityGeneration: 1, Available: true}, Recruitable: true, Probe: p}
		lp.Selection.Selected = append(lp.Selection.Selected, recruitment.Candidate{Endpoint: fmt.Sprintf("node%d.local:7445", i), Capabilities: c})
		x.plan.Slots[i].CapabilityHash, err = run.Digest(c)
		if err != nil {
			t.Fatal(err)
		}
		x.plan.Slots[i].Member.ParticipantID = nodeID
		x.plan.Slots[i].Allocation.NodeID = nodeID
	}
	x.plan.Coordinator.ClusterID = req.Cluster
	config := json.RawMessage(`{"model_dim":16}`)
	x.plan.Contract.ConfigHash = nodejob.Hash(config)
	for i := range x.regs {
		x.regs[i].Contract = x.plan.Contract
	}
	spec := Specification{Contract: x.plan.Contract, BaseModel: x.plan.BaseModel, Outer: x.plan.Outer, Coordinator: x.plan.Coordinator, ArtifactStoreID: x.plan.ArtifactStoreID, Endpoint: x.plan.Endpoint, Config: config, CreatedAt: x.plan.CreatedAt}
	dir, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if err = os.Chmod(dir, 0700); err != nil {
		t.Fatal(err)
	}
	path, err := statehome.Resolve(statehome.Options{ExactDir: filepath.Join(dir, "saga")}, statehome.Context{Kind: statehome.Principal})
	if err != nil {
		t.Fatal(err)
	}
	x.s, err = Begin(path, lp, spec, x.plan.ExpiresAt)
	if err != nil {
		t.Fatal(err)
	}
	x.p = Ports{Clock: func() int64 { return x.now }, Reserve: func(_ context.Context, c recruitment.Candidate, q nodeagent.Reserve) (nodeagent.Lease, error) {
		l, ok := x.leases[c.Capabilities.Node]
		if !ok {
			l = nodeagent.Lease{Format: nodeagent.LeaseVersion, ID: q.IdempotencyKey, Node: c.Capabilities.Node, Controller: lp.Controller, Run: lp.Run, CapabilityGeneration: 1, Version: 1, State: nodeagent.Reserved, Created: x.now, Expires: x.now + 300, RenewBy: x.now + 150}
			x.leases[l.Node] = l
		}
		return l, x.effect("reserve")
	},
		AbortReservation: func(_ context.Context, c recruitment.Candidate, _ nodeagent.Reserve) error {
			delete(x.leases, c.Capabilities.Node)
			delete(x.started, c.Capabilities.Node)
			return x.effect("release")
		},
		BuildPlan: func(_ context.Context, _ recruitment.LaunchPlan, _ Specification, leases []nodeagent.Lease) (run.Plan, error) {
			members := []distributed.LearnerMember{}
			for i, l := range leases {
				x.plan.Slots[i].Allocation.LeaseID = l.ID
				x.plan.Slots[i].Allocation.RefHash = ""
				h, _ := run.Digest(x.plan.Slots[i].Allocation)
				x.plan.Slots[i].Allocation.RefHash = h
				members = append(members, x.plan.Slots[i].Member)
			}
			var e error
			x.plan.Cohort, e = distributed.NewLearnerCohort(lp.Run, 0, members)
			if e != nil {
				return run.Plan{}, e
			}
			x.plan.SHA256, e = x.plan.Digest()
			return x.plan, e
		},
		Stage: func(_ context.Context, p run.Plan, _ coordinator.EvaluationPolicy) error {
			if x.abort {
				return errors.New("aborted")
			}
			x.stage = true
			return x.effect("stage")
		},
		Prepare: func(_ context.Context, p run.Plan, _ Specification, i int) (run.Registration, error) {
			r := x.regs[i]
			r.PlanHash = p.SHA256
			r.CohortHash = p.Cohort.CohortHash
			r.Slot = p.Slots[i]
			r.Coordinator = p.Coordinator
			x.regs[i] = r
			return r, x.effect("prepare")
		},
		Start: func(_ context.Context, _ run.Plan, r run.Registration) error {
			x.started[r.Slot.Allocation.NodeID] = true
			return x.effect("start")
		},
		Register: func(context.Context, run.Registration) error { return x.effect("register") },
		Commit: func(_ context.Context, c run.Commit) error {
			if x.abort {
				return errors.New("aborted")
			}
			if x.commit != nil && x.commit.Binding != c.Binding {
				return errors.New("second commit")
			}
			x.commit = &c
			return x.effect("commit")
		},
		Abort: func(context.Context, run.Plan, coordinator.EvaluationPolicy) (*run.Commit, error) {
			if x.commit != nil {
				return x.commit, x.effect("abort")
			}
			x.abort = true
			return nil, x.effect("abort")
		},
		Finalize: func(_ context.Context, _ run.Plan, c run.Commit, i int) error {
			if x.commit == nil || x.commit.Binding != c.Binding {
				return errors.New("uncommitted handoff")
			}
			x.finalized[c.Registrations[i].Slot.Allocation.NodeID] = true
			return x.effect("finalize")
		},
	}
	return x
}
func (x *fixture) effect(name string) error {
	if x.fail == name && !x.lost {
		x.lost = true
		return errors.New("lost effect response")
	}
	return nil
}
func (x *fixture) finish(t *testing.T) {
	t.Helper()
	for range 100 {
		var err error
		x.s, err = Open(x.s.path)
		if err != nil {
			t.Fatal(err)
		}
		done, e := x.s.Step(context.Background(), x.p)
		if e != nil {
			if e.Error() == "lost effect response" {
				continue
			}
			t.Fatal(e)
		}
		if done {
			return
		}
	}
	t.Fatal("saga did not converge")
}
func TestAdmissionLostResponsesAtEveryBoundary(t *testing.T) {
	for _, point := range []string{"reserve", "stage", "prepare", "start", "register", "commit", "finalize"} {
		t.Run(point, func(t *testing.T) {
			x := setup(t)
			x.fail = point
			x.finish(t)
			if !x.lost || x.commit == nil || len(x.finalized) != 2 || len(x.started) != 2 {
				t.Fatal("incomplete handoff")
			}
			if x.s.Cancel(context.Background()) == nil {
				t.Fatal("submitter canceled committed run")
			}
		})
	}
}
func TestAdmissionCancelAtEveryDurableBoundary(t *testing.T) {
	for stop := range 20 {
		t.Run(fmt.Sprint(stop), func(t *testing.T) {
			x := setup(t)
			for range stop {
				done, err := x.s.Step(context.Background(), x.p)
				if err != nil {
					t.Fatal(err)
				}
				if done {
					break
				}
			}
			_, r, err := x.s.load()
			if err != nil {
				t.Fatal(err)
			}
			if !r.Committed && !r.Done {
				if err = x.s.Cancel(context.Background()); err != nil {
					t.Fatal(err)
				}
			}
			x.finish(t)
			if x.commit != nil {
				if len(x.finalized) != 2 {
					t.Fatal("committed run not handed off")
				}
			} else if len(x.leases) != 0 || len(x.started) != 0 {
				t.Fatal("stranded reservation or child")
			}
		})
	}
}
func TestAdmissionAmbiguousCommitMustNotRelease(t *testing.T) {
	x := setup(t)
	x.fail = "commit"
	ctx := context.Background()
	for range 50 {
		_, err := x.s.Step(ctx, x.p)
		if err != nil {
			if !x.lost {
				t.Fatal(err)
			}
			break
		}
	}
	if x.commit == nil {
		t.Fatal("commit not reached")
	}
	if err := x.s.Cancel(ctx); err != nil {
		t.Fatal(err)
	}
	x.finish(t)
	if len(x.leases) != 2 || len(x.finalized) != 2 {
		t.Fatal("lost commit response caused compensation")
	}
}
func TestAdmissionDeadlineCompensatesAmbiguousReserve(t *testing.T) {
	x := setup(t)
	x.fail = "reserve"
	ctx := context.Background()
	if _, err := x.s.Step(ctx, x.p); err != nil {
		t.Fatal(err)
	}
	if _, err := x.s.Step(ctx, x.p); err == nil {
		t.Fatal("expected lost response")
	}
	x.now = x.plan.ExpiresAt
	x.fail = "release"
	x.lost = false
	x.finish(t)
	if len(x.leases) != 0 {
		t.Fatal("unknown reservation stranded")
	}
}
