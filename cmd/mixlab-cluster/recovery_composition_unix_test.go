//go:build darwin || linux

package main

import (
	"bytes"
	"context"
	"crypto/x509"
	"encoding/json"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/admission"
	"github.com/mrothroc/mixlab/artifact"
	"github.com/mrothroc/mixlab/artifact/tensors"
	"github.com/mrothroc/mixlab/coordinator"
	coordinatorlocal "github.com/mrothroc/mixlab/coordinator/local"
	"github.com/mrothroc/mixlab/data"
	"github.com/mrothroc/mixlab/diloco/optimizer"
	runcontract "github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/discovery"
	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/recruitment"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/bootstrap"
	"github.com/mrothroc/mixlab/trust/enrollment"
	"github.com/mrothroc/mixlab/trust/enrollment/enrollee"
	"github.com/mrothroc/mixlab/trust/principal"
	"github.com/mrothroc/mixlab/trust/workload"
	"github.com/mrothroc/mixlab/workerjob"
)

func compositionCheck(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}

type compositionLog struct {
	mu    sync.Mutex
	b     bytes.Buffer
	ready chan struct{}
	once  sync.Once
}

func (l *compositionLog) Write(b []byte) (int, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.ready != nil && strings.Contains(string(b), `"Status":"ready"`) {
		l.once.Do(func() { close(l.ready) })
	}
	return l.b.Write(b)
}

func (l *compositionLog) String() string {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.b.String()
}

type installedRecoveryFixture struct {
	ctx        context.Context
	repo       *coordinatorlocal.Repository
	submission admission.Submission
	nodes      map[string]*nodeagent.Store
	logs       []*compositionLog
	agentPaths map[string]string
	admission  statehome.Path
	authority  statehome.Path
	anchor     trust.Anchor
	proxies    []*compositionDeadlineProxy
	pollGate   string
}

// Completion is broadcast, with the result published before close. Readiness
// and cleanup can both observe it without consuming each other's result.
type compositionService struct {
	done chan struct{}
	err  error
}

func startCompositionService(t *testing.T, parent context.Context, name string, run func(context.Context) error) *compositionService {
	t.Helper()
	ctx, stop := context.WithCancel(parent)
	s := &compositionService{done: make(chan struct{})}
	go func() {
		s.err = run(ctx)
		close(s.done)
	}()
	t.Cleanup(func() {
		stop()
		// The test context may already have expired; joining gets a fresh bound.
		// Production allows 12s HTTP drain followed by 45s execution cleanup.
		timer := time.NewTimer(75 * time.Second)
		defer timer.Stop()
		select {
		case <-s.done:
			if s.err != nil {
				t.Errorf("%s exit: %v", name, s.err)
			}
		case <-timer.C:
			t.Errorf("%s did not stop within cleanup bound", name)
		}
	})
	return s
}

func (s *compositionService) awaitReady(t *testing.T, ctx context.Context, ready <-chan struct{}, log *compositionLog) {
	t.Helper()
	select {
	case <-s.done:
		t.Fatalf("service exited before ready: %v; %s", s.err, log.String())
	case <-ready:
	case <-ctx.Done():
		t.Fatalf("service not ready: %s", log.String())
	}
}

func newInstalledRecoveryFixture(t *testing.T, tamperDeadline bool) *installedRecoveryFixture {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	t.Cleanup(cancel)
	root, err := filepath.EvalSymlinks(t.TempDir())
	compositionCheck(t, err)
	compositionCheck(t, os.Chmod(root, 0700))
	path := func(name string, kind statehome.Kind) statehome.Path {
		p, err := statehome.Resolve(statehome.Options{ExactDir: filepath.Join(root, name)}, statehome.Context{Kind: kind})
		compositionCheck(t, err)
		return p
	}
	al, err := net.Listen("tcp", "127.0.0.1:0")
	compositionCheck(t, err)
	t.Cleanup(func() { _ = al.Close() })
	cl, err := net.Listen("tcp", "127.0.0.1:0")
	compositionCheck(t, err)
	t.Cleanup(func() { _ = cl.Close() })
	ids, err := bootstrap.NewIDs()
	compositionCheck(t, err)
	ca := path("authority", statehome.Authority)
	config := bootstrap.Config{Cluster: ids.Cluster, Authority: ca, Backend: "file", Endpoint: "https://" + al.Addr().String(), Audience: "authority"}
	for i, role := range []trust.Role{trust.Authority, trust.Controller, trust.Coordinator} {
		config.Principals = append(config.Principals, bootstrap.Target{Role: role, Principal: []string{ids.Authority, ids.Controller, ids.Coordinator}[i], Final: path(string(role)+"-principal", statehome.Principal), Staging: path(string(role)+"-stage", statehome.Enrollment)})
	}
	_, err = bootstrap.Initialize(ctx, config, time.Now())
	compositionCheck(t, err)
	compositionCheck(t, clusterapp.InitializeAuthority(ctx, ca, time.Now()))
	authority, err := clusterapp.OpenAuthority(ctx, ca, time.Now())
	compositionCheck(t, err)
	t.Cleanup(func() { compositionCheck(t, authority.Close()) })
	startCompositionService(t, ctx, "authority", func(ctx context.Context) error {
		return clusterapp.ServeAuthority(ctx, al, authority, time.Now)
	})
	controller, err := principal.Open(config.Principals[1].Final, time.Now())
	compositionCheck(t, err)
	t.Cleanup(func() { compositionCheck(t, controller.Close()) })
	x := &installedRecoveryFixture{ctx: ctx, nodes: map[string]*nodeagent.Store{}, agentPaths: map[string]string{}, anchor: authority.Anchor, authority: ca}
	exe, err := os.Executable()
	compositionCheck(t, err)
	exe, err = filepath.EvalSymlinks(exe)
	compositionCheck(t, err)
	build, err := workerjob.FileDigest(exe)
	compositionCheck(t, err)
	dataset := filepath.Join(root, "train.bin")
	x.pollGate = dataset + ".composition-open"
	compositionCheck(t, os.WriteFile(dataset, []byte("composition dataset identity only"), 0600))
	datasetID, err := data.DistributedDatasetIdentity(dataset)
	compositionCheck(t, err)
	limits := nodejob.Limits{RuntimeSeconds: 180, CPUSeconds: 300, MemoryBytes: 1 << 30, DiskBytes: 1 << 30, LogBytes: 1 << 20}
	var hints []discovery.Hint
	for index := range 2 {
		now := time.Now()
		view, err := authority.Current(ctx, now)
		compositionCheck(t, err)
		invite, err := authority.Enrollment.Invite(ctx, enrollment.NodeEnrollment, config.Endpoint, "authority", time.Minute, view, now)
		compositionCheck(t, err)
		np := path(fmt.Sprintf("node-%d-principal", index), statehome.Principal)
		candidate, err := enrollee.Prepare(ctx, path(fmt.Sprintf("node-%d-stage", index), statehome.Enrollment), np, authority.Anchor, trust.Node, "file", now)
		compositionCheck(t, err)
		key, envelope, err := candidate.Keys()
		compositionCheck(t, err)
		request, err := enrollment.NewRequest(invite, key, envelope, now)
		compositionCheck(t, err)
		result, err := authority.Enrollment.Consume(ctx, request, invite.Secret, view, now)
		compositionCheck(t, err)
		compositionCheck(t, candidate.CompleteProvisioned(ctx, invite, request, result, now))
		compositionCheck(t, candidate.Close())
		invite.Clear()
		p, err := principal.Open(np, now)
		compositionCheck(t, err)
		identity, _, err := p.Active(now)
		compositionCheck(t, err)
		compositionCheck(t, p.Close())
		ap := path(fmt.Sprintf("agent-%d", index), statehome.Agent)
		installation := clusterapp.NodeInstallation{Cluster: ids.Cluster, Node: identity.Principal, PrincipalDirectory: np.Dir(), WorkerBinary: exe, GuardianBinary: exe, WorkerBuild: build, GuardianBuild: build, RelayAddress: availableAddress(t)}
		profile := nodeagent.Profile{Version: nodeagent.ProfileVersion, Node: identity.Principal, DisplayName: "composition node", TransportEndpoint: installation.RelayAddress, Generation: 1, Probe: compositionProbe(build), ProbeObservedAt: now.Unix(), Limits: limits, Datasets: []nodeagent.LocalDataset{{Dataset: nodeagent.Dataset{Selector: "train", ID: datasetID}, TrainPattern: dataset}, {Dataset: nodeagent.Dataset{Selector: "validation", ID: datasetID}, TrainPattern: dataset}}}
		compositionCheck(t, clusterapp.InitializeNodeInstallation(ctx, ap, installation, profile))
		store, err := nodeagent.Open(ap, ids.Cluster, identity.Principal)
		compositionCheck(t, err)
		x.nodes[identity.Principal] = store
		x.agentPaths[identity.Principal] = ap.Dir()
		address := availableAddress(t)
		log := &compositionLog{ready: make(chan struct{})}
		x.logs = append(x.logs, log)
		service := startCompositionService(t, ctx, "agent", func(ctx context.Context) error {
			if code := runAgentContext(ctx, []string{"-agent-state-dir", ap.Dir(), "-agent-listen", address, "-agent-advertise", "off"}, log, log); code != 0 {
				return fmt.Errorf("agent exit %d: %s", code, log.String())
			}
			return nil
		})
		service.awaitReady(t, ctx, log.ready, log)
		endpoint := address
		if tamperDeadline {
			proxy := newCompositionDeadlineProxy(t, np, config.Principals[1].Final, config.Principals[2].Final, address)
			x.proxies = append(x.proxies, proxy)
			endpoint = proxy.endpoint
		}
		hints = append(hints, discovery.Hint{Service: discovery.Node, Endpoint: endpoint})
	}
	o := clusterapp.DiLoCoCoordinatorOptions{Principal: config.Principals[2].Final, AdmissionPrincipal: config.Principals[1].Final, State: path("run-state", statehome.Principal), Admission: path("admission", statehome.Principal), Checkpoints: path("checkpoints", statehome.Principal), Artifacts: path("artifacts", statehome.Principal), Endpoint: "https://" + cl.Addr().String(), Clock: time.Now}
	compositionCheck(t, clusterapp.InitializeDiLoCoCoordinator(o))
	x.admission = o.Admission
	ready := make(chan struct{})
	log := &compositionLog{}
	x.logs = append(x.logs, log)
	o.Ready = func() { close(ready) }
	o.Event = func(err error) { _, _ = fmt.Fprintln(log, err) }
	service := startCompositionService(t, ctx, "coordinator", func(ctx context.Context) error {
		return clusterapp.ServeInstalledDiLoCo(ctx, cl, o)
	})
	service.awaitReady(t, ctx, ready, log)
	client, err := clusterapp.NewDiLoCoClient(controller, o.Endpoint, ids.Coordinator, time.Now)
	compositionCheck(t, err)
	lookup, err := clusterapp.NodeLookup(controller, time.Now)
	compositionCheck(t, err)
	requirements := recruitment.Requirements{Cluster: ids.Cluster, Count: 2, BuildID: build, MLXVersion: "0.32.3", DeviceKind: "metal", DType: "fp32", CustomOps: []string{}, DatasetSelector: "train", DatasetID: datasetID, Limits: limits, ProbeMaxAge: 10 * time.Minute}
	selection, err := recruitment.SelectLearners(ctx, hints, requirements, lookup, time.Now)
	compositionCheck(t, err)
	var body bytes.Buffer
	compositionCheck(t, tensors.Write(&body, []optimizer.Tensor{{Name: "w0", Shape: []int{1}, Values: []float32{4}}}))
	model, err := client.UploadInitialModel(ctx, artifact.Ref{SHA256: nodejob.Hash(body.Bytes()), Bytes: uint64(body.Len())}, bytes.NewReader(body.Bytes()))
	compositionCheck(t, err)
	runID, err := submissionID()
	compositionCheck(t, err)
	attempt, err := submissionID()
	compositionCheck(t, err)
	cfg := json.RawMessage(`{"model_dim":16}`)
	plan := compositionPlan(build, cfg)
	now := time.Now().Unix()
	q := admission.Submission{Owner: ids.Controller, Launch: recruitment.LaunchPlan{Cluster: ids.Cluster, Controller: ids.Controller, Run: runID, Attempt: attempt, Selection: selection, TTLSeconds: 300}, Deadline: now + 180, Specification: admission.Specification{Config: cfg, CreatedAt: now, BaseModel: model, Outer: plan.Learner.Outer, Endpoint: o.Endpoint, ArtifactStoreID: "coordinator-" + ids.Coordinator, Coordinator: runcontract.Coordinator{ClusterID: ids.Cluster, PrincipalID: ids.Coordinator, Role: "coordinator", Audience: "learner-run"}, Contract: runcontract.Contract{ConfigHash: plan.ConfigHash, ProgramHash: plan.ProgramHash, WeightLayoutHash: plan.WeightLayoutHash, InnerOptimizerHash: plan.OptimizerHash, DatasetHash: datasetID, BuildHash: build}, Policy: coordinator.EvaluationPolicy{WorkerID: clusterapp.DiLoCoWorkerID(runID, attempt, selection.Selected[0].Capabilities.Node), ValidationHash: datasetID, Split: "validation", BatchTokens: 8, SequenceLength: 4, Batches: 1, TimeoutSeconds: 120}}}
	_, err = client.Submit(ctx, q)
	compositionCheck(t, err)
	x.submission = q
	x.repo, err = coordinatorlocal.Open(o.State)
	compositionCheck(t, err)
	return x
}

type compositionBudget struct {
	admission, recovery   *time.Timer
	admissionC, recoveryC <-chan time.Time
}

func newCompositionBudget(t *testing.T) *compositionBudget {
	t.Helper()
	b := &compositionBudget{admission: time.NewTimer(90 * time.Second)}
	b.admissionC = b.admission.C
	t.Cleanup(func() {
		b.admission.Stop()
		if b.recovery != nil {
			b.recovery.Stop()
		}
	})
	return b
}

func (b *compositionBudget) observe(t *testing.T, x *installedRecoveryFixture, phase coordinator.Phase) {
	t.Helper()
	b.check(t, x, phase, nil)
	if phase == coordinator.Open && b.recovery == nil {
		b.admission.Stop()
		b.admissionC = nil
		// Start exactly once. Later phases, retries and state reads cannot renew it.
		b.recovery = time.NewTimer(65 * time.Second)
		b.recoveryC = b.recovery.C
		// The gate only releases the fake numerical executable's first Poll;
		// coordinator admission and all recovery paths remain production-owned.
		compositionCheck(t, os.WriteFile(x.pollGate, []byte("open\n"), 0600))
	}
}

func (b *compositionBudget) check(t *testing.T, x *installedRecoveryFixture, phase coordinator.Phase, delay <-chan time.Time) {
	t.Helper()
	// A nil delay checks expiry without sleeping; a real delay is a post-call
	// observer pause, so slow signed-state validation cannot create a busy loop.
	if delay == nil {
		select {
		case <-b.admissionC:
			t.Fatalf("composition admission exceeded 90s: phase=%s; %s", phase, x.diagnostics())
		case <-b.recoveryC:
			t.Fatalf("composition recovery exceeded 65s after first Open: phase=%s; %s", phase, x.diagnostics())
		default:
			return
		}
	}
	select {
	case <-x.ctx.Done():
		t.Fatalf("composition context expired: phase=%s; %s", phase, x.diagnostics())
	case <-b.admissionC:
		t.Fatalf("composition admission exceeded 90s: phase=%s; %s", phase, x.diagnostics())
	case <-b.recoveryC:
		t.Fatalf("composition recovery exceeded 65s after first Open: phase=%s; %s", phase, x.diagnostics())
	case <-delay:
	}
}

func (b *compositionBudget) wait(t *testing.T, x *installedRecoveryFixture, phase coordinator.Phase) {
	t.Helper()
	delay := time.NewTimer(500 * time.Millisecond)
	defer delay.Stop()
	b.check(t, x, phase, delay.C)
}

// This exercises production installation wiring, not numerical checkpoint replay.
func TestInstalledLearnerRecoveryComposition(t *testing.T) {
	x := newInstalledRecoveryFixture(t, false)
	budget := newCompositionBudget(t)
	var before *coordinator.State
	var last coordinator.State
	for {
		s, err := x.repo.Load(x.ctx, x.submission.Launch.Run)
		if err == nil {
			last = s
			budget.observe(t, x, s.Phase)
			if s.Phase == coordinator.Failed {
				t.Fatalf("installed recovery failed: %s; logs=%s", s.Failure, x.diagnostics())
			}
			if before == nil && s.Phase == coordinator.Open {
				before = &s
			}
			if len(s.Attempts) == 1 && s.Recovery == nil {
				a := s.Attempts[0]
				node := x.nodes[a.Allocation.NodeID]
				execution, err := node.ActiveExecution(x.ctx)
				compositionCheck(t, err)
				if execution != nil && execution.Outcome != nil && execution.Outcome.PID > 0 && !execution.Learner.Stopped && execution.Learner.Assignment.AttemptID == a.AttemptID {
					if before == nil || !reflect.DeepEqual(before.Plan.Slots, s.Plan.Slots) || !reflect.DeepEqual(before.Assignments, s.Assignments) || a.PrivateHash != strings.Repeat("c", 64) || execution.Learner.Assignment.ExpectedPrivateHash != a.PrivateHash || execution.Workload.Generation != 2 {
						t.Fatal("composition recovery changed allocation, assignment or private binding")
					}
					q := x.submission
					want, err := (nodejob.Manifest{Created: q.Specification.CreatedAt, Expires: q.Deadline, Limits: q.Launch.Selection.Requirements.Limits}).WorkloadDeadline()
					compositionCheck(t, err)
					if execution.Workload.Deadline != want.Unix() || execution.Learner.Credential == nil || execution.Learner.Credential.Binding.Attempt != a.AttemptID || len(execution.Learner.History) != 1 {
						t.Fatal("retry did not retain original deadline and authority-issued scope")
					}
					history, err := node.LearnerOutcomes(x.ctx, execution.Job.ID)
					compositionCheck(t, err)
					if len(history) != 3 || !history[1].Terminal() || history[1].AttemptID == a.AttemptID || history[1].PID <= 0 || history[1].Exit.Kind == "" || history[2] != *execution.Outcome {
						t.Fatalf("retry lost actual prior child exit evidence: %+v", history)
					}
					if !strings.Contains(x.diagnostics(), "learner exit:") {
						t.Fatal("installed agent did not log the learner exit")
					}
					grantBytes, err := os.ReadFile(x.grantPath(execution.Workload))
					compositionCheck(t, err)
					var grant workload.Grant
					compositionCheck(t, json.Unmarshal(grantBytes, &grant))
					credential := *execution.Learner.Credential
					compositionCheck(t, workload.ValidateResult(x.anchor, grant, credential, time.Now()))
					cert, err := x509.ParseCertificate(credential.Chain[0])
					compositionCheck(t, err)
					if grant.Request.Request.Scope.Deadline != want.Unix() || credential.Binding.ExpiresAt != want.Unix() || !cert.NotAfter.Equal(want) {
						t.Fatal("persisted grant or authority credential extended original deadline")
					}
					marker := fmt.Sprintf("composition-start-ack attempt=%s pid=%d\n", a.AttemptID, execution.Outcome.PID)
					paths, err := filepath.Glob(filepath.Join(x.agentPaths[a.Allocation.NodeID], "runtime", "*", "worker.log"))
					compositionCheck(t, err)
					for _, path := range paths {
						b, err := os.ReadFile(path)
						compositionCheck(t, err)
						if bytes.Contains(b, []byte(marker)) {
							t.Logf("installed recovery replacement acknowledged IPC Start: attempt %s, PID %d", a.AttemptID, execution.Outcome.PID)
							return
						}
					}
				}
			}
		}
		budget.wait(t, x, last.Phase)
	}
}

func (x *installedRecoveryFixture) grantPath(s workload.Scope) string {
	return filepath.Join(x.admission.Dir(), s.Run+"-"+s.Job+"-recovery-"+s.Attempt+"-grant")
}

func (x *installedRecoveryFixture) diagnostics() string {
	var b strings.Builder
	for _, log := range x.logs {
		b.WriteString(log.String())
	}
	for node, path := range x.agentPaths {
		for _, name := range []string{"worker.log", "physical-exit.json", "execution-attempt.json"} {
			paths, err := filepath.Glob(filepath.Join(path, "runtime", "*", name))
			if err != nil {
				fmt.Fprintf(&b, "node %s diagnostics: %v\n", node, err)
				continue
			}
			for _, file := range paths {
				body, err := os.ReadFile(file)
				if err != nil {
					fmt.Fprintf(&b, "%s: %v\n", file, err)
					continue
				}
				fmt.Fprintf(&b, "%s: %s\n", file, body)
			}
		}
	}
	return b.String()
}
