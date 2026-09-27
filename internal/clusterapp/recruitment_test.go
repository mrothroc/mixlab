package clusterapp

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"github.com/mrothroc/mixlab/artifact"
	artifactlocal "github.com/mrothroc/mixlab/artifact/local"
	"github.com/mrothroc/mixlab/workerjob"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/discovery"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/recruitment"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/transport/managedtls"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/enrollment"
	"github.com/mrothroc/mixlab/trust/enrollment/enrollee"
	"github.com/mrothroc/mixlab/trust/principal"
	"github.com/mrothroc/mixlab/workercontrol"
	"github.com/mrothroc/mixlab/workerhost"
	"github.com/mrothroc/mixlab/workerhost/contract"
	"github.com/mrothroc/mixlab/workerprobe"
)

// Only the native numerical process is simulated. Enrollment, controller
// selection, HTTP/TLS, issuance, relays, durable launch and node cleanup are real.
func TestRecruitmentAuthenticatedTwoNodeLaunch(t *testing.T) {
	testRecruitmentAuthenticatedTwoNodeLaunch(t, false)
}

func TestRecruitmentPushedWorkloadRevocationCleansCohort(t *testing.T) {
	testRecruitmentAuthenticatedTwoNodeLaunch(t, true)
}

func TestRecruitmentCheckpointResumeTransfer(t *testing.T) {
	testRecruitmentAuthenticatedTwoNodeLaunch(t, false, true)
}

func testRecruitmentAuthenticatedTwoNodeLaunch(t *testing.T, revoke bool, resumeMode ...bool) {
	doResume := len(resumeMode) > 0 && resumeMode[0]
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	check(t, err)
	endpoint := "https://" + l.Addr().String()
	caPath := initializedAt(t, endpoint)
	check(t, InitializeAuthority(ctx, caPath, testNow))
	a, err := OpenAuthority(ctx, caPath, testNow)
	check(t, err)
	defer func() { check(t, a.Close()) }()
	var currentTime atomic.Int64
	currentTime.Store(testNow.Unix())
	clock := func() time.Time { return time.Unix(currentTime.Load(), 0) }
	authorityDone := make(chan error, 1)
	go func() { authorityDone <- ServeAuthority(ctx, l, a, clock) }()
	defer func() { cancel(); check(t, <-authorityDone) }()
	resolve := func(name string, kind statehome.Kind) statehome.Path {
		p, err := statehome.Resolve(statehome.Options{ExactDir: filepath.Join(filepath.Dir(caPath.Dir()), name)}, statehome.Context{Kind: kind})
		check(t, err)
		return p
	}
	controller, err := principal.Open(resolve("controller", statehome.Principal), testNow)
	check(t, err)
	defer func() { check(t, controller.Close()) }()
	identity, _, err := controller.Active(testNow)
	check(t, err)
	build, dataset := strings.Repeat("a", 64), strings.Repeat("b", 64)
	config := []byte(`{"test":"numerical worker is a test port"}`)
	plan := workerprobe.Plan{Version: workerprobe.PlanVersion, BuildID: build, ConfigHash: nodejob.Hash(config), ProgramHash: strings.Repeat("c", 64), WeightLayoutHash: strings.Repeat("d", 64), OptimizerHash: strings.Repeat("e", 64), DType: "fp32"}
	limits := nodejob.Limits{RuntimeSeconds: 60, CPUSeconds: 120, MemoryBytes: 1 << 30, DiskBytes: 1 << 30, LogBytes: 1 << 20}
	probe := workerprobe.Report{Version: workerprobe.Version, BuildID: build, BuildVersion: "test", WorkerProtocol: workercontrol.Version, OS: "darwin", Arch: "arm64", Available: true, MLXVersion: "test", MLXSupported: true, DeviceKind: "metal", DeviceName: "synthetic test device", Backends: []string{"ring"}, DTypes: []string{"fp32"}, CustomOps: []string{}}
	var starts atomic.Int32
	allStarted := make(chan struct{})
	resumeStarted := make(chan struct{})
	var hints []discovery.Hint
	var stores []*nodeagent.Store
	apps := map[string]*NodeJobs{}
	principals := map[string]*principal.Store{}
	payload := bytes.Repeat([]byte("synthetic weights"), 80000)
	hash := sha256.Sum256(payload)
	outputRef := artifact.Ref{SHA256: hex.EncodeToString(hash[:]), Bytes: uint64(len(payload))}
	for i := range 2 {
		name := fmt.Sprintf("node-%d", i)
		view, err := a.Current(ctx, testNow)
		check(t, err)
		invite, err := a.Enrollment.Invite(ctx, enrollment.NodeEnrollment, endpoint, "authority", time.Minute, view, testNow)
		check(t, err)
		staging, final := resolve(name+"-staging", statehome.Enrollment), resolve(name, statehome.Principal)
		candidate, err := enrollee.Prepare(ctx, staging, final, a.Anchor, trust.Node, "file", testNow)
		check(t, err)
		key, envelope, err := candidate.Keys()
		check(t, err)
		request, err := enrollment.NewRequest(invite, key, envelope, testNow)
		check(t, err)
		approved, err := a.Enrollment.Consume(ctx, request, invite.Secret, view, testNow)
		check(t, err)
		check(t, candidate.CompleteProvisioned(ctx, invite, request, approved, testNow))
		invite.Clear()
		check(t, candidate.Close())
		node, err := principal.Open(final, testNow)
		check(t, err)
		defer func() { check(t, node.Close()) }()
		principals[approved.Approval.Principal] = node
		agentPath := resolve(name+"-agent", statehome.Agent)
		check(t, agentPath.Ensure())
		store, err := nodeagent.Initialize(ctx, agentPath, identity.Cluster, approved.Approval.Principal, 1)
		check(t, err)
		stores = append(stores, store)
		relay := recruitmentFreeAddress(t)
		profile := nodeagent.Profile{Version: nodeagent.ProfileVersion, Node: approved.Approval.Principal, DisplayName: name, Generation: 1, Probe: probe, ProbeObservedAt: testNow.Unix(), Limits: limits, TransportEndpoint: relay, Datasets: []nodeagent.LocalDataset{{Dataset: nodeagent.Dataset{Selector: "toy", ID: dataset}, TrainPattern: "/synthetic/train.bin"}}}
		check(t, store.InstallProfile(ctx, profile))
		root := resolve(name+"-credentials", statehome.Agent)
		check(t, root.Ensure())
		wp, err := NodeWorkloadPorts(node, root, clock)
		check(t, err)
		runtimes, err := workerhost.InitializeRuntimeStore(resolve(name+"-runtime", statehome.Worker))
		check(t, err)
		app, err := NewNodeJobs(ctx, NodeJobOptions{Store: store, Node: profile.Node, Anchor: a.Anchor, CurrentTrust: NodeCurrentTrust(node), CredentialRoot: root, RelayAddress: relay, Clock: clock, Workload: wp, Preparation: nodeagent.PreparationPorts{Clock: clock, Probe: func(context.Context) (workerprobe.Report, error) { return probe, nil }, DatasetID: func(context.Context, string) (string, error) { return dataset, nil }, ArtifactPresent: func(context.Context, nodejob.ArtifactRef) error { return fmt.Errorf("unexpected artifact") }}})
		check(t, err)
		app.options.Output = runtimes.Output
		app.options.InputRoot = agentPath
		apps[profile.Node] = app
		runner := &recruitmentRunner{starts: &starts, allStarted: allStarted, resumeStarted: resumeStarted, waitForCancel: revoke, output: func(ctx context.Context, a contract.Approved) error {
			if a.Assignment.Resume != nil {
				f, err := os.Open(a.Assignment.ResumePath)
				if err != nil {
					return err
				}
				err = artifact.Copy(ctx, io.Discard, f, *a.Assignment.Resume)
				_ = f.Close()
				if err != nil {
					return err
				}
			}
			if a.Assignment.View.LocalRank != 0 {
				return nil
			}
			p, err := runtimes.ExistingAttempt(ctx, a.Assignment.JobID, a.Assignment.AttemptID)
			if err != nil {
				return err
			}
			s, err := artifactlocal.Open(p)
			if err != nil {
				return err
			}
			if err := s.Put(ctx, outputRef, bytes.NewReader(payload)); err != nil {
				return err
			}
			b, err := json.Marshal(outputRef)
			if err != nil {
				return err
			}
			return p.CompareAndSwap(workerjob.OutputReceiptFile, nil, b)
		}}
		listener, err := net.Listen("tcp", "127.0.0.1:0")
		check(t, err)
		hints = append(hints, discovery.Hint{Service: discovery.Node, Endpoint: listener.Addr().String()})
		life, stop := context.WithCancel(ctx)
		ready, done := make(chan struct{}), make(chan error, 1)
		go func() {
			done <- ServeNode(life, listener, NodeServerOptions{Store: store, Jobs: app, Snapshots: &NodeSnapshots{Principal: node, Clock: clock}, Policy: func() (*managedtls.Policy, error) { return NodePrincipalPolicy(node, clock) }, Maintenance: func(ctx context.Context) error { <-ctx.Done(); return nil }, Ready: func() { close(ready) }, Execution: NodeExecutionPorts{Clock: clock, Authorize: app.Authorize, Cleanup: app.Cleanup, Open: func(ctx context.Context, job, attempt string) (*workerhost.AttemptStore, workerhost.Runner, error) {
				p, err := runtimes.Attempt(ctx, job, attempt)
				if err != nil {
					return nil, nil, err
				}
				host, err := workerhost.NewAttemptStore(p)
				return host, runner, err
			}}})
		}()
		defer func() { stop(); check(t, <-done) }()
		select {
		case <-ready:
		case err := <-done:
			t.Fatal("node exited before ready", err)
		case <-ctx.Done():
			t.Fatal(ctx.Err())
		}
	}
	lookup, err := NodeLookup(controller, clock)
	check(t, err)
	requirements := recruitment.Requirements{Cluster: identity.Cluster, Count: 2, BuildID: build, MLXVersion: probe.MLXVersion, DeviceKind: "metal", DType: "fp32", CustomOps: []string{}, DatasetSelector: "toy", DatasetID: dataset, Limits: limits, ProbeMaxAge: time.Minute}
	selection, err := recruitment.Select(ctx, hints, requirements, lookup, clock)
	check(t, err)
	options := RecruitmentOptions{Principal: controller, Config: config, NumericalPlan: plan, AuthorityEndpoint: endpoint, LoopbackPorts: []int{recruitmentFreePort(t), recruitmentFreePort(t)}, Clock: clock}
	if doResume {
		options.CheckpointAt = 2
	}
	ports, err := RecruitmentPorts(options)
	check(t, err)
	prepare := ports.Prepare
	ports.Prepare = func(ctx context.Context, candidate recruitment.Candidate, version uint64, signed nodejob.Signed) (nodeagent.PreparedWorkload, error) {
		out, err := prepare(ctx, candidate, version, signed)
		if err != nil {
			return out, err
		}
		client, err := NewNodeClient(controller, candidate.Endpoint, candidate.Capabilities.Node, clock)
		if err != nil {
			return out, err
		}
		var ref artifact.Ref
		if err := client.request(ctx, http.MethodGet, "/v1/agent/jobs/"+out.Job.ID+"/output", nil, &ref); err == nil {
			return out, fmt.Errorf("unfinished job output exposed")
		}
		return out, nil
	}
	run, err := controllerRandomID()
	check(t, err)
	attempt, err := controllerRandomID()
	check(t, err)
	launchPath := resolve("launch", statehome.Principal)
	launch, err := recruitment.BeginLaunch(launchPath, recruitment.LaunchPlan{Cluster: identity.Cluster, Controller: identity.Principal, Run: run, Attempt: attempt, Selection: selection, TTLSeconds: 300})
	check(t, err)
	if revoke {
		done := make(chan error, 1)
		go func() {
			select {
			case <-ctx.Done():
				done <- ctx.Err()
				return
			case <-allStarted:
			}
			x, err := stores[0].ActiveExecution(ctx)
			if err != nil {
				done <- err
				return
			}
			if x == nil || x.Workload.Workload == "" {
				done <- fmt.Errorf("missing active workload")
				return
			}
			next, err := a.Snapshots.Revoke(ctx, "principal", x.Workload.Workload, "prospective", "active workload push test", clock())
			if err != nil {
				done <- err
				return
			}
			for _, candidate := range selection.Selected {
				if _, err := PushNodeSnapshot(ctx, a.Anchor, next, candidate.Endpoint, clock); err != nil {
					done <- err
					return
				}
			}
			done <- nil
		}()
		if err := launch.Run(ctx, ports); err == nil {
			t.Fatal("revoked running cohort reported success")
		}
		check(t, <-done)
		for _, store := range stores {
			availability, err := store.Availability(clock())
			check(t, err)
			if !availability.Available {
				t.Fatal("revoked cohort not physically cleaned up")
			}
		}
		if starts.Load() != 2 {
			t.Fatal("unexpected duplicate or absent launches")
		}
		return
	}
	check(t, launch.Run(ctx, ports))
	if starts.Load() != 2 {
		t.Fatal("wrong native launch count", starts.Load())
	}
	for _, store := range stores {
		availability, err := store.Availability(testNow)
		check(t, err)
		if !availability.Available {
			t.Fatal("success reported before physical cleanup")
		}
	}
	reopened, err := recruitment.OpenLaunch(launchPath)
	check(t, err)
	check(t, reopened.Run(ctx, ports))
	if starts.Load() != 2 {
		t.Fatal("completed launch restarted workers")
	}
	candidate, job, err := reopened.OutputOwner(identity.Principal)
	check(t, err)
	if _, _, err := reopened.OutputOwner(strings.Repeat("f", 32)); err == nil {
		t.Fatal("foreign controller output accepted")
	}
	client, err := NewNodeClient(controller, candidate.Endpoint, candidate.Capabilities.Node, clock)
	check(t, err)
	actor := trust.AuthenticatedPrincipal{Cluster: identity.Cluster, Principal: strings.Repeat("f", 32), Role: trust.Controller, NotBefore: testNow.Add(-time.Minute), ExpiresAt: testNow.Add(time.Minute)}
	if _, _, err := apps[candidate.Capabilities.Node].Output(ctx, actor, job, nil); err == nil {
		t.Fatal("foreign controller read output")
	}
	ref, err := client.FetchOutput(ctx, job, launchPath, "model.safetensors")
	check(t, err)
	if ref != outputRef {
		t.Fatal("output identity changed")
	}
	b, err := launchPath.ReadFileLimit("model.safetensors", 2<<20)
	check(t, err)
	if !bytes.Equal(b, payload) {
		t.Fatal("download bytes differ")
	}
	if doResume {
		source, err := reopened.CheckpointSource(identity.Principal)
		check(t, err)
		if _, err := reopened.CheckpointSource(strings.Repeat("f", 32)); err == nil {
			t.Fatal("foreign checkpoint source exposed")
		}
		options.ResumeSource = &source
		options.ResumeArtifact = &nodejob.ArtifactRef{SHA256: outputRef.SHA256, Bytes: outputRef.Bytes, Kind: "checkpoint"}
		options.OpenResume = func() (io.ReadSeekCloser, error) { return launchPath.OpenRead("model.safetensors") }
		options.CheckpointAt = 4
		resumePorts, err := RecruitmentPorts(options)
		check(t, err)
		selection, err := recruitment.Select(ctx, hints, requirements, lookup, clock)
		check(t, err)
		attempt, err := controllerRandomID()
		check(t, err)
		resumed, err := recruitment.BeginLaunch(resolve("resumed", statehome.Principal), recruitment.LaunchPlan{Cluster: identity.Cluster, Controller: identity.Principal, Run: run, Attempt: attempt, Selection: selection, TTLSeconds: 300})
		check(t, err)
		originalPrepare := resumePorts.Prepare
		resumePorts.Prepare = func(ctx context.Context, c recruitment.Candidate, version uint64, s nodejob.Signed) (nodeagent.PreparedWorkload, error) {
			app := apps[c.Capabilities.Node]
			badActor := actor
			if _, err := app.Input(ctx, badActor, NodeInputRequest{Signed: s}); err == nil {
				t.Fatal("foreign input upload accepted")
			}
			n, err := NewNodeClient(controller, c.Endpoint, c.Capabilities.Node, clock)
			check(t, err)
			f, err := options.OpenResume()
			check(t, err)
			check(t, n.UploadCheckpoint(ctx, s, f))
			check(t, f.Close())
			var receipt NodeInputReceipt
			changed := bytes.Clone(payload[:outputChunkBytes])
			changed[0] ^= 1
			err = n.request(ctx, http.MethodPut, "/v1/agent/jobs/"+s.Manifest.Job+"/input", NodeInputRequest{Signed: s, Data: changed}, &receipt)
			if err == nil {
				t.Fatal("changed input retry accepted")
			}
			// The normal prepare path repeats the exact upload and commit safely.
			out, err := originalPrepare(ctx, c, version, s)
			if err == nil {
				if !reflect.DeepEqual(s.Manifest.Membership, source.Membership) || s.Manifest.Attempt == source.Attempt {
					t.Fatal("resume topology/attempt changed incorrectly")
				}
			}
			return out, err
		}
		check(t, resumed.Run(ctx, resumePorts))
		if starts.Load() != 4 {
			t.Fatal("resume duplicated or omitted workers", starts.Load())
		}
		for _, s := range stores {
			available, err := s.Availability(clock())
			check(t, err)
			if !available.Available {
				t.Fatal("resume retained busy lease")
			}
		}
		return
	}
	_, err = client.FetchOutput(ctx, job, launchPath, "model.safetensors")
	check(t, err)
	for _, q := range []NodeOutputRequest{{Ref: ref, Bytes: outputChunkBytes + 1}, {Ref: ref, Offset: ref.Bytes, Bytes: 1}, {Ref: ref, Offset: ^uint64(0), Bytes: 1}, {Ref: artifact.Ref{SHA256: strings.Repeat("f", 64), Bytes: ref.Bytes}, Bytes: 1}} {
		var out NodeOutputChunk
		if err := client.request(ctx, http.MethodPost, "/v1/agent/jobs/"+job+"/output", q, &out); err == nil {
			t.Fatal("invalid range accepted", q)
		}
	}
	check(t, launchPath.WriteFile("model.safetensors", []byte("corrupt")))
	if _, err := client.FetchOutput(ctx, job, launchPath, "model.safetensors"); err == nil {
		t.Fatal("corrupt existing output overwritten")
	}
	jobStatus, err := client.JobStatus(ctx, job)
	check(t, err)
	stored, _, err := apps[candidate.Capabilities.Node].options.Output(ctx, job, jobStatus.Attempt)
	check(t, err)
	bad := bytes.Clone(payload)
	bad[0] ^= 1
	check(t, stored.WriteFile(ref.SHA256, bad))
	if _, err := client.FetchOutput(ctx, job, launchPath, "corrupt-download.safetensors"); err == nil {
		t.Fatal("corrupted remote artifact published")
	}
	if _, err := launchPath.OpenRead("corrupt-download.safetensors"); err == nil {
		t.Fatal("failed checksum left published download")
	}
	// A stale node accepts only a newer root-authorized signed snapshot. Pushing
	// controller revocation must work without that controller's TLS credentials.
	old, err := a.Snapshots.Load(clock())
	check(t, err)
	assertSnapshotRouteIsolation(t, ctx, candidate.Endpoint, clock, old)
	currentTime.Store(testNow.Add(trust.SnapshotLifetime + time.Second).Unix())
	for _, p := range principals {
		if _, _, err := p.Active(clock()); err == nil {
			t.Fatal("fixture node trust should be stale")
		}
	}
	next, err := a.Snapshots.Revoke(ctx, "principal", identity.Principal, "prospective", "push test", clock())
	check(t, err)
	for _, candidate := range selection.Selected {
		receipt, err := PushNodeSnapshot(ctx, a.Anchor, next, candidate.Endpoint, clock)
		check(t, err)
		if receipt.Node != candidate.Capabilities.Node || receipt.Generation != next.Payload.Generation {
			t.Fatal("push receipt binding")
		}
		_, _, err = principals[receipt.Node].Active(clock())
		check(t, err)
		_, err = PushNodeSnapshot(ctx, a.Anchor, next, candidate.Endpoint, clock)
		check(t, err)
		if _, err := PushNodeSnapshot(ctx, a.Anchor, old, candidate.Endpoint, clock); err == nil {
			t.Fatal("stale snapshot push accepted")
		}
	}
	// The same stale-trust condition is recoverable before a restarted node
	// becomes recruitable; this does not change its key or leaf certificate.
	currentTime.Store(clock().Add(trust.SnapshotLifetime + time.Second).Unix())
	for _, p := range principals {
		before, err := p.View(clock())
		check(t, err)
		check(t, RefreshNodeStartup(ctx, p, clock))
		after, _, err := p.Active(clock())
		check(t, err)
		if before.Key.ID != after.Key.ID || !bytes.Equal(before.Chain[0], after.Chain[0]) {
			t.Fatal("startup snapshot refresh changed identity")
		}
	}
	// Refreshing the controller cannot undo its own revocation.
	check(t, RefreshRemoteTrust(ctx, controller, endpoint, clock))
	if _, err := client.JobStatus(ctx, job); err == nil {
		t.Fatal("revoked controller reached normal node route")
	}
	next, err = a.Snapshots.Revoke(ctx, "principal", candidate.Capabilities.Node, "prospective", "retire node", clock())
	check(t, err)
	_, err = PushNodeSnapshot(ctx, a.Anchor, next, candidate.Endpoint, clock)
	check(t, err)
	if err := RefreshNodeStartup(ctx, principals[candidate.Capabilities.Node], clock); err == nil {
		t.Fatal("startup refresh reactivated a revoked node")
	}
}

type recruitmentRunner struct {
	starts        *atomic.Int32
	allStarted    chan struct{}
	resumeStarted chan struct{}
	waitForCancel bool
	output        func(context.Context, contract.Approved) error
}

func (r *recruitmentRunner) Run(ctx context.Context, a contract.Approved, started func(int) error) error {
	if err := started(1234); err != nil {
		return err
	}
	n := r.starts.Add(1)
	if n == 2 {
		close(r.allStarted)
	}
	ready := r.allStarted
	if a.Assignment.Resume != nil {
		ready = r.resumeStarted
		if n == 4 {
			close(ready)
		}
	}
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-ready:
		if r.waitForCancel {
			<-ctx.Done()
			return ctx.Err()
		}
		return r.output(ctx, a)
	}
}
func (*recruitmentRunner) Reconcile(context.Context, workerhost.Attempt) (bool, error) {
	return true, nil
}

func recruitmentFreeAddress(t *testing.T) string {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	check(t, err)
	address := l.Addr().String()
	check(t, l.Close())
	return address
}
func recruitmentFreePort(t *testing.T) int {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	check(t, err)
	port := l.Addr().(*net.TCPAddr).Port
	check(t, l.Close())
	return port
}
