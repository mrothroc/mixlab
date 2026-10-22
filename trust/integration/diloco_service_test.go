package integration

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"slices"
	"strings"
	"sync/atomic"
	"syscall"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/admission"
	"github.com/mrothroc/mixlab/coordinator"
	coordinatorlocal "github.com/mrothroc/mixlab/coordinator/local"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/recruitment"
	"github.com/mrothroc/mixlab/transport/managedtls"
	"github.com/mrothroc/mixlab/trust"
)

func serviceSubmission(t *testing.T, f *tlsFixture) admission.Submission {
	t.Helper()
	q := admission.Submission{Owner: f.clientID, Deadline: f.now.Add(10 * time.Minute).Unix()}
	q.Launch = recruitment.LaunchPlan{Cluster: f.a.Cluster(), Controller: id(t), Run: id(t), Attempt: id(t), TTLSeconds: 300, Selection: recruitment.Selection{Policy: recruitment.SelectionPolicy}}
	actor, err := trust.AuthenticatePrincipal(f.a, f.view, f.client.Chain, f.now)
	check(t, err)
	for range 2 {
		n, _, _, r := capabilityNode(t, f)
		c, err := n.Capabilities(context.Background(), actor, f.now)
		check(t, err)
		q.Launch.Selection.Requirements = r
		q.Launch.Selection.Selected = append(q.Launch.Selection.Selected, recruitment.Candidate{Endpoint: "node.local:7445", Capabilities: c})
	}
	slices.SortFunc(q.Launch.Selection.Selected, func(a, b recruitment.Candidate) int {
		if a.Capabilities.Node < b.Capabilities.Node {
			return -1
		}
		return 1
	})
	var p run.Plan
	b, err := os.ReadFile("../../diloco/testdata/plan.json")
	check(t, err)
	check(t, json.Unmarshal(b, &p))
	config := json.RawMessage(`{"model_dim":16}`)
	p.Contract.ConfigHash = nodejob.Hash(config)
	p.Contract.DatasetHash = q.Launch.Selection.Requirements.DatasetID
	p.Contract.BuildHash = q.Launch.Selection.Requirements.BuildID
	q.Specification = admission.Specification{Contract: p.Contract, BaseModel: p.BaseModel, Outer: p.Outer, Coordinator: run.Coordinator{ClusterID: f.a.Cluster(), PrincipalID: f.serverID, Role: "coordinator", Audience: "learner-run"}, ArtifactStoreID: "test", Endpoint: "https://coordinator.local:7446", Config: config, CreatedAt: f.now.Unix()}
	check(t, q.Validate())
	return q
}

func TestDiLoCoSubmissionDurableHTTPAndOwnership(t *testing.T) {
	f := newTLSFixture(t)
	f.server = f.leaf(t, trust.Coordinator, f.serverID)
	q := serviceSubmission(t, f)
	inbox, err := admission.OpenInbox(artifactPath(t))
	check(t, err)
	repo, err := coordinatorlocal.Open(artifactPath(t))
	check(t, err)
	verify := func(chain [][]byte, now time.Time) (trust.AuthenticatedPrincipal, error) {
		f.mu.RLock()
		defer f.mu.RUnlock()
		return trust.AuthenticatePrincipal(f.a, f.view, chain, now)
	}
	sp, err := managedtls.New(f.server, verify, f.clock)
	check(t, err)
	cp, err := managedtls.New(f.client, f.verify(trust.Coordinator, f.serverID), f.clock)
	check(t, err)
	var preflights atomic.Int32
	a := clusterapp.DiLoCoSubmissionAPI{Inbox: inbox, Repository: repo, Policy: sp, ControllerID: q.Launch.Controller, CoordinatorID: f.serverID, Cluster: f.a.Cluster(), Clock: f.clock, Verify: func(context.Context, admission.Submission) error { preflights.Add(1); return nil }}
	s := serveCapabilities(t, sp, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		h, e := a.Handler()
		if e != nil {
			t.Error(e)
			http.Error(w, "setup", 500)
			return
		}
		h.ServeHTTP(w, r)
	}))
	a.Endpoint = s.URL
	q.Specification.Endpoint = s.URL
	c, err := clusterapp.NewDiLoCoClientWithPolicy(s.URL, f.serverID, func() (*managedtls.Policy, error) { return cp, nil })
	check(t, err)
	i, err := c.Info(context.Background())
	check(t, err)
	if i.Controller != q.Launch.Controller {
		t.Fatal("wrong admission principal")
	}
	request, cancel := context.WithCancel(context.Background())
	ack, err := c.Submit(request, q)
	check(t, err)
	cancel()
	if ack.Admission != "ACCEPTED" {
		t.Fatal(ack)
	}
	got, err := inbox.Read(context.Background(), q.Launch.Run)
	check(t, err)
	if got.Owner != q.Owner {
		t.Fatal("lost owner")
	}
	status, err := c.Status(context.Background(), q.Launch.Run, false)
	check(t, err)
	if status.Admission != "ADMITTING" {
		t.Fatal(status)
	}
	// Drive an actual service effect through the node-client error boundary,
	// persist it, and read the public status over authenticated HTTP.
	nodeID := q.Launch.Selection.Selected[0].Capabilities.Node
	refusing, err := clusterapp.NewNodeClientWithPolicy("127.0.0.1:7445", nodeID, func() (*managedtls.Policy, error) {
		return nil, fmt.Errorf("PRIVATE-TLS-DETAIL: %w", &net.OpError{Op: "dial", Net: "tcp", Err: syscall.EHOSTUNREACH})
	})
	check(t, err)
	life, stop := context.WithCancel(context.Background())
	service := clusterapp.DiLoCoService{Inbox: inbox, Clock: f.clock, Interval: 10 * time.Millisecond,
		Maintain: func(ctx context.Context, _ admission.Submission, _ admission.Snapshot) error {
			_, e := refusing.Reserve(ctx, nodeagent.Reserve{})
			return e
		},
		Admission: func(context.Context, admission.Submission) (admission.Ports, error) {
			return admission.Ports{}, errors.New("unexpected forward effect")
		},
		Run: func(context.Context, admission.Submission, admission.Snapshot) error {
			return errors.New("unexpected run")
		},
	}
	done := make(chan error, 1)
	go func() { done <- service.Serve(life, nil) }()
	func() {
		defer func() {
			stop()
			if e := <-done; !errors.Is(e, context.Canceled) {
				t.Errorf("service shutdown: %v", e)
			}
		}()
		until := time.Now().Add(5 * time.Second)
		for time.Now().Before(until) {
			status, err = c.Status(context.Background(), q.Launch.Run, false)
			check(t, err)
			if status.ServiceError {
				break
			}
			time.Sleep(10 * time.Millisecond)
		}
		if status.ServiceNode != nodeID || status.ServiceFailure == nil || status.ServiceFailure.Reason != "route_or_permission_denied" || status.NextRetryAt <= status.ObservedAt {
			t.Fatalf("missing node failure over HTTP: %+v", status)
		}
		payload, e := json.Marshal(status)
		check(t, e)
		if strings.Contains(string(payload), "PRIVATE-TLS-DETAIL") || !strings.Contains(string(payload), "allowlist") {
			t.Fatalf("unsafe/unhelpful status: %s", payload)
		}
	}()
	// Retry an exact lost acknowledgement after admission expires. It must
	// not re-run preflight or change the accepted request.
	f.mu.Lock()
	f.now = time.Unix(q.Deadline+1, 0)
	f.snapshot.Payload.Generation++
	f.snapshot.Payload.IssuedAt = f.now.Unix()
	f.snapshot.Payload.ExpiresAt = f.now.Add(trust.SnapshotLifetime).Unix()
	f.signSnapshot(t)
	f.mu.Unlock()
	_, err = c.Submit(context.Background(), q)
	check(t, err)
	if preflights.Load() != 1 {
		t.Fatal("retry repeated preflight")
	}
	changed := q
	changed.Specification.Outer.LR *= 2
	if _, err = c.Submit(context.Background(), changed); err == nil {
		t.Fatal("accepted conflicting run")
	}
	other := f.leaf(t, trust.Controller, id(t))
	op, err := managedtls.New(other, f.verify(trust.Coordinator, f.serverID), f.clock)
	check(t, err)
	oc, err := clusterapp.NewDiLoCoClientWithPolicy(s.URL, f.serverID, func() (*managedtls.Policy, error) { return op, nil })
	check(t, err)
	if _, err = oc.Status(context.Background(), q.Launch.Run, false); err == nil {
		t.Fatal("another owner read run")
	}
	if _, err = oc.Status(context.Background(), q.Launch.Run, true); err == nil {
		t.Fatal("another owner canceled run")
	}
	status, err = c.Status(context.Background(), q.Launch.Run, true)
	check(t, err)
	if status.Admission != "ABORTING" {
		t.Fatal(status)
	}
	if !status.AwaitingNode {
		t.Fatal("pending compensation not reported")
	}
}

func TestDiLoCoServiceJoinsEffectsAndRestartsWithoutCompensation(t *testing.T) {
	f := newTLSFixture(t)
	q := serviceSubmission(t, f)
	i, err := admission.OpenInbox(artifactPath(t))
	check(t, err)
	check(t, i.Accept(context.Background(), q))
	for range 2 {
		ctx, cancel := context.WithCancel(context.Background())
		entered, exited := make(chan struct{}), make(chan struct{})
		s := clusterapp.DiLoCoService{Inbox: i, Clock: f.clock, Interval: 10 * time.Millisecond, Admission: func(ctx context.Context, _ admission.Submission) (admission.Ports, error) {
			close(entered)
			<-ctx.Done()
			close(exited)
			return admission.Ports{}, ctx.Err()
		}, Run: func(context.Context, admission.Submission, admission.Snapshot) error {
			return errors.New("unexpected run")
		}}
		done := make(chan error, 1)
		go func() { done <- s.Serve(ctx, nil) }()
		select {
		case <-entered:
		case <-time.After(5 * time.Second):
			t.Fatal("service did not resume accepted intent")
		}
		cancel()
		select {
		case e := <-done:
			if !errors.Is(e, context.Canceled) {
				t.Fatal(e)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("service failed to join")
		}
		select {
		case <-exited:
		default:
			t.Fatal("effect outlived service")
		}
		saga, err := i.Resume(context.Background(), q.Launch.Run)
		check(t, err)
		v, err := saga.Snapshot(context.Background())
		check(t, err)
		if v.Aborting || v.Done {
			t.Fatal("service shutdown canceled run")
		}
	}
}

func TestDiLoCoServerUnexpectedListenerExitStopsLifecycle(t *testing.T) {
	f := newTLSFixture(t)
	sp, _ := f.policies(t)
	i, err := admission.OpenInbox(artifactPath(t))
	check(t, err)
	l, err := net.Listen("tcp", "127.0.0.1:0")
	check(t, err)
	ready := make(chan struct{})
	s := clusterapp.DiLoCoService{Inbox: i, Clock: f.clock, Interval: 10 * time.Millisecond, Admission: func(context.Context, admission.Submission) (admission.Ports, error) {
		return admission.Ports{}, errors.New("unused")
	}, Run: func(context.Context, admission.Submission, admission.Snapshot) error { return nil }}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() {
		done <- clusterapp.ServeDiLoCo(ctx, l, clusterapp.DiLoCoServerOptions{Service: s, Policy: func() (*managedtls.Policy, error) { return sp, nil }, Handler: func(*managedtls.Policy) (http.Handler, error) { return http.NotFoundHandler(), nil }, Maintenance: func(ctx context.Context) error { <-ctx.Done(); return ctx.Err() }, Ready: func() { close(ready) }})
	}()
	select {
	case <-ready:
	case <-time.After(5 * time.Second):
		t.Fatal("server not ready")
	}
	check(t, l.Close())
	select {
	case e := <-done:
		if e == nil {
			t.Fatal("listener failure silently succeeded")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("dead listener left coordinator running")
	}
}

var _ coordinator.Repository = (*coordinatorlocal.Repository)(nil)
