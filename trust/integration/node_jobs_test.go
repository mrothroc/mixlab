package integration

import (
	"context"
	"crypto/ecdh"
	"crypto/rand"
	"errors"
	"net"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/grouptransport"
	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/nodecredentials"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/transport/managedtls"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/internal/certificates"
	"github.com/mrothroc/mixlab/trust/workload"
	"github.com/mrothroc/mixlab/workerhost"
	"github.com/mrothroc/mixlab/workerprobe"
)

type nodeJobsFixture struct {
	f       *tlsFixture
	store   *nodeagent.Store
	app     *clusterapp.NodeJobs
	actor   trust.AuthenticatedPrincipal
	root    statehome.Path
	options clusterapp.NodeJobOptions
	ctx     context.Context
	cancel  context.CancelFunc
	policy  *managedtls.Policy
	client  *managedtls.Policy
}

func freeRelayAddress(t *testing.T) string {
	t.Helper()
	l, err := net.Listen("tcp4", "127.0.0.1:0")
	check(t, err)
	a := l.Addr().String()
	check(t, l.Close())
	return a
}

func newNodeJobsFixture(t *testing.T) *nodeJobsFixture {
	t.Helper()
	f := newTLSFixture(t)
	s, policy, client, _ := capabilityNode(t, f)
	f.snapshot.Payload.EligibleRoles = append(f.snapshot.Payload.EligibleRoles, trust.Worker)
	f.signSnapshot(t)
	actor, err := trust.AuthenticatePrincipal(f.a, f.view, f.client.Chain, f.now)
	check(t, err)
	capability, err := s.Capabilities(context.Background(), actor, f.now)
	check(t, err)
	nodeKey := key(t)
	envelope, err := ecdh.X25519().GenerateKey(rand.Reader)
	check(t, err)
	der, err := certificates.IssueNode(f.a, f.issuer, f.issuerKey, capability.Node, nodeKey.Public(), envelope.PublicKey().Bytes(), f.now)
	check(t, err)
	chain := [][]byte{der, f.issuer, f.a.DER()}
	x := &nodeJobsFixture{f: f, store: s, actor: actor, root: runtimePath(t, statehome.Agent), policy: policy, client: client}
	x.ctx, x.cancel = context.WithCancel(context.Background())
	t.Cleanup(x.cancel)
	path := func(scope workload.Scope) statehome.Path {
		p, err := statehome.Resolve(statehome.Options{ExactDir: filepath.Join(x.root.Dir(), "job-"+scope.Job+"-"+scope.Attempt)}, statehome.Context{Kind: statehome.Agent})
		check(t, err)
		return p
	}
	x.options = clusterapp.NodeJobOptions{Store: s, Node: capability.Node, Anchor: f.a, CredentialRoot: x.root, RelayAddress: freeRelayAddress(t), Clock: f.clock,
		CurrentTrust: func(ctx context.Context, now time.Time) (trust.Anchor, trust.VerifiedSnapshot, error) {
			f.mu.RLock()
			defer f.mu.RUnlock()
			v, err := trust.VerifySnapshot(f.a, f.snapshot, now)
			if err == nil {
				_, err = trust.AuthenticatePrincipal(f.a, v, chain, now)
			}
			return f.a, v, errors.Join(err, ctx.Err())
		},
		Preparation: nodeagent.PreparationPorts{Clock: f.clock,
			Probe: func(context.Context) (workerprobe.Report, error) { return capability.Probe, nil },
			DatasetID: func(_ context.Context, p string) (string, error) {
				if p != "/private/local/train.bin" {
					return "", errors.New("unexpected remote path")
				}
				return capability.Datasets[0].ID, nil
			},
			ArtifactPresent: func(context.Context, nodejob.ArtifactRef) error { return errors.New("no artifacts in fixture") },
		},
		Workload: nodeagent.WorkloadPorts{Clock: f.clock,
			Initialize: func(ctx context.Context, scope workload.Scope) error {
				c, err := nodecredentials.InitializeTransport(ctx, path(scope), f.a, scope, "file")
				if errors.Is(err, statehome.ErrExists) {
					c, err = nodecredentials.OpenTransport(path(scope), f.a, scope)
				}
				if err != nil {
					return err
				}
				return c.Close()
			},
			Request: func(ctx context.Context, scope workload.Scope) (workload.SignedRequest, error) {
				c, err := nodecredentials.OpenTransport(path(scope), f.a, scope)
				if err != nil {
					return workload.SignedRequest{}, err
				}
				defer func() { _ = c.Close() }()
				f.mu.RLock()
				defer f.mu.RUnlock()
				return c.PrepareRequest(ctx, func(_ context.Context, q trust.SignRequest) (trust.SignedProof, error) {
					return trust.SignPrincipalProof(f.a, chain, nodeKey, f.view, q, f.now)
				}, f.view, f.now)
			},
		},
	}
	x.app, err = clusterapp.NewNodeJobs(x.ctx, x.options)
	check(t, err)
	return x
}

func (x *nodeJobsFixture) prepare(t *testing.T) (nodeagent.PreparedWorkload, nodejob.Manifest) {
	t.Helper()
	f := x.f
	lease, err := x.store.Reserve(x.ctx, x.actor, nodeagent.Reserve{IdempotencyKey: id(t), ExpectedNodeVersion: 1, CapabilityGeneration: 1, Run: id(t), TTLSeconds: 300}, f.now)
	check(t, err)
	signed := signedJobFixture(t, f, x.options.Node, lease)
	signed.Manifest.DatasetSelector = "toy"
	q, err := signed.Manifest.SigningRequest()
	check(t, err)
	signed.Proof, err = trust.SignPrincipalProof(f.a, f.client.Chain, f.client.Key, f.view, q, f.now)
	check(t, err)
	req := clusterapp.NodePrepareRequest{ExpectedLeaseVersion: lease.Version, Signed: signed}
	p, err := x.app.Prepare(x.ctx, x.actor, req)
	check(t, err)
	again, err := x.app.Prepare(x.ctx, x.actor, req)
	check(t, err)
	if p.Job != again.Job || string(p.Request.Request.CSR) != string(again.Request.Request.CSR) {
		t.Fatal("prepare retry changed workload")
	}
	return p, signed.Manifest
}

func (x *nodeJobsFixture) transportRequest(t *testing.T, p nodeagent.PreparedWorkload, m nodejob.Manifest) clusterapp.NodeTransportRequest {
	t.Helper()
	f := x.f
	f.mu.Lock()
	defer f.mu.Unlock()
	q, err := p.Request.GrantRequest()
	check(t, err)
	proof, err := trust.SignPrincipalProof(f.a, f.client.Chain, f.client.Key, f.view, q, f.now)
	check(t, err)
	grant := workload.Grant{Request: p.Request, Proof: proof}
	issuer, err := workload.Initialize(x.ctx, runtimePath(t, statehome.Authority), workload.Authority{Anchor: f.a, Issuer: f.issuer, Key: f.issuerKey}, f.view, f.now)
	check(t, err)
	result, err := issuer.Issue(x.ctx, f.client.Chain, grant, f.view, f.now)
	check(t, err)
	sp := signedTransportFixture(t, f, m)
	local := &sp.Plan.Members[m.Rank]
	local.Chain, local.Binding, local.CertificateHash, local.Endpoint = result.Chain, result.Binding, nodejob.Hash(result.Chain[0]), x.options.RelayAddress
	for i := range sp.Plan.LoopbackPorts {
		_, port, err := net.SplitHostPort(freeRelayAddress(t))
		check(t, err)
		sp.Plan.LoopbackPorts[i], err = strconv.Atoi(port)
		check(t, err)
	}
	q, err = sp.Plan.SigningRequest()
	check(t, err)
	sp.Proof, err = trust.SignPrincipalProof(f.a, f.client.Chain, f.client.Key, f.view, q, f.now)
	check(t, err)
	return clusterapp.NodeTransportRequest{ExpectedVersion: p.Job.Version, Signed: sp, Grant: grant, Credential: result}
}

func TestNodeJobsComposedExecutionAndCleanup(t *testing.T) {
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
	j, err := x.app.Transport(x.ctx, x.actor, p.Job.ID, q)
	check(t, err)
	again, err := x.app.Transport(x.ctx, x.actor, p.Job.ID, q)
	check(t, err)
	if again != j {
		t.Fatal("transport retry changed job")
	}
	start := nodeagent.StartCommand{IdempotencyKey: id(t), Job: j.ID, ExpectedVersion: j.Version}
	_, err = x.app.Start(x.ctx, x.actor, start)
	check(t, err)
	select {
	case <-runner.started:
	case <-time.After(5 * time.Second):
		t.Fatal("managed worker not started")
	}
	close(runner.finish)
	deadline := time.Now().Add(5 * time.Second)
	for {
		av, err := x.store.Availability(x.f.now)
		check(t, err)
		if av.Available {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("completed job did not join relay and release")
		}
		time.Sleep(10 * time.Millisecond)
	}
	if _, err := x.app.Start(x.ctx, x.actor, start); err == nil {
		t.Fatal("terminal job restarted")
	}
	if _, err := x.app.Transport(x.ctx, x.actor, p.Job.ID, q); err == nil {
		t.Fatal("terminal credential/relay revived")
	}
	dir, err := statehome.Resolve(statehome.Options{ExactDir: filepath.Join(x.root.Dir(), "job-"+m.Job+"-"+m.Attempt)}, statehome.Context{Kind: statehome.Agent})
	check(t, err)
	c, err := nodecredentials.OpenTransport(dir, x.f.a, p.Request.Request.Scope)
	check(t, err)
	defer func() { _ = c.Close() }()
	if _, _, err := c.Identity(x.f.view, x.f.now); err == nil {
		t.Fatal("terminal cleanup left a usable key")
	}
	l, err := net.Listen("tcp", x.options.RelayAddress)
	check(t, err)
	check(t, l.Close())
}

func TestNodeJobsRejectUnapprovedRelayAndFenceBindFailure(t *testing.T) {
	x := newNodeJobsFixture(t)
	p, m := x.prepare(t)
	q := x.transportRequest(t, p, m)
	if _, err := x.app.Start(x.ctx, x.actor, nodeagent.StartCommand{IdempotencyKey: id(t), Job: m.Job, ExpectedVersion: p.Job.Version}); err == nil {
		t.Fatal("worker started before relay")
	}
	changed := q
	changed.Signed.Plan.Members = append([]grouptransport.Member(nil), q.Signed.Plan.Members...)
	changed.Signed.Plan.Members[m.Rank].Endpoint = freeRelayAddress(t)
	req, err := changed.Signed.Plan.SigningRequest()
	check(t, err)
	changed.Signed.Proof, err = trust.SignPrincipalProof(x.f.a, x.f.client.Chain, x.f.client.Key, x.f.view, req, x.f.now)
	check(t, err)
	if _, err := x.app.Transport(x.ctx, x.actor, m.Job, changed); err == nil {
		t.Fatal("controller chose an unapproved listener")
	}
	occupied, err := net.Listen("tcp", x.options.RelayAddress)
	check(t, err)
	defer func() { _ = occupied.Close() }()
	if _, err := x.app.Transport(x.ctx, x.actor, m.Job, q); err == nil {
		t.Fatal("occupied listener accepted")
	}
	active, err := x.store.ActiveExecution(x.ctx)
	check(t, err)
	if active.Lease.State != nodeagent.Releasing || active.Transport == nil || active.Approval != nil {
		t.Fatal("partial transport activation not fenced")
	}
	fence, err := x.store.PreparationFence(x.ctx, m.Job)
	check(t, err)
	host, err := workerhost.NewAttemptStore(runtimePath(t, statehome.Worker))
	check(t, err)
	out, err := host.FencePreparation(x.ctx, fence)
	check(t, err)
	_, err = x.store.ApplyOutcome(x.ctx, out)
	check(t, err)
	check(t, x.store.CompleteCleanup(x.ctx, m.Job, x.app.Cleanup))
}
