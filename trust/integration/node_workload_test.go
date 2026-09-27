package integration

import (
	"bytes"
	"context"
	"crypto/ecdh"
	"crypto/rand"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/nodecredentials"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/internal/certificates"
	"github.com/mrothroc/mixlab/trust/workload"
	"github.com/mrothroc/mixlab/workerhost"
)

type workloadPreparation struct {
	f           *tlsFixture
	store       *nodeagent.Store
	path        statehome.Path
	actor       trust.AuthenticatedPrincipal
	job         nodeagent.Job
	ports       nodeagent.WorkloadPorts
	initialized int
	requested   int
	scope       workload.Scope
}

func preparedWorkloadFixture(t *testing.T) *workloadPreparation {
	t.Helper()
	x := &workloadPreparation{f: newTLSFixture(t)}
	f := x.f
	f.snapshot.Payload.EligibleRoles = append(f.snapshot.Payload.EligibleRoles, trust.Node)
	f.signSnapshot(t)
	node, nodeKey := id(t), key(t)
	envelope, err := ecdh.X25519().GenerateKey(rand.Reader)
	check(t, err)
	der, err := certificates.IssueNode(f.a, f.issuer, f.issuerKey, node, nodeKey.Public(), envelope.PublicKey().Bytes(), f.now)
	check(t, err)
	chain := [][]byte{der, f.issuer, f.a.DER()}
	dir, err := filepath.EvalSymlinks(t.TempDir())
	check(t, err)
	check(t, os.Chmod(dir, 0700))
	x.path, err = statehome.Resolve(statehome.Options{ExactDir: dir}, statehome.Context{Kind: statehome.Agent})
	check(t, err)
	x.store, err = nodeagent.Initialize(context.Background(), x.path, f.a.Cluster(), node, 1)
	check(t, err)
	x.actor, err = trust.AuthenticatePrincipal(f.a, f.view, f.client.Chain, f.now)
	check(t, err)
	lease, err := x.store.Reserve(context.Background(), x.actor, nodeagent.Reserve{IdempotencyKey: id(t), ExpectedNodeVersion: 1, CapabilityGeneration: 1, Run: id(t), TTLSeconds: 300}, f.now)
	check(t, err)
	signed := signedJobFixture(t, f, node, lease)
	accepted, err := nodejob.Accept(f.a, f.view, x.actor, node, signed, f.now)
	check(t, err)
	x.job, err = x.store.PrepareJob(context.Background(), x.actor, accepted, lease.Version, f.now)
	check(t, err)
	credentialPath, err := statehome.Resolve(statehome.Options{ExactDir: filepath.Join(dir, "credential")}, statehome.Context{Kind: statehome.Agent})
	check(t, err)
	x.ports = nodeagent.WorkloadPorts{
		Clock: f.clock,
		Initialize: func(ctx context.Context, scope workload.Scope) error {
			x.initialized++
			x.scope = scope
			c, err := nodecredentials.InitializeTransport(ctx, credentialPath, f.a, scope, "file")
			if errors.Is(err, statehome.ErrExists) {
				c, err = nodecredentials.OpenTransport(credentialPath, f.a, scope)
			}
			if err != nil {
				return err
			}
			return c.Close()
		},
		Request: func(ctx context.Context, scope workload.Scope) (workload.SignedRequest, error) {
			x.requested++
			c, err := nodecredentials.OpenTransport(credentialPath, f.a, scope)
			if err != nil {
				return workload.SignedRequest{}, err
			}
			defer func() { _ = c.Close() }()
			return c.PrepareRequest(ctx, func(_ context.Context, q trust.SignRequest) (trust.SignedProof, error) {
				return trust.SignPrincipalProof(f.a, chain, nodeKey, f.view, q, f.now)
			}, f.view, f.now)
		},
	}
	return x
}

func TestPreparedWorkloadExactRetryAndCancelFence(t *testing.T) {
	x := preparedWorkloadFixture(t)
	ctx := context.Background()
	if _, err := x.store.PrepareWorkload(ctx, x.actor, x.job.ID, x.job.Version+1, x.ports); err == nil || x.initialized != 0 {
		t.Fatal("wrong version created workload intent", err)
	}
	first, err := x.store.PrepareWorkload(ctx, x.actor, x.job.ID, x.job.Version, x.ports)
	check(t, err)
	if first.Job.Version <= x.job.Version || first.Request.Request.Scope.Job != x.job.ID {
		t.Fatal("workload not tied to durable prepared job")
	}
	x.store, err = nodeagent.Open(x.path, x.f.a.Cluster(), first.Request.Request.Scope.Node)
	check(t, err)
	again, err := x.store.PrepareWorkload(ctx, x.actor, x.job.ID, x.job.Version, x.ports)
	check(t, err)
	if x.initialized != 1 || again.Job != first.Job || !bytes.Equal(first.Request.Request.CSR, again.Request.Request.CSR) {
		t.Fatal("prepared retry replaced identity or advanced state")
	}
	if _, err := x.store.PrepareWorkload(ctx, x.actor, x.job.ID, first.Job.Version+1, x.ports); err == nil || x.requested != 2 {
		t.Fatal("retry accepted unobserved version", err)
	}
	_, err = x.store.CancelJob(ctx, x.actor, x.job.ID, first.Job.Version, x.f.now)
	check(t, err)
	if _, err := x.store.PrepareWorkload(ctx, x.actor, x.job.ID, x.job.Version, x.ports); err == nil || x.requested != 2 {
		t.Fatal("cancellation did not fence workload key access", err)
	}
	wp, err := statehome.Resolve(statehome.Options{ExactDir: filepath.Join(x.path.Dir(), "worker")}, statehome.Context{Kind: statehome.Worker})
	check(t, err)
	check(t, wp.Ensure())
	host, err := workerhost.NewAttemptStore(wp)
	check(t, err)
	fence, err := x.store.PreparationFence(ctx, x.job.ID)
	check(t, err)
	outcome, err := host.FencePreparation(ctx, fence)
	check(t, err)
	_, err = x.store.ApplyOutcome(ctx, outcome)
	check(t, err)
	failure := errors.New("credential cleanup uncertain")
	if err := x.store.CompleteCleanup(ctx, x.job.ID, func(context.Context, nodeagent.CleanupRequest) error { return failure }); !errors.Is(err, failure) {
		t.Fatal(err)
	}
	availability, err := x.store.Availability(x.f.now)
	check(t, err)
	if availability.Available {
		t.Fatal("uncertain credential cleanup released accelerator")
	}
	check(t, x.store.CompleteCleanup(ctx, x.job.ID, func(ctx context.Context, q nodeagent.CleanupRequest) error {
		if q.Workload != x.scope || !q.WorkloadInitialized || q.Job != x.job.ID || q.Attempt != x.job.Attempt {
			t.Fatal("terminal cleanup lost exact workload scope")
		}
		dir, err := statehome.Resolve(statehome.Options{ExactDir: filepath.Join(x.path.Dir(), "credential")}, statehome.Context{Kind: statehome.Agent})
		check(t, err)
		c, err := nodecredentials.OpenTransport(dir, x.f.a, q.Workload)
		check(t, err)
		defer func() { _ = c.Close() }()
		return c.Destroy(ctx)
	}))
	availability, err = x.store.Availability(x.f.now)
	check(t, err)
	if !availability.Available {
		t.Fatal("confirmed credential cleanup did not release accelerator")
	}
}

func TestPreparedWorkloadPublicationRecovery(t *testing.T) {
	for _, interrupted := range []string{"initialization", "request", "expired", "canceled-context"} {
		t.Run(interrupted, func(t *testing.T) {
			x := preparedWorkloadFixture(t)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			ports := x.ports
			failure := errors.New("interrupted local publication")
			if interrupted == "request" {
				ports.Request = func(context.Context, workload.Scope) (workload.SignedRequest, error) {
					return workload.SignedRequest{}, failure
				}
			} else {
				ports.Initialize = func(ctx context.Context, scope workload.Scope) error {
					if err := x.ports.Initialize(ctx, scope); err != nil {
						return err
					}
					switch interrupted {
					case "expired":
						x.f.now = x.f.now.Add(time.Minute)
						return nil
					case "canceled-context":
						cancel()
						return nil
					default:
						return failure
					}
				}
			}
			if _, err := x.store.PrepareWorkload(ctx, x.actor, x.job.ID, x.job.Version, ports); err == nil || x.requested != 0 {
				t.Fatal("interruption allowed key creation", err)
			}
			scope := x.scope
			if interrupted == "expired" || interrupted == "canceled-context" {
				return
			}
			if interrupted == "request" {
				// The ready marker was committed; losing the credential directory
				// must not re-run initialization or create a new key context.
				check(t, os.RemoveAll(filepath.Join(x.path.Dir(), "credential")))
				if _, err := x.store.PrepareWorkload(ctx, x.actor, x.job.ID, x.job.Version, x.ports); err == nil || x.initialized != 1 {
					t.Fatal("missing published credential context recreated", err)
				}
				return
			}
			out, err := x.store.PrepareWorkload(ctx, x.actor, x.job.ID, x.job.Version, x.ports)
			check(t, err)
			if x.scope != scope || out.Request.Request.Scope != scope || x.initialized != 2 || x.requested != 1 {
				t.Fatal("initialization retry changed workload identity")
			}
		})
	}
}
