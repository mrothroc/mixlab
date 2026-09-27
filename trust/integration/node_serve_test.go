package integration

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/transport/managedtls"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/workerhost"
)

type delayedNodeJobs struct {
	clusterapp.NodeJobOperations
	entered, resume chan struct{}
}

func (j delayedNodeJobs) Prepare(ctx context.Context, actor trust.AuthenticatedPrincipal, q clusterapp.NodePrepareRequest) (nodeagent.PreparedWorkload, error) {
	close(j.entered)
	select {
	case <-ctx.Done():
		return nodeagent.PreparedWorkload{}, ctx.Err()
	case <-j.resume:
	}
	return j.NodeJobOperations.Prepare(ctx, actor, q)
}

func TestNodeServerDrainsPreparationBeforeRuntimeCleanup(t *testing.T) {
	x := newNodeJobsFixture(t)
	host, err := workerhost.NewAttemptStore(runtimePath(t, "worker"))
	check(t, err)
	runner := &runtimeTestRunner{}
	l, err := net.Listen("tcp4", "127.0.0.1:0")
	check(t, err)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	ready, done := make(chan struct{}), make(chan error, 1)
	jobs := delayedNodeJobs{NodeJobOperations: x.app, entered: make(chan struct{}), resume: make(chan struct{})}
	go func() {
		done <- clusterapp.ServeNode(ctx, l, clusterapp.NodeServerOptions{Store: x.store, Jobs: jobs,
			Policy:      func() (*managedtls.Policy, error) { return x.policy, nil },
			Maintenance: func(ctx context.Context) error { <-ctx.Done(); return nil },
			Ready:       func() { close(ready) },
			Execution: clusterapp.NodeExecutionPorts{
				Open: func(context.Context, string, string) (*workerhost.AttemptStore, workerhost.Runner, error) {
					return host, runner, nil
				},
				Authorize: x.app.Authorize, Cleanup: x.app.Cleanup, Clock: x.f.clock,
			},
		})
	}()
	joined := false
	t.Cleanup(func() {
		cancel()
		if !joined {
			select {
			case <-done:
			case <-time.After(20 * time.Second):
				t.Error("server leaked")
			}
		}
	})
	select {
	case <-ready:
	case err := <-done:
		joined = true
		t.Fatal(err)
	case <-time.After(5 * time.Second):
		t.Fatal("server not ready")
	}
	client, err := x.client.HTTPClient(5 * time.Second)
	check(t, err)
	defer client.CloseIdleConnections()
	resp, err := client.Get("https://" + l.Addr().String() + "/v1/agent/capabilities")
	check(t, err)
	_, err = io.Copy(io.Discard, resp.Body)
	check(t, err)
	check(t, resp.Body.Close())
	if resp.StatusCode != http.StatusOK {
		t.Fatal(resp.Status)
	}
	lease, err := x.store.Reserve(x.ctx, x.actor, nodeagent.Reserve{IdempotencyKey: id(t), ExpectedNodeVersion: 1, CapabilityGeneration: 1, Run: id(t), TTLSeconds: 300}, x.f.now)
	check(t, err)
	signed := signedJobFixture(t, x.f, x.options.Node, lease)
	signed.Manifest.DatasetSelector = "toy"
	q, err := signed.Manifest.SigningRequest()
	check(t, err)
	signed.Proof, err = trust.SignPrincipalProof(x.f.a, x.f.client.Chain, x.f.client.Key, x.f.view, q, x.f.now)
	check(t, err)
	b, err := json.Marshal(clusterapp.NodePrepareRequest{ExpectedLeaseVersion: lease.Version, Signed: signed})
	check(t, err)
	response := make(chan int, 1)
	go func() {
		r, err := http.NewRequest(http.MethodPut, "https://"+l.Addr().String()+"/v1/agent/jobs/"+signed.Manifest.Job+"/prepare", bytes.NewReader(b))
		if err != nil {
			response <- 0
			return
		}
		r.Header.Set("Content-Type", "application/json")
		resp, err := client.Do(r)
		if err != nil {
			response <- 0
			return
		}
		_, _ = io.Copy(io.Discard, resp.Body)
		_ = resp.Body.Close()
		response <- resp.StatusCode
	}()
	select {
	case <-jobs.entered:
	case <-time.After(5 * time.Second):
		t.Fatal("prepare did not enter")
	}
	cancel()
	close(jobs.resume)
	if code := <-response; code != http.StatusOK {
		t.Fatal("in-flight preparation not drained", code)
	}
	select {
	case err := <-done:
		joined = true
		check(t, err)
	case <-time.After(10 * time.Second):
		t.Fatal("server failed to join cleanup")
	}
	av, err := x.store.Availability(x.f.now)
	check(t, err)
	if !av.Available || runner.calls.Load() != 0 {
		t.Fatal("shutdown did not fence/drain prepared-only job")
	}
}
