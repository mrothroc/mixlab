//go:build darwin || linux

package clusterapp

import (
	"context"
	"errors"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/workerprobe"
)

func TestLearnerPreparationHTTPDisconnectRejoinsAuthenticatedOperation(t *testing.T) {
	f := newLearnerRuntimeFixtureMode(t, false)
	entered, release := make(chan struct{}), make(chan struct{})
	var probes atomic.Int32
	life, stop := context.WithCancel(context.Background())
	defer stop()
	app := preparationApp(t, f, life, func(ctx context.Context, _ []byte) (workerprobe.Plan, error) {
		if probes.Add(1) == 1 {
			close(entered)
		}
		select {
		case <-release:
			return preparationNumerical(f), nil
		case <-ctx.Done():
			return workerprobe.Plan{}, ctx.Err()
		}
	})
	policy, err := NodePrincipalPolicy(f.remote.runtime.Principal, app.options.Clock)
	check(t, err)
	handler, err := NodeManagementHandler(app.options.Store, policy, app.options.Clock, app)
	check(t, err)
	server := httptest.NewUnstartedServer(handler)
	server.TLS = policy.ServerConfig()
	server.StartTLS()
	defer server.Close()
	client, err := NewNodeClient(f.controller, server.Listener.Addr().String(), app.options.Node, app.options.Clock)
	check(t, err)
	request, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { _, err := client.PrepareLearner(request, f.prepare); done <- err }()
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("HTTP preparation did not start")
	}
	cancel()
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("request disconnect: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	// A real authenticated control request succeeds while the owned probe is
	// still blocked, proving the request cancellation did not own the operation.
	_, err = client.LeaseStatus(ctx, f.prepare.Signed.Manifest.Slot.Allocation.LeaseID)
	check(t, err)
	close(release)
	out, err := client.PrepareLearner(ctx, f.prepare)
	check(t, err)
	check(t, out.Registration.ValidateFor(f.prepare.Signed.Manifest.Plan))
	if probes.Load() != 1 {
		t.Fatalf("HTTP retry ran %d probes", probes.Load())
	}
	changed := f.prepare
	changed.Signed.Manifest.Config = []byte(`{"model_dim":32}`)
	if _, err := client.PrepareLearner(ctx, changed); err == nil {
		t.Fatal("cached response exposed for altered manifest")
	}
}
