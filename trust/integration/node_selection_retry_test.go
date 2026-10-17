package integration

import (
	"context"
	"crypto/tls"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/discovery"
	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/recruitment"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/transport/managedtls"
)

// This composes production authenticated handlers and clients over real TLS;
// it is not a full installed-service/LaunchAgent test.
func TestNodeSelectionAndInventorySlowAuthenticatedTLS(t *testing.T) {
	f := newTLSFixture(t)
	var hints []discovery.Hint
	var client *managedtls.Policy
	var requirements recruitment.Requirements
	var handshakes atomic.Int32
	for i := range 2 {
		store, policy, current, req := capabilityNode(t, f)
		client, requirements = current, req
		h, err := clusterapp.NodeCapabilitiesHandler(store, policy, f.clock)
		check(t, err)
		s := httptest.NewUnstartedServer(h)
		s.Config.ErrorLog = log.New(io.Discard, "", 0)
		s.TLS = policy.ServerConfig()
		if i == 0 {
			// The delay is inside a genuine mutual-TLS handshake on every
			// connection. Neither a retry nor cached JSON can bypass it.
			s.TLS.GetConfigForClient = func(hello *tls.ClientHelloInfo) (*tls.Config, error) {
				handshakes.Add(1)
				timer := time.NewTimer(4 * time.Second)
				defer timer.Stop()
				select {
				case <-hello.Context().Done():
					return nil, hello.Context().Err()
				case <-timer.C:
					return nil, nil
				}
			}
		}
		s.StartTLS()
		t.Cleanup(s.Close)
		hints = append(hints, discovery.Hint{Service: discovery.Node, Endpoint: strings.TrimPrefix(s.URL, "https://"), Claims: discovery.Claims{Node: "untrusted"}})
	}
	lookup := func(ctx context.Context, endpoint string) (recruitment.Observation, error) {
		return clusterapp.QueryNodeCapabilities(ctx, endpoint, client)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	for i, selectNodes := range []func(context.Context, []discovery.Hint, recruitment.Requirements, recruitment.Lookup, func() time.Time) (recruitment.Selection, error){recruitment.Select, recruitment.SelectLearners} {
		started := time.Now()
		selection, err := selectNodes(ctx, hints, requirements, lookup, f.clock)
		check(t, err)
		if len(selection.Selected) != 2 || selection.Selected[0].Capabilities.Node >= selection.Selected[1].Capabilities.Node || handshakes.Load() != int32(i+1) || time.Since(started) < 4*time.Second {
			t.Fatalf("selector %d omitted or retried slow authenticated node: selection=%+v handshakes=%d elapsed=%v", i, selection, handshakes.Load(), time.Since(started))
		}
		for _, candidate := range selection.Selected {
			if !slices.Contains(candidate.Capabilities.ManifestVersions, nodejob.LearnerVersion) || !candidate.Capabilities.Availability.Available || candidate.Capabilities.Probe.DeviceKind != "metal" {
				t.Fatal("fixture is not an available Metal learner")
			}
		}
	}
	slices.Reverse(hints)
	rows, err := recruitment.Inventory(ctx, hints, requirements.Cluster, lookup, f.clock)
	check(t, err)
	if len(rows) != 2 || rows[0].Endpoint >= rows[1].Endpoint || rows[0].Reason != "available" || rows[1].Reason != "available" || handshakes.Load() != 3 {
		t.Fatalf("inventory did not use the same authenticated slow-node policy: rows=%+v handshakes=%d", rows, handshakes.Load())
	}
}

func TestNodeLearnerSelectionWaitsForHeldStoreLock(t *testing.T) {
	f := newTLSFixture(t)
	directory := t.TempDir()
	store, policy, client, requirements := capabilityNodeIn(t, f, directory)
	directory, err := filepath.EvalSymlinks(directory)
	check(t, err)
	path, err := statehome.Resolve(statehome.Options{ExactDir: directory}, statehome.Context{Kind: statehome.Agent})
	check(t, err)
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	entered := make(chan struct{})
	acquired := make(chan struct{})
	done := make(chan struct{})
	var lockErr error
	go func() {
		defer close(done)
		lockErr = path.WithProcessLock(ctx, "node-management.lock", func() error {
			close(acquired)
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-entered:
			}
			// Start the hold after the production handler creates its store
			// context, so TLS/setup time cannot dilute the former 2s regression.
			timer := time.NewTimer(3 * time.Second)
			defer timer.Stop()
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-timer.C:
				return nil
			}
		})
	}()
	t.Cleanup(func() { cancel(); <-done })
	select {
	case <-acquired:
	case <-done:
		t.Fatalf("acquire node store lock: %v", lockErr)
	}
	var calls atomic.Int32
	var once sync.Once
	h, err := clusterapp.NodeCapabilitiesHandler(store, policy, func() time.Time {
		calls.Add(1)
		once.Do(func() { close(entered) })
		return f.clock()
	})
	check(t, err)
	lockedServer := serveCapabilities(t, policy, h)
	healthyStore, healthyPolicy, _, _ := capabilityNode(t, f)
	healthyHandler, err := clusterapp.NodeCapabilitiesHandler(healthyStore, healthyPolicy, f.clock)
	check(t, err)
	healthyServer := serveCapabilities(t, healthyPolicy, healthyHandler)
	hints := []discovery.Hint{
		{Service: discovery.Node, Endpoint: strings.TrimPrefix(lockedServer.URL, "https://")},
		{Service: discovery.Node, Endpoint: strings.TrimPrefix(healthyServer.URL, "https://")},
	}
	started := time.Now()
	selection, err := recruitment.SelectLearners(ctx, hints, requirements, func(ctx context.Context, endpoint string) (recruitment.Observation, error) {
		return clusterapp.QueryNodeCapabilities(ctx, endpoint, client)
	}, f.clock)
	check(t, err)
	<-done
	check(t, lockErr)
	// Retrying a 2s server timeout would eventually succeed too; require a
	// single authenticated store read to prove that hidden cap is gone.
	if len(selection.Selected) != 2 || calls.Load() != 1 || time.Since(started) < 3*time.Second {
		t.Fatalf("selection=%+v store reads=%d elapsed=%v", selection, calls.Load(), time.Since(started))
	}
}

func TestNodeInventoryAuthenticatedProbeFailureClassification(t *testing.T) {
	for _, mode := range []string{"transient-503", "unauthorized", "forbidden", "malformed"} {
		t.Run(mode, func(t *testing.T) {
			f := newTLSFixture(t)
			store, policy, client, requirements := capabilityNode(t, f)
			h, err := clusterapp.NodeCapabilitiesHandler(store, policy, f.clock)
			check(t, err)
			var calls atomic.Int32
			server := serveCapabilities(t, policy, policy.AuthenticateHTTP(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				n := calls.Add(1)
				if r.Method != http.MethodGet || r.URL.Path != "/v1/agent/capabilities" || r.ContentLength != 0 {
					t.Error("probe replay changed its read-only request")
				}
				switch mode {
				case "transient-503":
					if n == 1 {
						http.Error(w, "temporarily unavailable", http.StatusServiceUnavailable)
						return
					}
					h.ServeHTTP(w, r)
				case "unauthorized":
					http.Error(w, "rejected", http.StatusUnauthorized)
				case "forbidden":
					http.Error(w, "rejected", http.StatusForbidden)
				case "malformed":
					_, _ = w.Write([]byte(`{"version":`))
				}
			})))
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			hints := []discovery.Hint{{Service: discovery.Node, Endpoint: strings.TrimPrefix(server.URL, "https://")}}
			rows, err := recruitment.Inventory(ctx, hints, requirements.Cluster, func(ctx context.Context, endpoint string) (recruitment.Observation, error) {
				return clusterapp.QueryNodeCapabilities(ctx, endpoint, client)
			}, f.clock)
			check(t, err)
			if len(rows) != 1 {
				t.Fatalf("inventory rows=%+v", rows)
			}
			if mode == "transient-503" {
				if calls.Load() != 2 || rows[0].Reason != "available" {
					t.Fatalf("transient read-only response not retried: calls=%d rows=%+v", calls.Load(), rows)
				}
			} else if calls.Load() != 1 || rows[0].Capabilities != nil || rows[0].Reason == "available" {
				t.Fatalf("permanent rejection retried or accepted: calls=%d rows=%+v", calls.Load(), rows)
			}
		})
	}
}
