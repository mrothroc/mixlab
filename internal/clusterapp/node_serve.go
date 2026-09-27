package clusterapp

import (
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"sync"
	"time"

	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/transport/managedtls"
)

type NodeServerOptions struct {
	Store       *nodeagent.Store
	Jobs        NodeJobOperations
	Execution   NodeExecutionPorts
	Policy      func() (*managedtls.Policy, error)
	Maintenance func(context.Context) error
	Ready       func()
	Snapshots   *NodeSnapshots
}

// ServeNode gates HTTP on the durable execution restart fence. Shutdown first
// stops/drains HTTP mutations, then cancels and joins worker/relay cleanup.
// Policy reloads the local identity at each handshake and request so renewal
// never leaves a long-lived server presenting the initial leaf certificate.
func ServeNode(ctx context.Context, listener net.Listener, o NodeServerOptions) (result error) {
	if listener == nil || o.Store == nil || o.Jobs == nil || o.Policy == nil || o.Maintenance == nil || o.Ready == nil || o.Execution.Clock == nil {
		return fmt.Errorf("node listener, jobs, execution, trust maintenance and readiness required")
	}
	defer func() { _ = listener.Close() }()
	policyForTLS := o.Policy
	if o.Snapshots != nil {
		policyForTLS = o.Snapshots.Policy
	}
	policy, err := policyForTLS()
	if err != nil {
		return err
	}
	// Parent cancellation is sequenced below, not delivered to the runtime
	// before in-flight HTTP handlers have stopped publishing new intent.
	life, cancel := context.WithCancel(context.WithoutCancel(ctx))
	defer cancel()
	requests, cancelRequests := context.WithCancel(life)
	defer cancelRequests()
	gate := &nodeRequestGate{clock: o.Execution.Clock}
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !gate.enter(w) {
			return
		}
		defer gate.leave()
		if o.Snapshots != nil && r.URL.Path == nodeSnapshotRoute {
			o.Snapshots.ServeHTTP(w, r)
			return
		}
		current, err := o.Policy()
		if err != nil {
			http.Error(w, "node identity unavailable", http.StatusServiceUnavailable)
			return
		}
		h, err := NodeManagementHandler(o.Store, current, o.Execution.Clock, o.Jobs)
		if err != nil {
			http.Error(w, "node management unavailable", http.StatusServiceUnavailable)
			return
		}
		h.ServeHTTP(w, r)
	})
	server := &http.Server{Handler: handler, ReadHeaderTimeout: 5 * time.Second, ReadTimeout: 10 * time.Second, WriteTimeout: 15 * time.Second, IdleTimeout: 30 * time.Second, MaxHeaderBytes: 8 << 10, BaseContext: func(net.Listener) context.Context { return requests }, ErrorLog: log.New(io.Discard, "", 0)}
	config := policy.ServerConfig()
	if o.Snapshots != nil {
		config = policy.SnapshotServerConfig()
	}
	config.GetConfigForClient = func(*tls.ClientHelloInfo) (*tls.Config, error) {
		current, err := policyForTLS()
		if err != nil {
			return nil, err
		}
		if o.Snapshots != nil {
			return current.SnapshotServerConfig(), nil
		}
		return current.ServerConfig(), nil
	}
	runtimeReady := make(chan struct{})
	o.Execution.Ready = func() { close(runtimeReady) }
	runtimeDone, maintenanceDone := make(chan struct{}), make(chan struct{})
	var runtimeErr, maintenanceErr error
	go func() { runtimeErr = RunNodeExecution(life, o.Store, o.Execution); close(runtimeDone) }()
	go func() { maintenanceErr = o.Maintenance(life); close(maintenanceDone) }()
	defer func() {
		gate.stop()
		drain, done := context.WithTimeout(context.Background(), 12*time.Second)
		defer done()
		if err := server.Shutdown(drain); err != nil {
			result = errors.Join(result, err)
			_ = server.Close()
		}
		cancelRequests()
		// Handlers are bounded by their request contexts and local ports. The
		// gate prevents a new Add racing this Wait after the listener closes.
		gate.active.Wait()
		cancel()
		<-runtimeDone
		<-maintenanceDone
		result = errors.Join(result, runtimeErr, maintenanceErr)
	}()
	select {
	case <-ctx.Done():
		return nil
	case <-runtimeDone:
		return fmt.Errorf("node execution stopped before serving")
	case <-maintenanceDone:
		return fmt.Errorf("node trust maintenance stopped before serving")
	case <-runtimeReady:
	}
	serveDone := make(chan struct{})
	var serveErr error
	go func() {
		serveErr = server.Serve(tls.NewListener(&limitListener{Listener: listener, permits: make(chan struct{}, 64)}, config))
		close(serveDone)
	}()
	defer func() {
		// Unblock Serve even if shutdown has not run yet (LIFO defers).
		_ = listener.Close()
		<-serveDone
		if serveErr != nil && !errors.Is(serveErr, net.ErrClosed) && !errors.Is(serveErr, http.ErrServerClosed) {
			result = errors.Join(result, serveErr)
		}
	}()
	o.Ready()
	select {
	case <-ctx.Done():
		return nil
	case <-runtimeDone:
		return fmt.Errorf("node execution owner stopped")
	case <-maintenanceDone:
		return fmt.Errorf("node trust maintenance stopped")
	case <-serveDone:
		return fmt.Errorf("node listener stopped")
	}
}

type nodeRequestGate struct {
	mu       sync.Mutex
	active   sync.WaitGroup
	stopping bool
	clock    func() time.Time
	window   time.Time
	requests int
}

func (g *nodeRequestGate) enter(w http.ResponseWriter) bool {
	g.mu.Lock()
	if g.stopping {
		g.mu.Unlock()
		http.Error(w, "node stopping", http.StatusServiceUnavailable)
		return false
	}
	now := g.clock()
	if g.window.IsZero() || now.Sub(g.window) >= time.Minute {
		g.window, g.requests = now, 0
	}
	if g.requests >= 1024 {
		g.mu.Unlock()
		http.Error(w, "node rate limit", http.StatusTooManyRequests)
		return false
	}
	g.requests++
	g.active.Add(1)
	g.mu.Unlock()
	return true
}

func (g *nodeRequestGate) leave() { g.active.Done() }
func (g *nodeRequestGate) stop()  { g.mu.Lock(); g.stopping = true; g.mu.Unlock() }
