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
	"time"

	"github.com/mrothroc/mixlab/transport/managedtls"
)

type DiLoCoServerOptions struct {
	Service     DiLoCoService
	Policy      func() (*managedtls.Policy, error)
	Handler     func(*managedtls.Policy) (http.Handler, error)
	Maintenance func(context.Context) error
	Ready       func()
}

// ServeDiLoCo owns both HTTP and the durable service loop. A request context
// cannot cancel a run. Shutdown drains admission requests before joining run
// effects; the next service instance resumes the same durable inbox.
func ServeDiLoCo(ctx context.Context, l net.Listener, o DiLoCoServerOptions) (result error) {
	if l == nil || o.Policy == nil || o.Handler == nil || o.Maintenance == nil || o.Ready == nil {
		return fmt.Errorf("coordinator listener, handlers, maintenance and readiness required")
	}
	defer func() { _ = l.Close() }()
	p, err := o.Policy()
	if err != nil {
		return err
	}
	life, cancel := context.WithCancel(context.WithoutCancel(ctx))
	defer cancel()
	requests, cancelRequests := context.WithCancel(life)
	defer cancelRequests()
	gate := &nodeRequestGate{clock: o.Service.Clock, limit: 32768}
	if gate.clock == nil {
		return fmt.Errorf("service clock required")
	}
	h := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !gate.enter(w) {
			return
		}
		defer gate.leave()
		p, e := o.Policy()
		if e != nil {
			http.Error(w, "coordinator identity unavailable", http.StatusServiceUnavailable)
			return
		}
		h, e := o.Handler(p)
		if e != nil {
			http.Error(w, "coordinator unavailable", http.StatusServiceUnavailable)
			return
		}
		h.ServeHTTP(w, r)
	})
	server := &http.Server{Handler: h, ReadHeaderTimeout: 5 * time.Second, ReadTimeout: 30 * time.Second, WriteTimeout: 35 * time.Second, IdleTimeout: 30 * time.Second, MaxHeaderBytes: 8 << 10, BaseContext: func(net.Listener) context.Context { return requests }, ErrorLog: log.New(io.Discard, "", 0)}
	tlsConfig := p.ServerConfig()
	tlsConfig.GetConfigForClient = func(*tls.ClientHelloInfo) (*tls.Config, error) {
		p, e := o.Policy()
		if e != nil {
			return nil, e
		}
		return p.ServerConfig(), nil
	}
	ready := make(chan struct{})
	done, maintenance := make(chan error, 1), make(chan error, 1)
	go func() { done <- o.Service.Serve(life, func() { close(ready) }) }()
	go func() { maintenance <- o.Maintenance(life) }()
	var runErr, maintenanceErr error
	runStopped, maintenanceStopped := false, false
	defer func() {
		gate.stop()
		drain, stop := context.WithTimeout(context.Background(), 5*time.Second)
		if err := server.Shutdown(drain); err != nil {
			_ = server.Close()
		}
		stop()
		cancelRequests()
		gate.active.Wait()
		cancel()
		if !runStopped {
			runErr = <-done
		}
		if !maintenanceStopped {
			maintenanceErr = <-maintenance
		}
		for _, err := range []error{runErr, maintenanceErr} {
			if err != nil && !errors.Is(err, context.Canceled) {
				result = errors.Join(result, err)
			}
		}
	}()
	select {
	case <-ctx.Done():
		return nil
	case runErr = <-done:
		runStopped = true
		return fmt.Errorf("coordinator lifecycle stopped before ready")
	case maintenanceErr = <-maintenance:
		maintenanceStopped = true
		return fmt.Errorf("coordinator trust maintenance stopped before ready")
	case <-ready:
	}
	served := make(chan error, 1)
	var serveErr error
	serveStopped := false
	go func() {
		served <- server.Serve(tls.NewListener(&limitListener{Listener: l, permits: make(chan struct{}, 64)}, tlsConfig))
	}()
	defer func() {
		_ = l.Close()
		if !serveStopped {
			serveErr = <-served
		}
		if serveErr != nil && !errors.Is(serveErr, net.ErrClosed) && !errors.Is(serveErr, http.ErrServerClosed) {
			result = errors.Join(result, serveErr)
		}
	}()
	o.Ready()
	select {
	case <-ctx.Done():
		return nil
	case serveErr = <-served:
		serveStopped = true
		return fmt.Errorf("coordinator HTTP listener stopped")
	case runErr = <-done:
		runStopped = true
		return fmt.Errorf("coordinator lifecycle stopped")
	case maintenanceErr = <-maintenance:
		maintenanceStopped = true
		return fmt.Errorf("coordinator trust maintenance stopped")
	}
}
