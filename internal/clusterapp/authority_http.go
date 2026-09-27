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

	"github.com/mrothroc/mixlab/transport/managedtls"
	"github.com/mrothroc/mixlab/trust"
)

// ServeAuthority exposes public signed snapshots and mutually authenticated
// same-principal renewal. No administrator decision/revocation route exists.
func ServeAuthority(ctx context.Context, l net.Listener, a *Authority, clock func() time.Time) error {
	return ServeAuthorityWithEvents(ctx, l, a, clock, nil)
}

func ServeAuthorityWithEvents(ctx context.Context, l net.Listener, a *Authority, clock func() time.Time, event func(error)) (result error) {
	if l == nil || a == nil || clock == nil {
		return fmt.Errorf("authority listener, application and clock required")
	}
	defer func() { _ = l.Close() }()
	life, cancel := context.WithCancel(ctx)
	defer cancel()
	scheduler, err := PrincipalScheduler(a.principalPath, a.Principal, func(ctx context.Context, at time.Time) error { return RenewLocalPrincipal(ctx, a, a.Principal, at) })
	if err != nil {
		return err
	}
	maintenance := make(chan error, 1)
	go func() { maintenance <- authorityMaintenance(life, a, scheduler, clock, event); cancel() }()
	defer func() { cancel(); result = errors.Join(result, <-maintenance) }()
	if _, err := a.Current(ctx, clock()); err != nil {
		return err
	}
	s, k, err := a.Principal.Active(clock())
	if err != nil {
		return err
	}
	policy, err := managedtls.New(managedtls.Identity{Chain: s.Chain, Key: k}, func(chain [][]byte, now time.Time) (trust.AuthenticatedPrincipal, error) {
		v, err := a.Current(life, now)
		if err != nil {
			return trust.AuthenticatedPrincipal{}, err
		}
		return trust.AuthenticatePrincipal(a.Anchor, v, chain, now)
	}, clock)
	if err != nil {
		return err
	}
	secure := policy.AuthenticateHTTP(authorityManagedHandler(a, clock))
	// A global bounded rate is sufficient here; per-node admission remains the
	// owning application's responsibility. Reject rather than queue excess work.
	var mu sync.Mutex
	window := clock()
	requests := 0
	h := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		now := clock()
		if now.Sub(window) >= time.Minute {
			window = now
			requests = 0
		}
		requests++
		allowed := requests <= 1024
		mu.Unlock()
		if !allowed {
			http.Error(w, "authority rate limit", http.StatusTooManyRequests)
			return
		}
		if r.Method == http.MethodGet && r.URL.Path == "/v1/trust/snapshots/latest" && r.URL.RawQuery == "" {
			v, err := a.Current(r.Context(), now)
			if err != nil {
				http.Error(w, "authority unavailable", http.StatusServiceUnavailable)
				return
			}
			b, err := v.Bytes()
			if err != nil {
				http.Error(w, "snapshot unavailable", http.StatusServiceUnavailable)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			w.Header().Set("Cache-Control", "no-store")
			_, _ = w.Write(b)
			return
		}
		secure.ServeHTTP(w, r)
	})
	server := &http.Server{Handler: h, ReadHeaderTimeout: 5 * time.Second, ReadTimeout: 10 * time.Second, WriteTimeout: 10 * time.Second, IdleTimeout: 30 * time.Second, MaxHeaderBytes: 8 << 10, BaseContext: func(net.Listener) context.Context { return life }, ErrorLog: log.New(io.Discard, "", 0)}
	stop := context.AfterFunc(life, func() { _ = server.Close() })
	defer stop()
	config := policy.SnapshotServerConfig()
	config.GetConfigForClient = func(*tls.ClientHelloInfo) (*tls.Config, error) {
		if _, err := a.Current(life, clock()); err != nil {
			return nil, err
		}
		s, k, err := a.Principal.Active(clock())
		if err != nil {
			return nil, err
		}
		fresh, err := managedtls.New(managedtls.Identity{Chain: s.Chain, Key: k}, func(chain [][]byte, now time.Time) (trust.AuthenticatedPrincipal, error) {
			v, err := a.Current(life, now)
			if err != nil {
				return trust.AuthenticatedPrincipal{}, err
			}
			return trust.AuthenticatePrincipal(a.Anchor, v, chain, now)
		}, clock)
		if err != nil {
			return nil, err
		}
		return fresh.SnapshotServerConfig(), nil
	}
	err = server.Serve(tls.NewListener(&limitListener{Listener: l, permits: make(chan struct{}, 64)}, config))
	if life.Err() != nil && (errors.Is(err, http.ErrServerClosed) || errors.Is(err, net.ErrClosed)) {
		return nil
	}
	return err
}
