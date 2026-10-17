package clusterapp

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"testing/synctest"
	"time"
)

type managedReadDeadlineRecorder struct {
	*httptest.ResponseRecorder
	deadline time.Time
	err      error
}

func (w *managedReadDeadlineRecorder) SetReadDeadline(deadline time.Time) error {
	w.deadline = deadline
	return w.err
}

func TestManagedHTTPServerBudgets(t *testing.T) {
	for _, tc := range []struct {
		owner string
		body  time.Duration
	}{
		{"node", 10 * time.Second},
		{"coordinator", 30 * time.Second},
		{"authority", 55 * time.Second},
	} {
		t.Run(tc.owner, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				calls := 0
				s := newManagedHTTPServer(ctx, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					calls++
					if r.Context() != ctx {
						t.Error("server changed the owner's request context")
					}
					if got := w.(*managedReadDeadlineRecorder).deadline; !got.Equal(time.Now().Add(tc.body)) {
						t.Errorf("body deadline = %v, want %v", got, time.Now().Add(tc.body))
					}
					w.WriteHeader(http.StatusNoContent)
				}), tc.body)
				if s.ReadHeaderTimeout != 55*time.Second || s.ReadTimeout != 55*time.Second || s.WriteTimeout != 55*time.Second {
					t.Fatal("server shortened the bounded TLS handshake envelope")
				}
				if s.IdleTimeout != 30*time.Second || s.MaxHeaderBytes != 8<<10 || s.ErrorLog == nil || s.ErrorLog.Writer() != io.Discard || s.BaseContext(nil) != ctx {
					t.Fatal("server changed idle/header/logging/base-context policy")
				}
				for range 2 {
					w := &managedReadDeadlineRecorder{ResponseRecorder: httptest.NewRecorder()}
					s.Handler.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/", nil).WithContext(ctx))
					if w.Code != http.StatusNoContent {
						t.Fatalf("handler status = %d", w.Code)
					}
					time.Sleep(time.Second)
				}
				if calls != 2 {
					t.Fatalf("handler calls = %d", calls)
				}
				cancel()
				if !errors.Is(s.BaseContext(nil).Err(), context.Canceled) {
					t.Fatal("server lost owner cancellation")
				}
			})
		})
	}
}

func TestManagedHTTPServerReadDeadlineFailure(t *testing.T) {
	for _, tc := range []struct {
		name    string
		writer  func() http.ResponseWriter
		allowed bool
	}{
		{"recorder", func() http.ResponseWriter { return httptest.NewRecorder() }, true},
		{"wrapped_not_supported", func() http.ResponseWriter {
			return &managedReadDeadlineRecorder{ResponseRecorder: httptest.NewRecorder(), err: fmt.Errorf("recorder: %w", http.ErrNotSupported)}
		}, true},
		{"closed_connection", func() http.ResponseWriter {
			return &managedReadDeadlineRecorder{ResponseRecorder: httptest.NewRecorder(), err: net.ErrClosed}
		}, false},
		{"other_failure", func() http.ResponseWriter {
			return &managedReadDeadlineRecorder{ResponseRecorder: httptest.NewRecorder(), err: errors.New("not supported")}
		}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			called := false
			s := newManagedHTTPServer(context.Background(), http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				called = true
				w.WriteHeader(http.StatusNoContent)
			}), 10*time.Second)
			w := tc.writer()
			s.Handler.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/", nil))
			if called != tc.allowed {
				t.Fatalf("handler called = %v, want %v", called, tc.allowed)
			}
			if !tc.allowed && w.(*managedReadDeadlineRecorder).Code != http.StatusServiceUnavailable {
				t.Fatal("deadline failure did not fail closed")
			}
		})
	}
}

func TestManagedHTTPServerBodyBudgetStartsAfterTLS(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	const body = "body after slow TLS"
	server := httptest.NewUnstartedServer(nil)
	slow := &delayedAuthorityListener{Listener: server.Listener, delay: 12 * time.Second, started: make(chan struct{})}
	server.Listener = slow
	server.Config = newManagedHTTPServer(ctx, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.TLS == nil || !r.TLS.HandshakeComplete {
			t.Error("body handler ran before TLS completed")
			http.Error(w, "TLS required", http.StatusBadRequest)
			return
		}
		b, err := io.ReadAll(io.LimitReader(r.Body, 1024))
		if err != nil || string(b) != body {
			t.Errorf("post-handshake body failed: body=%q err=%v", b, err)
			http.Error(w, "body unavailable", http.StatusBadRequest)
			return
		}
		w.WriteHeader(http.StatusNoContent)
	}), 10*time.Second)
	server.StartTLS()
	defer server.Close()
	// The client trusts httptest's certificate. A real 12-second handshake
	// must not consume the shorter 10-second post-TLS body-read budget.
	client := server.Client()
	client.Timeout = 25 * time.Second
	started := time.Now()
	response, err := client.Post(server.URL, "text/plain", strings.NewReader(body))
	check(t, err)
	defer func() { _ = response.Body.Close() }()
	if response.StatusCode != http.StatusNoContent || slow.accepted.Load() != 1 || time.Since(started) < slow.delay {
		t.Fatalf("slow TLS/body result: status=%d connections=%d elapsed=%v", response.StatusCode, slow.accepted.Load(), time.Since(started))
	}
}
