package clusterapp

import (
	"bufio"
	"context"
	"crypto/tls"
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
					s.Handler.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/", strings.NewReader("body")).WithContext(ctx))
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
			s.Handler.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/", strings.NewReader("body")))
			if called != tc.allowed {
				t.Fatalf("handler called = %v, want %v", called, tc.allowed)
			}
			if !tc.allowed && w.(*managedReadDeadlineRecorder).Code != http.StatusServiceUnavailable {
				t.Fatal("deadline failure did not fail closed")
			}
		})
	}
}

func TestManagedHTTPServerBodyDeadlineScope(t *testing.T) {
	const bodyBudget = 100 * time.Millisecond
	for _, tc := range []struct {
		name, request     string
		hasBody           bool
		disableKeepAlives bool
	}{
		{name: "bodyless_lease_get_keepalive"},
		{name: "bodyless_lease_get_no_keepalive", disableKeepAlives: true},
		{name: "stalled_post", request: "POST / HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n\r\nx", hasBody: true},
		{name: "stalled_get", request: "GET / HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n\r\nx", hasBody: true},
		{name: "stalled_chunked_get", request: "GET / HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: chunked\r\n\r\n2\r\nx", hasBody: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			type result struct {
				err     error
				elapsed time.Duration
			}
			observed := make(chan result, 1)
			server := httptest.NewUnstartedServer(nil)
			server.EnableHTTP2 = false
			server.Config = newManagedHTTPServer(context.Background(), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				started := time.Now()
				if r.ProtoMajor != 1 || r.TLS == nil || !r.TLS.HandshakeComplete {
					t.Error("regression requires HTTP/1 over completed TLS")
				}
				var err error
				status := http.StatusNoContent
				if tc.hasBody {
					_, err = io.ReadAll(io.LimitReader(r.Body, 1024))
					status = http.StatusRequestTimeout
				} else {
					if r.Method != http.MethodGet || r.ContentLength != 0 || len(r.TransferEncoding) != 0 || r.Close != tc.disableKeepAlives {
						t.Errorf("unexpected lease GET framing: method=%s length=%d encoding=%v close=%v", r.Method, r.ContentLength, r.TransferEncoding, r.Close)
					}
					// The old unconditional deadline cancels this request at
					// 100ms via net/http's background disconnect read.
					timer := time.NewTimer(3 * bodyBudget)
					defer timer.Stop()
					select {
					case <-r.Context().Done():
					case <-timer.C:
					}
					err = r.Context().Err()
					if err != nil {
						status = http.StatusServiceUnavailable
					}
				}
				observed <- result{err, time.Since(started)}
				w.WriteHeader(status)
			}), bodyBudget)
			server.StartTLS()
			defer server.Close()
			var response *http.Response
			if tc.hasBody {
				// Retain the real TLS connection while withholding the
				// remainder of each declared HTTP/1 body.
				config := server.Client().Transport.(*http.Transport).TLSClientConfig.Clone()
				conn, err := tls.DialWithDialer(&net.Dialer{Timeout: 5 * time.Second}, "tcp", server.Listener.Addr().String(), config)
				check(t, err)
				defer func() { _ = conn.Close() }()
				check(t, conn.SetDeadline(time.Now().Add(5*time.Second)))
				_, err = io.WriteString(conn, tc.request)
				check(t, err)
				response, err = http.ReadResponse(bufio.NewReader(conn), nil)
				check(t, err)
			} else {
				transport := server.Client().Transport.(*http.Transport).Clone()
				transport.DisableKeepAlives = tc.disableKeepAlives
				transport.ForceAttemptHTTP2 = false
				transport.TLSNextProto = map[string]func(string, *tls.Conn) http.RoundTripper{}
				defer transport.CloseIdleConnections()
				client := &http.Client{Transport: transport, Timeout: 5 * time.Second}
				// Same bodyless request/connection policy as managedtls,
				// representing a lease GET whose operation may outlive 10s.
				var err error
				response, err = client.Get(server.URL + "/v1/agent/leases/" + strings.Repeat("a", 32))
				check(t, err)
			}
			defer func() { _ = response.Body.Close() }()
			got := <-observed
			if tc.hasBody {
				var timeout net.Error
				if response.StatusCode != http.StatusRequestTimeout || !errors.As(got.err, &timeout) || !timeout.Timeout() {
					t.Fatalf("body read was not bounded: status=%d result=%+v", response.StatusCode, got)
				}
			} else if response.StatusCode != http.StatusNoContent || got.err != nil || got.elapsed < 3*bodyBudget {
				t.Fatalf("body deadline canceled bodyless operation: status=%d result=%+v", response.StatusCode, got)
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
