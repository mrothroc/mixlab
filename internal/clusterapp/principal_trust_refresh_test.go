package clusterapp

import (
	"bytes"
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync/atomic"
	"syscall"
	"testing"
	"testing/synctest"
	"time"

	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/transport/managedtls"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/principal"
)

func TestStartupTrustTransientCacheFallback(t *testing.T) {
	for _, mode := range []string{"connection_refused", "attempt_timeout", "cache_expires_during_attempt", "invalid_cache_exhausted"} {
		t.Run(mode, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				start := time.Now()
				calls, cacheChecks := 0, 0
				stale := errors.New("cached trust expired")
				err := retryTrustRefreshWithCache(context.Background(), func(ctx context.Context, attempt int) error {
					calls++
					if attempt != calls-1 {
						t.Fatal("retry index reset")
					}
					deadline, ok := ctx.Deadline()
					want := min(50*time.Second, 180*time.Second-time.Since(start))
					if !ok || time.Until(deadline) != want {
						t.Fatalf("attempt budget=%v want=%v", time.Until(deadline), want)
					}
					if mode == "connection_refused" {
						return &net.OpError{Op: "dial", Net: "tcp", Err: syscall.ECONNREFUSED}
					}
					if mode == "cache_expires_during_attempt" && calls == 2 {
						return nil
					}
					<-ctx.Done()
					return ctx.Err()
				}, func() error {
					cacheChecks++
					if mode == "invalid_cache_exhausted" || (mode == "cache_expires_during_attempt" && time.Since(start) >= 30*time.Second) {
						return stale
					}
					return nil
				})
				wantCalls, wantChecks, wantTime := 1, 1, time.Duration(0)
				switch mode {
				case "attempt_timeout":
					wantTime = 50 * time.Second
				case "cache_expires_during_attempt":
					wantCalls, wantTime = 2, 50*time.Second+250*time.Millisecond
				case "invalid_cache_exhausted":
					wantCalls, wantChecks, wantTime = 4, 3, 180*time.Second
				}
				if mode == "invalid_cache_exhausted" {
					if !errors.Is(err, context.DeadlineExceeded) || !errors.Is(err, stale) {
						t.Fatalf("invalid cache accepted or failure lost: %v", err)
					}
				} else if err != nil {
					t.Fatal(err)
				}
				if calls != wantCalls || cacheChecks != wantChecks || time.Since(start) != wantTime {
					t.Fatalf("calls=%d checks=%d elapsed=%v want=%d/%d/%v", calls, cacheChecks, time.Since(start), wantCalls, wantChecks, wantTime)
				}
			})
		})
	}
}

func TestStartupTrustCancellationPreventsCacheFallback(t *testing.T) {
	for _, mode := range []string{"during_attempt", "during_cache_check", "during_backoff"} {
		t.Run(mode, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				if mode != "during_cache_check" {
					go func() { time.Sleep(100 * time.Millisecond); cancel() }()
				}
				calls, cacheChecks := 0, 0
				err := retryTrustRefreshWithCache(ctx, func(call context.Context, _ int) error {
					calls++
					if mode == "during_attempt" {
						<-call.Done()
						// Cancellation wins even if transport reports a transient error.
					}
					return io.EOF
				}, func() error {
					cacheChecks++
					if mode == "during_backoff" {
						return errors.New("cached trust expired")
					}
					cancel()
					return nil
				})
				wantChecks := 1
				if mode == "during_attempt" {
					wantChecks = 0
				}
				if !errors.Is(err, context.Canceled) || calls != 1 || cacheChecks != wantChecks {
					t.Fatalf("err=%v calls=%d cache checks=%d", err, calls, cacheChecks)
				}
			})
		})
	}
}

func TestTrustRefreshLoadBudgets(t *testing.T) {
	for _, mode := range []string{"slow_success", "timeout_then_success", "exhausted", "late_success"} {
		t.Run(mode, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				start := time.Now()
				calls := 0
				err := retryTrustRefresh(context.Background(), func(ctx context.Context, attempt int) error {
					calls++
					if attempt != calls-1 {
						t.Fatal("retry index reset")
					}
					deadline, ok := ctx.Deadline()
					want := min(50*time.Second, 180*time.Second-time.Since(start))
					if !ok || time.Until(deadline) != want {
						t.Fatalf("attempt %d remaining=%v want=%v", attempt, time.Until(deadline), want)
					}
					if mode == "slow_success" {
						time.Sleep(20 * time.Second)
						return nil
					}
					if mode == "exhausted" || calls == 1 {
						<-ctx.Done()
						if mode == "late_success" {
							return nil
						}
						return ctx.Err()
					}
					return nil
				})
				wantCalls, wantTime := 2, 50*time.Second+250*time.Millisecond
				if mode == "slow_success" {
					wantCalls, wantTime = 1, 20*time.Second
				}
				if mode == "exhausted" {
					wantCalls, wantTime = 4, 180*time.Second
					if !errors.Is(err, context.DeadlineExceeded) {
						t.Fatalf("refresh exhaustion accepted: %v", err)
					}
				} else if err != nil {
					t.Fatal(err)
				}
				if calls != wantCalls || time.Since(start) != wantTime {
					t.Fatalf("calls=%d elapsed=%v want=%d/%v", calls, time.Since(start), wantCalls, wantTime)
				}
			})
		})
	}
}

func TestTrustRefreshParentCancellation(t *testing.T) {
	for _, duringCall := range []bool{false, true} {
		t.Run(fmt.Sprint(duringCall), func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				go func() { time.Sleep(100 * time.Millisecond); cancel() }()
				start, calls := time.Now(), 0
				err := retryTrustRefresh(ctx, func(call context.Context, _ int) error {
					calls++
					if duringCall {
						<-call.Done()
						return call.Err()
					}
					return net.ErrClosed
				})
				if !errors.Is(err, context.Canceled) || calls != 1 || time.Since(start) != 100*time.Millisecond {
					t.Fatalf("err=%v calls=%d elapsed=%v", err, calls, time.Since(start))
				}
			})
		})
	}
}

func TestTrustRefreshPermanentFailure(t *testing.T) {
	for _, failure := range []error{
		&enrollmentHTTPError{Status: http.StatusForbidden},
		errors.New("invalid signed snapshot"),
		errors.Join(io.EOF, errors.New("invalid signed snapshot")),
		errors.Join(&enrollmentHTTPError{Status: 503}, managedtls.ErrPeerRejected),
		&tls.CertificateVerificationError{Err: &enrollmentHTTPError{Status: 503}},
	} {
		t.Run(failure.Error(), func(t *testing.T) {
			calls := 0
			err := retryTrustRefresh(context.Background(), func(context.Context, int) error { calls++; return failure })
			if !errors.Is(err, failure) || calls != 1 {
				t.Fatalf("permanent failure retried: err=%v calls=%d", err, calls)
			}
		})
	}
}

func TestTrustRefreshMalformedCompleteBodyIsPermanent(t *testing.T) {
	for _, body := range []string{"", `{"payload":`} {
		t.Run(fmt.Sprintf("length_%d", len(body)), func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { _, _ = io.WriteString(w, body) }))
			defer server.Close()
			var result any
			err := enrollmentJSON(context.Background(), server.Client(), http.MethodGet, server.URL, nil, &result)
			if err == nil || !strings.Contains(err.Error(), "invalid enrollment response") || transientControllerTrust(err) {
				t.Fatalf("malformed complete response classified transient: %v", err)
			}
		})
	}
}

func TestTrustRefreshPinnedHTTPSRetryAndRejection(t *testing.T) {
	for _, mode := range []string{"retry", "startup_cached", "startup_expired_cache", "forbidden", "malformed", "invalid_signature", "expired_response"} {
		t.Run(mode, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			server := httptest.NewUnstartedServer(nil)
			defer server.Close()
			endpoint := "https://" + server.Listener.Addr().String()
			path := initializedAt(t, endpoint)
			check(t, InitializeAuthority(ctx, path, testNow))
			a, err := OpenAuthority(ctx, path, testNow)
			check(t, err)
			defer func() { check(t, a.Close()) }()
			var timestamp atomic.Int64
			timestamp.Store(testNow.Unix())
			clock := func() time.Time { return time.Unix(timestamp.Load(), 0) }
			v, err := a.Current(ctx, clock())
			check(t, err)
			oldSnapshot, err := v.Bytes()
			check(t, err)
			policy, err := managedPrincipalPolicy(a.Principal, trust.Controller, clock)
			check(t, err)
			server.TLS = policy.SnapshotServerConfig()
			var calls atomic.Int32
			server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				n := calls.Add(1)
				if r.Method != http.MethodGet || r.URL.Path != "/v1/trust/snapshots/latest" {
					t.Error("trust retry changed the read-only request")
				}
				switch mode {
				case "startup_cached":
					http.Error(w, "busy", http.StatusServiceUnavailable)
					return
				case "retry", "startup_expired_cache":
					if n == 1 {
						timestamp.Add(int64((24 * time.Hour) / time.Second))
						http.Error(w, "busy", http.StatusServiceUnavailable)
						return
					}
				case "forbidden":
					http.Error(w, "denied", http.StatusForbidden)
					return
				case "malformed":
					_, _ = io.WriteString(w, `{"payload":`)
					return
				case "invalid_signature":
					_, _ = io.WriteString(w, `{}`)
					return
				case "expired_response":
					timestamp.Add(int64((24 * time.Hour) / time.Second))
					_, _ = w.Write(oldSnapshot)
					return
				}
				current, err := a.Current(r.Context(), clock())
				if err != nil {
					t.Error(err)
					http.Error(w, "unavailable", http.StatusServiceUnavailable)
					return
				}
				body, err := current.Bytes()
				if err != nil {
					t.Error(err)
					return
				}
				_, _ = w.Write(body)
			})
			server.StartTLS()
			cp, err := statehome.Resolve(statehome.Options{ExactDir: filepath.Join(filepath.Dir(path.Dir()), "controller")}, statehome.Context{Kind: statehome.Principal})
			check(t, err)
			p, err := principal.Open(cp, clock())
			check(t, err)
			defer func() { check(t, p.Close()) }()
			_, _, err = p.Active(clock())
			check(t, err)
			before, err := cp.ReadFile("principal.json")
			check(t, err)
			if mode == "retry" {
				err = RefreshRemoteTrust(ctx, p, endpoint, clock)
			} else {
				// Even an initially valid cache must not hide a permanent
				// authority rejection or an invalid/expired signed response.
				err = refreshStartupTrust(ctx, p, clock)
			}
			switch mode {
			case "retry", "startup_expired_cache":
				check(t, err)
				_, _, err = p.Active(clock())
				check(t, err)
				if calls.Load() != 2 {
					t.Fatalf("snapshot calls=%d, want 2", calls.Load())
				}
			case "startup_cached":
				check(t, err)
				_, _, err = p.Active(clock())
				check(t, err)
				if calls.Load() != 1 {
					t.Fatalf("valid cache startup calls=%d, want 1", calls.Load())
				}
				after, err := cp.ReadFile("principal.json")
				check(t, err)
				if !bytes.Equal(before, after) {
					t.Fatal("cached startup changed pinned principal state")
				}
			default:
				if err == nil || calls.Load() != 1 {
					t.Fatalf("permanent failure retried/accepted: err=%v calls=%d", err, calls.Load())
				}
				after, err := cp.ReadFile("principal.json")
				check(t, err)
				if !bytes.Equal(before, after) {
					t.Fatal("rejected snapshot changed pinned principal state")
				}
			}
		})
	}
}
