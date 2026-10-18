package clusterapp

import (
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"path/filepath"
	"testing"
	"testing/synctest"
	"time"

	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/transport/managedtls"
	"github.com/mrothroc/mixlab/transport/snapshottls"
	"github.com/mrothroc/mixlab/trust/principal"
)

func TestControllerStartupTrustTimeoutThenSuccess(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		start := time.Now()
		parent := context.Background()
		attempts, validations := 0, 0
		view := 1
		err := refreshControllerStartup(parent, func(ctx context.Context) error {
			attempts++
			deadline, ok := ctx.Deadline()
			if !ok || time.Until(deadline) != 50*time.Second {
				t.Fatalf("attempt deadline = %v, ok = %v", deadline, ok)
			}
			if attempts == 1 {
				<-ctx.Done()
				return ctx.Err()
			}
			if elapsed := time.Since(start); elapsed != 50*time.Second+250*time.Millisecond {
				t.Fatalf("retry time = %v", elapsed)
			}
			view = 2
			return nil
		}, func() error {
			validations++
			if view != 2 || time.Since(start) <= 50*time.Second {
				t.Fatal("validation did not use post-retry view/time")
			}
			return nil
		}, func(err error) { t.Fatalf("unexpected fallback: %v", err) })
		if err != nil || attempts != 2 || validations != 1 || parent.Err() != nil {
			t.Fatalf("err = %v, attempts = %d, validations = %d", err, attempts, validations)
		}
	})
}

func TestControllerStartupTrustBudgetAndCachedFallback(t *testing.T) {
	for _, expires := range []bool{false, true} {
		t.Run(fmt.Sprintf("cache_expires_%v", expires), func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				start := time.Now()
				stale := errors.New("cached trust expired")
				attempts, validations, warnings := 0, 0, 0
				err := refreshControllerStartup(context.Background(), func(ctx context.Context) error {
					attempts++
					deadline, _ := ctx.Deadline()
					if deadline.After(start.Add(2 * time.Minute)) {
						t.Fatal("attempt extended original budget")
					}
					<-ctx.Done()
					return ctx.Err()
				}, func() error {
					validations++
					if time.Since(start) != 2*time.Minute {
						t.Fatal("cached trust validated before retry budget elapsed")
					}
					if expires {
						return stale
					}
					return nil
				}, func(err error) {
					warnings++
					if !errors.Is(err, context.DeadlineExceeded) {
						t.Fatalf("fallback warning = %v", err)
					}
				})
				if attempts != 3 || validations != 1 || time.Since(start) != 2*time.Minute {
					t.Fatalf("attempts = %d, validations = %d, elapsed = %v", attempts, validations, time.Since(start))
				}
				if expires {
					if !errors.Is(err, stale) || !errors.Is(err, context.DeadlineExceeded) || warnings != 0 {
						t.Fatalf("expired cache err = %v, warnings = %d", err, warnings)
					}
				} else if err != nil || warnings != 1 {
					t.Fatalf("valid cached fallback err = %v, warnings = %d", err, warnings)
				}
			})
		})
	}
}

func TestControllerStartupTrustSlowFetch(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		start := time.Now()
		attempts := 0
		err := refreshControllerStartup(context.Background(), func(ctx context.Context) error {
			attempts++
			time.Sleep(20 * time.Second)
			return ctx.Err()
		}, func() error { return nil }, func(err error) { t.Fatalf("slow fetch fell back: %v", err) })
		if err != nil || attempts != 1 || time.Since(start) != 20*time.Second {
			t.Fatalf("err = %v, attempts = %d, elapsed = %v", err, attempts, time.Since(start))
		}
	})
}

func TestControllerStartupTrustTimeoutPolicy(t *testing.T) {
	// Ordinary refresh and startup share the load-tolerant attempt budget.
	for _, tc := range []struct {
		name    string
		timeout time.Duration
		want    time.Duration
	}{
		{"ordinary", remoteTrustRefreshTimeout, 50 * time.Second},
		{"startup", controllerStartupTrustAttempt, 50 * time.Second},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.timeout != tc.want {
				t.Fatalf("timeout = %v, want %v", tc.timeout, tc.want)
			}
			client, err := snapshottls.HTTPClient("https://authority.example:7443", func([][]byte, time.Time) error { return nil }, time.Now, tc.timeout)
			check(t, err)
			defer client.CloseIdleConnections()
			if client.Timeout != tc.want {
				t.Fatalf("HTTP timeout = %v, want %v", client.Timeout, tc.want)
			}
		})
	}
}

func TestControllerStartupTrustBackoff(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		start := time.Now()
		want := []time.Duration{0, 250 * time.Millisecond, 750 * time.Millisecond, 1750 * time.Millisecond, 3750 * time.Millisecond, 5750 * time.Millisecond}
		attempts := 0
		err := refreshControllerStartup(context.Background(), func(context.Context) error {
			if attempts >= len(want) || time.Since(start) != want[attempts] {
				t.Fatalf("attempt %d at %v", attempts, time.Since(start))
			}
			attempts++
			if attempts == len(want) {
				return nil
			}
			return &enrollmentHTTPError{Status: http.StatusServiceUnavailable}
		}, func() error { return nil }, nil)
		if err != nil || attempts != len(want) {
			t.Fatalf("err = %v, attempts = %d", err, attempts)
		}
	})
}

func TestControllerStartupTrustCancellation(t *testing.T) {
	for _, duringCall := range []bool{false, true} {
		t.Run(fmt.Sprintf("during_call_%v", duringCall), func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
				defer cancel()
				start := time.Now()
				attempts := 0
				err := refreshControllerStartup(ctx, func(call context.Context) error {
					attempts++
					if duringCall {
						<-call.Done()
						return call.Err()
					}
					return net.ErrClosed
				}, func() error {
					t.Fatal("cancelled command used cached trust")
					return nil
				}, func(err error) { t.Fatalf("cancelled command fell back: %v", err) })
				if !errors.Is(err, context.DeadlineExceeded) || attempts != 1 || time.Since(start) != 100*time.Millisecond {
					t.Fatalf("err = %v, attempts = %d, elapsed = %v", err, attempts, time.Since(start))
				}
			})
		})
	}
}

func TestControllerStartupTrustPermanentFailure(t *testing.T) {
	for _, failure := range []error{
		&enrollmentHTTPError{Status: http.StatusUnauthorized},
		&enrollmentHTTPError{Status: http.StatusForbidden},
		errors.New("invalid signed snapshot"),
		fmt.Errorf("authority rejected: %w", errors.Join(managedtls.ErrPeerRejected, context.DeadlineExceeded)),
	} {
		t.Run(failure.Error(), func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				start := time.Now()
				attempts := 0
				err := refreshControllerStartup(context.Background(), func(context.Context) error {
					attempts++
					return failure
				}, func() error {
					t.Fatal("permanent failure used cached trust")
					return nil
				}, nil)
				if !errors.Is(err, failure) || attempts != 1 || time.Now() != start {
					t.Fatalf("err = %v, attempts = %d", err, attempts)
				}
			})
		})
	}
}

func TestControllerStartupTrustRefreshedRevocation(t *testing.T) {
	revoked := errors.New("controller revoked")
	err := refreshControllerStartup(context.Background(), func(context.Context) error { return nil }, func() error { return revoked },
		func(err error) { t.Fatalf("revocation fell back: %v", err) })
	if !errors.Is(err, revoked) {
		t.Fatalf("err = %v", err)
	}
}

func TestControllerStartupTrustClassification(t *testing.T) {
	for _, err := range []error{net.ErrClosed, io.EOF, io.ErrUnexpectedEOF, context.DeadlineExceeded,
		&enrollmentHTTPError{Status: 408}, &enrollmentHTTPError{Status: 429}, &enrollmentHTTPError{Status: 500},
		&enrollmentHTTPError{Status: 502}, &enrollmentHTTPError{Status: 503}, &enrollmentHTTPError{Status: 504}} {
		if !transientControllerTrust(fmt.Errorf("fetch: %w", err)) {
			t.Errorf("not transient: %v", err)
		}
	}
	for _, err := range []error{context.Canceled, tls.AlertError(42), &tls.CertificateVerificationError{Err: io.EOF},
		&enrollmentHTTPError{Status: 400}, &enrollmentHTTPError{Status: 401}, &enrollmentHTTPError{Status: 403},
		&enrollmentHTTPError{Status: 404}, errors.New("signature mismatch")} {
		if transientControllerTrust(fmt.Errorf("fetch: %w", err)) {
			t.Errorf("permanent error classified transient: %v", err)
		}
	}
}

func TestControllerStartupTrustUnpinnedEndpointRejectsValidCache(t *testing.T) {
	p := initializedAt(t, "https://authority.example:7443")
	cp, err := statehome.Resolve(statehome.Options{ExactDir: filepath.Join(filepath.Dir(p.Dir()), "controller")}, statehome.Context{Kind: statehome.Principal})
	check(t, err)
	c, err := principal.Open(cp, testNow)
	check(t, err)
	defer func() { check(t, c.Close()) }()
	_, _, err = c.Active(testNow)
	check(t, err)
	for _, timeout := range []time.Duration{0, -time.Second, time.Minute + time.Nanosecond} {
		if err := refreshRemoteTrustWithTimeout(context.Background(), c, "https://authority.example:7443", func() time.Time { return testNow }, timeout); err == nil {
			t.Fatalf("invalid timeout accepted: %v", timeout)
		}
	}
	if err := RefreshControllerStartup(context.Background(), c, "https://other.example:7443", func() time.Time { return testNow }, nil); err == nil {
		t.Fatal("unpinned endpoint accepted via valid cache")
	}
}
