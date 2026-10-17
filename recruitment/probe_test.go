package recruitment

import (
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"net"
	"reflect"
	"strings"
	"sync"
	"syscall"
	"testing"
	"testing/synctest"
	"time"

	"github.com/mrothroc/mixlab/discovery"
	"github.com/mrothroc/mixlab/internal/clusterdiagnostic"
	"github.com/mrothroc/mixlab/transport/managedtls"
)

func TestProbeAttemptDeadlineThenRetry(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		_, observation := fixture()
		start := time.Now()
		attempts := 0
		results := probeEndpoints(context.Background(), []string{"node.local:1"}, func(ctx context.Context, _ string) (Observation, error) {
			attempts++
			deadline, ok := ctx.Deadline()
			want := start.Add(20 * time.Second)
			if attempts == 2 {
				want = start.Add(25 * time.Second)
			}
			if !ok || !deadline.Equal(want) {
				t.Errorf("attempt %d deadline = %v, want %v", attempts, deadline, want)
			}
			if attempts == 1 {
				<-ctx.Done()
				return Observation{}, ctx.Err()
			}
			return observation, nil
		}, time.Now)
		if attempts != 2 || results[0].err != nil || time.Since(start) != 20*time.Second+250*time.Millisecond || !results[0].at.Equal(time.Now()) {
			t.Fatalf("attempts=%d result=%+v elapsed=%v", attempts, results[0], time.Since(start))
		}
	})
}

func TestProbeBackoffAndCancellation(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		_, observation := fixture()
		start := time.Now()
		want := []time.Duration{0, 250 * time.Millisecond, 750 * time.Millisecond, 1750 * time.Millisecond, 2750 * time.Millisecond}
		calls := 0
		results := probeEndpoints(context.Background(), []string{"node.local:1"}, func(context.Context, string) (Observation, error) {
			if calls >= len(want) || time.Since(start) != want[calls] {
				t.Errorf("attempt %d at %v", calls, time.Since(start))
			}
			calls++
			if calls < len(want) {
				return Observation{}, net.ErrClosed
			}
			return observation, nil
		}, time.Now)
		if calls != len(want) || results[0].err != nil {
			t.Fatalf("calls=%d error=%v", calls, results[0].err)
		}
	})
	for _, inCall := range []bool{false, true} {
		t.Run(fmt.Sprintf("cancel_in_call_%v", inCall), func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				start := time.Now()
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				go func() { time.Sleep(100 * time.Millisecond); cancel() }()
				calls := 0
				probeEndpoints(ctx, []string{"node.local:1"}, func(call context.Context, _ string) (Observation, error) {
					calls++
					if inCall {
						<-call.Done()
						return Observation{}, call.Err()
					}
					return Observation{}, io.EOF
				}, time.Now)
				if calls != 1 || time.Since(start) != 100*time.Millisecond {
					t.Fatalf("calls=%d elapsed=%v", calls, time.Since(start))
				}
			})
		})
	}
}

func TestProbeFirstPassPrecedesRetriesWithBoundedConcurrency(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		_, observation := fixture()
		endpoints := make([]string, 10)
		for i := range endpoints {
			endpoints[i] = fmt.Sprintf("node%d.local:1", i)
		}
		var mu sync.Mutex
		calls := map[string]int{}
		active, peak := 0, 0
		start := time.Now()
		results := probeEndpoints(context.Background(), endpoints, func(ctx context.Context, endpoint string) (Observation, error) {
			mu.Lock()
			calls[endpoint]++
			active++
			peak = max(peak, active)
			if calls[endpoint] > 1 && (calls[endpoints[8]] == 0 || calls[endpoints[9]] == 0) {
				t.Error("retry starved an untouched endpoint")
			}
			mu.Unlock()
			defer func() { mu.Lock(); active--; mu.Unlock() }()
			if endpoint == endpoints[8] || endpoint == endpoints[9] {
				return observation, nil
			}
			<-ctx.Done()
			return Observation{}, ctx.Err()
		}, time.Now)
		if peak != 8 || active != 0 || time.Since(start) != 25*time.Second || results[8].err != nil || results[9].err != nil || calls[endpoints[8]] != 1 || calls[endpoints[9]] != 1 {
			t.Fatalf("peak=%d active=%d elapsed=%v calls=%v", peak, active, time.Since(start), calls)
		}
	})
}

func TestSelectionAndInventoryKeepHealthyNodesWithUnavailableExtra(t *testing.T) {
	for _, mode := range []string{"selection", "inventory", "stale", "parent-canceled"} {
		t.Run(mode, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				r, first := fixture()
				_, second := fixture()
				second.Peer.Principal, second.Capabilities.Node = strings.Repeat("2", 32), strings.Repeat("2", 32)
				if mode == "stale" {
					first.Capabilities.ObservedAt -= 10
					first.Capabilities.ProbeObservedAt -= 10
					if reason := reject(first, r, selectionNow); reason != "" {
						t.Fatalf("stale fixture must start eligible: %s", reason)
					}
				}
				hints := []discovery.Hint{{Service: discovery.Node, Endpoint: "z.local:1"}, {Service: discovery.Node, Endpoint: "a-offline.local:1"}, {Service: discovery.Node, Endpoint: "b.local:1"}}
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				start := time.Now()
				if mode == "parent-canceled" {
					go func() { time.Sleep(time.Second); cancel() }()
				}
				lookup := func(call context.Context, endpoint string) (Observation, error) {
					switch endpoint {
					case "a-offline.local:1":
						<-call.Done()
						return Observation{}, call.Err()
					case "z.local:1":
						return first, nil
					default:
						return second, nil
					}
				}
				clock := func() time.Time { return selectionNow.Add(time.Since(start)) }
				if mode == "inventory" {
					rows, err := Inventory(ctx, hints, r.Cluster, lookup, clock)
					if err != nil || len(rows) != 3 || rows[0].Reason != "timeout" || rows[1].Reason != "available" || rows[2].Reason != "available" || time.Since(start) != 25*time.Second {
						t.Fatalf("inventory=%+v err=%v elapsed=%v", rows, err, time.Since(start))
					}
					return
				}
				out, err := Select(ctx, hints, r, lookup, clock)
				if mode == "parent-canceled" {
					if !errors.Is(err, context.Canceled) || len(out.Selected) != 0 || time.Since(start) != time.Second {
						t.Fatalf("cancellation selected nodes: %+v error=%v", out, err)
					}
					return
				}
				if time.Since(start) != 25*time.Second {
					t.Fatalf("scan consumed fold headroom: %v", time.Since(start))
				}
				if mode == "stale" {
					if err == nil || len(out.Selected) != 0 || out.Decisions[2].Reason != "stale_capabilities" {
						t.Fatalf("commit freshness relaxed: %+v error=%v", out, err)
					}
					return
				}
				if err != nil || len(out.Selected) != 2 || out.Selected[0].Endpoint != "z.local:1" || out.Selected[1].Endpoint != "b.local:1" || out.Decisions[0].Reason != "unavailable_or_unauthenticated" {
					t.Fatalf("unrelated outage rejected healthy cohort: %+v error=%v", out, err)
				}
			})
		})
	}
}

type probeTimeoutWrapper struct{ error }

func (e probeTimeoutWrapper) Unwrap() error { return e.error }
func (probeTimeoutWrapper) Timeout() bool   { return true }
func (probeTimeoutWrapper) Temporary() bool { return true }

func TestProbeTransientClassificationAndPermanentNoRetry(t *testing.T) {
	for _, err := range []error{context.DeadlineExceeded, io.EOF, io.ErrUnexpectedEOF, net.ErrClosed, syscall.ECONNREFUSED, ErrProbeUnavailable,
		&net.DNSError{IsTimeout: true}, probeTimeoutWrapper{errors.Join(io.EOF, context.DeadlineExceeded)}} {
		if !transientProbe(fmt.Errorf("lookup: %w", err)) {
			t.Errorf("transient failure rejected: %v", err)
		}
	}
	for _, err := range []error{context.Canceled, tls.AlertError(42), &tls.CertificateVerificationError{Err: io.EOF},
		errors.Join(managedtls.ErrPeerRejected, context.DeadlineExceeded), errors.Join(clusterdiagnostic.ErrLocalIdentity, io.EOF),
		probeTimeoutWrapper{errors.Join(context.DeadlineExceeded, errors.New("authorization denied"))}, errors.New("malformed capabilities")} {
		t.Run(fmt.Sprintf("%T_%v", err, err), func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				start := time.Now()
				calls := 0
				results := probeEndpoints(context.Background(), []string{"node.local:1"}, func(context.Context, string) (Observation, error) {
					calls++
					return Observation{}, err
				}, time.Now)
				if calls != 1 || !reflect.DeepEqual(results[0].err, err) || time.Now() != start {
					t.Fatalf("permanent failure retried: calls=%d error=%v elapsed=%v", calls, results[0].err, time.Since(start))
				}
			})
		})
	}
}
