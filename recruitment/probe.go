package recruitment

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"io"
	"net"
	"sync"
	"syscall"
	"time"

	"github.com/mrothroc/mixlab/internal/clusterdiagnostic"
	"github.com/mrothroc/mixlab/transport/managedtls"
)

// ProbeAttemptTimeout bounds a capability probe and its server-side store read.
// Parent deadlines may shorten either operation.
const ProbeAttemptTimeout = 20 * time.Second

const (
	probeBudget      = 30 * time.Second
	probePhaseBudget = 25 * time.Second
	probeConcurrency = 8
)

// ErrProbeUnavailable marks a transient, authenticated read-only capability
// response. It must never wrap identity, authorization, or decoding failures.
var ErrProbeUnavailable = errors.New("node capability probe unavailable")

type probeResult struct {
	observation Observation
	at          time.Time
	err         error
}

// Probe in bounded rounds: untouched endpoints precede retries, so one slow
// peer cannot consume every worker while healthy endpoints remain queued.
// Each result has one writer; folding and selection remain in endpoint order.
func probeEndpoints(ctx context.Context, endpoints []string, lookup Lookup, clock func() time.Time) []probeResult {
	// Leave five seconds for deterministic folding and final validation. A
	// healthy t=0 response must not age out solely because another peer retries.
	ctx, cancel := context.WithTimeout(ctx, probePhaseBudget)
	defer cancel()
	results := make([]probeResult, len(endpoints))
	pending := make([]int, len(endpoints))
	for i := range pending {
		pending[i] = i
		// Endpoints still queued when the scan budget expires are unavailable,
		// never successful zero-valued observations.
		results[i].err = context.DeadlineExceeded
	}
	delay := 250 * time.Millisecond
	var clockMu sync.Mutex
	for len(pending) != 0 && ctx.Err() == nil {
		jobs := make(chan int, len(pending))
		for _, i := range pending {
			jobs <- i
		}
		close(jobs)
		var wg sync.WaitGroup
		for range min(probeConcurrency, len(pending)) {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for i := range jobs {
					if err := ctx.Err(); err != nil {
						results[i].err = err
						continue
					}
					call, cancel := context.WithTimeout(ctx, ProbeAttemptTimeout)
					o, err := lookup(call, endpoints[i])
					if err == nil {
						err = call.Err()
					}
					cancel()
					results[i] = probeResult{observation: o, err: err}
					if err == nil {
						clockMu.Lock()
						results[i].at = clock()
						clockMu.Unlock()
					}
				}
			}()
		}
		wg.Wait()
		next := pending[:0]
		for _, i := range pending {
			if transientProbe(results[i].err) {
				next = append(next, i)
			}
		}
		pending = next
		if len(pending) == 0 || ctx.Err() != nil {
			break
		}
		timer := time.NewTimer(delay)
		select {
		case <-ctx.Done():
			timer.Stop()
		case <-timer.C:
		}
		delay = min(2*delay, time.Second)
	}
	return results
}

func transientProbe(err error) bool {
	var alert tls.AlertError
	var verify *tls.CertificateVerificationError
	var invalid x509.CertificateInvalidError
	var unknown x509.UnknownAuthorityError
	var hostname x509.HostnameError
	var op *net.OpError
	if err == nil || errors.Is(err, context.Canceled) || errors.Is(err, clusterdiagnostic.ErrLocalIdentity) || errors.Is(err, managedtls.ErrPeerRejected) ||
		errors.As(err, &alert) || errors.As(err, &verify) || errors.As(err, &invalid) || errors.As(err, &unknown) || errors.As(err, &hostname) ||
		(errors.As(err, &op) && op.Op == "remote error") {
		return false
	}
	// Every joined cause must be transient; a timeout cannot mask a rejection.
	if joined, ok := err.(interface{ Unwrap() []error }); ok {
		causes := joined.Unwrap()
		if len(causes) == 0 {
			return false
		}
		for _, cause := range causes {
			if !transientProbe(cause) {
				return false
			}
		}
		return true
	}
	if wrapped, ok := err.(interface{ Unwrap() error }); ok {
		if cause := wrapped.Unwrap(); cause != nil {
			// Classify the cause before consulting an outer Timeout method: a
			// timeout-shaped wrapper must not hide joined permanent failures.
			return transientProbe(cause)
		}
	}
	if errors.Is(err, ErrProbeUnavailable) || errors.Is(err, context.DeadlineExceeded) || errors.Is(err, net.ErrClosed) ||
		errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) || errors.Is(err, syscall.ECONNREFUSED) ||
		errors.Is(err, syscall.ECONNRESET) || errors.Is(err, syscall.ECONNABORTED) || errors.Is(err, syscall.EPIPE) ||
		errors.Is(err, syscall.EHOSTUNREACH) || errors.Is(err, syscall.ENETUNREACH) || errors.Is(err, syscall.ENETDOWN) {
		return true
	}
	var timeout net.Error
	return errors.As(err, &timeout) && timeout.Timeout()
}
