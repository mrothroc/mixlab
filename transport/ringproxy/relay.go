// Package ringproxy translates loopback-only MLX streams into exact planned
// mutual-TLS workload streams. It treats collective bytes as opaque and owns
// neither membership selection nor numerical collective semantics.
package ringproxy

import (
	"bytes"
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"net"
	"strconv"
	"sync"
	"time"

	"github.com/mrothroc/mixlab/grouptransport"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/transport/ringtls"
	"github.com/mrothroc/mixlab/trust"
)

type Options struct {
	Accepted  grouptransport.Accepted
	LocalRank int
	Identity  ringtls.Identity
	// CurrentTrust must honor cancellation; it should read a local refreshed
	// snapshot, never perform unbounded network refresh inside a TLS callback.
	CurrentTrust   func(context.Context, time.Time) (trust.Anchor, trust.VerifiedSnapshot, error)
	Clock          func() time.Time
	Public         net.Listener
	ConnectTimeout time.Duration
	MaxConnections int
}
type Relay struct {
	options            Options
	plan               grouptransport.Plan
	listeners          []net.Listener
	mu                 sync.Mutex
	connections        map[net.Conn]bool
	incoming, outgoing chan struct{}
	used, closed       bool
}

// New takes ownership of Public only on success. Every outgoing shim is bound
// before returning, so failure cannot leave a partly activated local transport.
func New(o Options) (*Relay, error) {
	s, _, job, err := o.Accepted.Value()
	if err != nil {
		return nil, err
	}
	p := s.Plan
	if o.LocalRank < 0 || o.LocalRank >= len(p.Members) || p.Members[o.LocalRank].Job != job || o.CurrentTrust == nil || o.Clock == nil || o.Public == nil || o.ConnectTimeout <= 0 || o.ConnectTimeout > time.Minute || o.MaxConnections < 2 || o.MaxConnections > 128 {
		return nil, fmt.Errorf("invalid admitted ring relay options")
	}
	local := p.Members[o.LocalRank]
	if o.Public.Addr().String() != local.Endpoint || len(o.Identity.Chain) != 3 {
		return nil, fmt.Errorf("listener/identity differs from signed transport plan")
	}
	for i := range local.Chain {
		if !bytes.Equal(local.Chain[i], o.Identity.Chain[i]) {
			return nil, fmt.Errorf("relay local key/certificate substituted")
		}
	}
	r := &Relay{options: o, plan: p, connections: map[net.Conn]bool{}, incoming: make(chan struct{}, 1), outgoing: make(chan struct{}, 1)}
	if err := r.verifyMember(o.LocalRank, o.Identity.Chain, o.Clock()); err != nil {
		return nil, err
	}
	if _, err := ringtls.New(o.Identity, r.verifyIncoming, o.Clock); err != nil {
		return nil, err
	}
	for rank := range p.Members {
		if rank != r.nextRank() {
			continue
		}
		l, err := net.Listen("tcp4", r.rawAddress(rank))
		if err != nil {
			for _, opened := range r.listeners {
				_ = opened.Close()
			}
			return nil, fmt.Errorf("loopback shim bind: %w", err)
		}
		r.listeners = append(r.listeners, l)
	}
	return r, nil
}

// MLX v0.32.1's single-lane ring connects to its successor and accepts exactly
// its predecessor. Pin the TLS channel to that topology before forwarding any
// opaque bytes; an arbitrary admitted rank must not fill the predecessor slot.
func (r *Relay) nextRank() int { return (r.options.LocalRank + 1) % len(r.plan.Members) }
func (r *Relay) previousRank() int {
	return (r.options.LocalRank + len(r.plan.Members) - 1) % len(r.plan.Members)
}
func (r *Relay) rawAddress(rank int) string {
	return net.JoinHostPort("127.0.0.1", strconv.Itoa(r.plan.LoopbackPorts[rank]))
}
func (r *Relay) Addresses() [][]string {
	out := make([][]string, len(r.plan.Members))
	for i := range out {
		out[i] = []string{r.rawAddress(i)}
	}
	return out
}

func (r *Relay) verifyMember(rank int, chain [][]byte, now time.Time) error {
	if now.Unix() < r.plan.Created || now.Unix() >= r.plan.Expires || len(chain) != 3 {
		return fmt.Errorf("secure ring attempt expired or chain malformed")
	}
	member := r.plan.Members[rank]
	if nodejob.Hash(chain[0]) != member.CertificateHash {
		return fmt.Errorf("unexpected ring workload certificate")
	}
	for i := range chain {
		if !bytes.Equal(chain[i], member.Chain[i]) {
			return fmt.Errorf("ring certificate chain differs from signed plan")
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	a, v, err := r.options.CurrentTrust(ctx, now)
	if err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	return trust.VerifyWorkload(a, v, chain, member.Binding, now)
}
func (r *Relay) verifyIncoming(chain [][]byte, now time.Time) error {
	if len(chain) != 3 {
		return fmt.Errorf("malformed ring peer chain")
	}
	return r.verifyMember(r.previousRank(), chain, now)
}
func (r *Relay) track(c net.Conn) func() {
	r.mu.Lock()
	if r.closed {
		_ = c.Close()
	} else {
		r.connections[c] = true
	}
	r.mu.Unlock()
	return func() { _ = c.Close(); r.mu.Lock(); delete(r.connections, c); r.mu.Unlock() }
}
func (r *Relay) Close() error {
	_ = r.options.Public.Close()
	for _, l := range r.listeners {
		_ = l.Close()
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.closed = true
	for c := range r.connections {
		_ = c.Close()
	}
	return nil
}

// Run is single-use. A failed admitted stream or listener fails this fixed
// group. Unknown unauthenticated clients are rejected without killing a job.
// Cancellation closes all streams/listeners and waits for relay goroutines.
func (r *Relay) Run(parent context.Context) error {
	r.mu.Lock()
	if r.used || r.closed {
		r.mu.Unlock()
		return fmt.Errorf("ring relay cannot restart")
	}
	r.used = true
	r.mu.Unlock()
	ctx, cancel := context.WithCancelCause(parent)
	defer cancel(nil)
	stop := context.AfterFunc(ctx, func() { _ = r.Close() })
	defer stop()
	defer func() { _ = r.Close() }()
	var wg sync.WaitGroup
	slots := make(chan struct{}, r.options.MaxConnections)
	anonymous := make(chan struct{}, min(16, r.options.MaxConnections))
	serve := func(l net.Listener, rank int) {
		defer wg.Done()
		for {
			c, err := l.Accept()
			if err != nil {
				if ctx.Err() == nil {
					cancel(fmt.Errorf("ring listener: %w", err))
				}
				return
			}
			budget := slots
			if rank < 0 {
				budget = anonymous
			}
			select {
			case budget <- struct{}{}:
			default:
				_ = c.Close()
				continue
			}
			wg.Add(1)
			go func() {
				defer wg.Done()
				var release sync.Once
				releaseBudget := func() { release.Do(func() { <-budget }) }
				defer releaseBudget()
				defer r.track(c)()
				if err := r.stream(ctx, c, rank, releaseBudget); err != nil && ctx.Err() == nil {
					cancel(err)
				}
			}()
		}
	}
	wg.Add(1)
	go serve(r.options.Public, -1)
	i := 0
	for rank := range r.plan.Members {
		if rank != r.nextRank() {
			continue
		}
		wg.Add(1)
		go serve(r.listeners[i], rank)
		i++
	}
	// Revalidate active streams too; expiry/revocation cannot be bypassed by
	// retaining an already-established TLS connection indefinitely.
	wg.Add(1)
	go func() {
		defer wg.Done()
		ticker := time.NewTicker(time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				for rank, m := range r.plan.Members {
					if err := r.verifyMember(rank, m.Chain, r.options.Clock()); err != nil {
						cancel(err)
						return
					}
				}
			}
		}
	}()
	<-ctx.Done()
	_ = r.Close()
	wg.Wait()
	return context.Cause(ctx)
}

func (r *Relay) stream(ctx context.Context, raw net.Conn, rank int, authenticated func()) error {
	handshake, done := context.WithTimeout(ctx, r.options.ConnectTimeout)
	defer done()
	var peer net.Conn
	if rank < 0 {
		policy, err := ringtls.New(r.options.Identity, r.verifyIncoming, r.options.Clock)
		if err != nil {
			return err
		}
		tlsConn := tls.Server(raw, policy.ServerConfig())
		short, stop := context.WithTimeout(handshake, 2*time.Second)
		err = tlsConn.HandshakeContext(short)
		stop()
		if err != nil {
			return nil
		} // unauthenticated traffic is not a group failure
		authenticated()
		select {
		case r.incoming <- struct{}{}:
			defer func() { <-r.incoming }()
		default:
			return nil
		}
		peer, err = r.dialLocal(handshake)
		if err != nil {
			return fmt.Errorf("admitted ring peer cannot reach local MLX: %w", err)
		}
		raw = tlsConn
	} else {
		select {
		case r.outgoing <- struct{}{}:
			defer func() { <-r.outgoing }()
		default:
			return nil
		}
		policy, err := ringtls.New(r.options.Identity, func(c [][]byte, t time.Time) error { return r.verifyMember(rank, c, t) }, r.options.Clock)
		if err != nil {
			return err
		}
		d := tls.Dialer{NetDialer: &net.Dialer{Timeout: r.options.ConnectTimeout}, Config: policy.ClientConfig()}
		peer, err = d.DialContext(handshake, "tcp", r.plan.Members[rank].Endpoint)
		if err != nil {
			return fmt.Errorf("secure ring connection: %w", err)
		}
	}
	defer r.track(peer)()
	// Socket expiry remains effective even while a trust refresh is delayed.
	deadline := time.Now().Add(time.Unix(r.plan.Expires, 0).Sub(r.options.Clock()))
	if err := raw.SetDeadline(deadline); err != nil {
		return err
	}
	if err := peer.SetDeadline(deadline); err != nil {
		return err
	}
	results := make(chan error, 2)
	copyStream := func(dst, src net.Conn) {
		_, err := io.CopyBuffer(dst, src, make([]byte, 64<<10))
		if err == nil {
			if cw, ok := dst.(interface{ CloseWrite() error }); ok {
				err = cw.CloseWrite()
			}
		}
		results <- err
	}
	go copyStream(peer, raw)
	go copyStream(raw, peer)
	first := <-results
	if first != nil {
		_ = raw.Close()
		_ = peer.Close()
	}
	second := <-results
	return errors.Join(first, second)
}

func (r *Relay) dialLocal(ctx context.Context) (net.Conn, error) {
	d := net.Dialer{Timeout: time.Second}
	for {
		c, err := d.DialContext(ctx, "tcp4", r.rawAddress(r.options.LocalRank))
		if err == nil {
			return c, nil
		}
		timer := time.NewTimer(25 * time.Millisecond)
		select {
		case <-ctx.Done():
			timer.Stop()
			return nil, ctx.Err()
		case <-timer.C:
		}
	}
}
