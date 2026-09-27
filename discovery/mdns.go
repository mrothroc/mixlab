package discovery

import (
	"context"
	"fmt"
	"io"
	"log"
	"net"
	"strconv"
	"sync"
	"time"

	"github.com/hashicorp/mdns"
)

// MDNS is link-local only. Interface and IPs are operator-selected local
// settings, not values obtained from advertisements.
type MDNS struct {
	Interface *net.Interface
	IPs       []net.IP
	Timeout   time.Duration
}

func (p MDNS) Browse(ctx context.Context, service Service) ([]Hint, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if !service.valid() {
		return nil, fmt.Errorf("invalid discovery service")
	}
	timeout := p.Timeout
	if timeout == 0 {
		timeout = time.Second
	}
	if timeout < 0 || timeout > 10*time.Second {
		return nil, fmt.Errorf("discovery timeout must be in (0,10s]")
	}
	if deadline, ok := ctx.Deadline(); ok && time.Until(deadline) < timeout {
		timeout = time.Until(deadline)
	}
	if timeout <= 0 {
		return nil, context.DeadlineExceeded
	}
	entries := make(chan *mdns.ServiceEntry, MaxHints)
	params := mdns.DefaultParams(string(service))
	params.Interface, params.Timeout, params.Entries = p.Interface, timeout, entries
	params.Logger = log.New(io.Discard, "", 0)
	// v1.0.6 mutates published entry pointers until Query returns, and its
	// concurrent cancellation/Close path races. Query synchronously into a
	// bounded channel, then decode only after it has stopped writing. Caller
	// cancellation is observed after this bounded (at most 10s) query.
	if err := mdns.Query(params); err != nil {
		return nil, fmt.Errorf("mDNS browse: %w", err)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	hints := make(map[string]Hint)
	consume := func(e *mdns.ServiceEntry) {
		for _, h := range entryHints(service, e) {
			// Bound memory even on a hostile multicast link. Never replace an
			// endpoint's first hint with later unauthenticated claims.
			if len(hints) < MaxHints {
				if _, exists := hints[h.Endpoint]; !exists {
					hints[h.Endpoint] = h
				}
			}
		}
	}
	for len(entries) > 0 {
		consume(<-entries)
	}
	out := make([]Hint, 0, len(hints))
	for _, h := range hints {
		out = append(out, h)
	}
	return ordered(out), nil
}

func entryHints(service Service, e *mdns.ServiceEntry) []Hint {
	if e == nil {
		return nil
	}
	claims, err := DecodeTXT(service, e.Port, e.InfoFields)
	if err != nil {
		return nil
	}
	var addresses []string
	if e.AddrV4 != nil {
		addresses = append(addresses, e.AddrV4.String())
	}
	if e.AddrV6IPAddr != nil {
		addresses = append(addresses, e.AddrV6IPAddr.String())
	}
	var out []Hint
	for _, address := range addresses {
		endpoint := net.JoinHostPort(address, strconv.Itoa(e.Port))
		if validEndpoint(endpoint) == nil {
			out = append(out, Hint{Service: service, Endpoint: endpoint, Claims: claims})
		}
	}
	return out
}

type advertisement struct {
	stop func() error
	once sync.Once
	done chan struct{}
	err  error
}

func (a *advertisement) Close() error {
	a.once.Do(func() { a.err = a.stop(); close(a.done) })
	return a.err
}

func (p MDNS) Advertise(ctx context.Context, a Advertisement) (io.Closer, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	txt, err := a.TXT()
	if err != nil {
		return nil, err
	}
	if len(p.IPs) == 0 || len(p.IPs) > 8 {
		return nil, fmt.Errorf("mDNS advertisement requires 1..8 explicit local IPs")
	}
	ips := make([]net.IP, len(p.IPs))
	for i, ip := range p.IPs {
		if ip == nil || ip.IsUnspecified() || ip.IsMulticast() {
			return nil, fmt.Errorf("invalid advertised IP")
		}
		ips[i] = append(net.IP(nil), ip...)
	}
	// A stable opaque instance avoids leaking the OS hostname in SRV records.
	zone, err := mdns.NewMDNSService(a.Instance, string(a.Service), "local.", a.Instance+".local.", a.Port, ips, txt)
	if err != nil {
		return nil, err
	}
	server, err := mdns.NewServer(&mdns.Config{Zone: zone, Iface: p.Interface, Logger: log.New(io.Discard, "", 0)})
	if err != nil {
		return nil, err
	}
	out := &advertisement{stop: server.Shutdown, done: make(chan struct{})}
	go func() {
		select {
		case <-ctx.Done():
			_ = out.Close()
		case <-out.done:
		}
	}()
	return out, nil
}
