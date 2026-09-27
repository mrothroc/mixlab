package discovery

import (
	"context"
	"fmt"
	"io"
	"net"
	"strconv"
	"sync"
)

// Fake is an in-memory link for deterministic application tests. Each
// registration has its own lifetime, including duplicate advertisements.
type Fake struct {
	mu      sync.Mutex
	next    uint64
	records map[uint64]Hint
}

func (f *Fake) Browse(ctx context.Context, s Service) ([]Hint, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if !s.valid() {
		return nil, fmt.Errorf("invalid discovery service")
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	var out []Hint
	// Iterate in registration order so duplicate hints are deterministic.
	for id := uint64(1); id <= f.next; id++ {
		if h, ok := f.records[id]; ok && h.Service == s {
			out = append(out, h)
		}
	}
	return ordered(out), nil
}

func (f *Fake) Advertise(ctx context.Context, a Advertisement) (io.Closer, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	txt, err := a.TXT()
	if err != nil {
		return nil, err
	}
	claims, err := DecodeTXT(a.Service, a.Port, txt)
	if err != nil {
		return nil, err
	}
	f.mu.Lock()
	if len(f.records) >= MaxHints {
		f.mu.Unlock()
		return nil, fmt.Errorf("fake discovery capacity exceeded")
	}
	if f.records == nil {
		f.records = make(map[uint64]Hint)
	}
	f.next++
	id := f.next
	f.records[id] = Hint{Service: a.Service, Endpoint: net.JoinHostPort("127.0.0.1", strconv.Itoa(a.Port)), Claims: claims}
	f.mu.Unlock()
	out := &advertisement{done: make(chan struct{}), stop: func() error { f.mu.Lock(); defer f.mu.Unlock(); delete(f.records, id); return nil }}
	go func() {
		select {
		case <-ctx.Done():
			_ = out.Close()
		case <-out.done:
		}
	}()
	return out, nil
}
