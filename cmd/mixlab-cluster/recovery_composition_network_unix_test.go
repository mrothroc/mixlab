//go:build darwin || linux

package main

import (
	"context"
	"net"
	"sync"
	"sync/atomic"
)

// Stall real coordinator TLS traffic without replacing its authenticated
// handlers or changing durable state. The fixture lifetime bounds every wait.
type compositionNetworkPause struct {
	mu    sync.Mutex
	gate  chan struct{}
	waits atomic.Int32
}

func (p *compositionNetworkPause) start() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.gate = make(chan struct{})
}

func (p *compositionNetworkPause) release() {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.gate != nil {
		close(p.gate)
		p.gate = nil
	}
}

type compositionPausedListener struct {
	net.Listener
	pause *compositionNetworkPause
	ctx   context.Context
}

func (l *compositionPausedListener) Accept() (net.Conn, error) {
	c, err := l.Listener.Accept()
	if err != nil {
		return nil, err
	}
	return &compositionPausedConn{Conn: c, listener: l}, nil
}

type compositionPausedConn struct {
	net.Conn
	listener *compositionPausedListener
}

func (c *compositionPausedConn) Write(b []byte) (int, error) {
	p := c.listener.pause
	p.mu.Lock()
	gate := p.gate
	p.mu.Unlock()
	if gate != nil {
		p.waits.Add(1)
		select {
		case <-gate:
		case <-c.listener.ctx.Done():
			return 0, c.listener.ctx.Err()
		}
	}
	return c.Conn.Write(b)
}
