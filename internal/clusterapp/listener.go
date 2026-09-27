package clusterapp

import (
	"net"
	"sync"
)

// limitListener bounds accepted live connections without an unbounded queue.
// Closing a connection releases exactly one permit, including failed TLS.
type limitListener struct {
	net.Listener
	permits chan struct{}
}

func (l *limitListener) Accept() (net.Conn, error) {
	for {
		c, err := l.Listener.Accept()
		if err != nil {
			return nil, err
		}
		select {
		case l.permits <- struct{}{}:
			return &limitedConn{Conn: c, release: func() { <-l.permits }}, nil
		default:
			_ = c.Close()
		}
	}
}

type limitedConn struct {
	net.Conn
	once    sync.Once
	release func()
}

func (c *limitedConn) Close() error { err := c.Conn.Close(); c.once.Do(c.release); return err }
