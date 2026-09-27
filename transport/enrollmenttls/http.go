package enrollmenttls

import (
	"context"
	"crypto/tls"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"sync"
	"time"
)

// ServeConnection serves bounded HTTP/1.1 messages over an already completed
// dedicated enrollment TLS connection. It never listens or accepts plaintext.
// A wrapper deliberately hides the tls.Conn from net/http, whose default ALPN
// dispatcher cannot serve our private enrollment protocol. TLS remains below
// every HTTP read/write, and r.TLS is populated from the actual connection.
func ServeConnection(ctx context.Context, conn *tls.Conn, handler func(*Channel) http.Handler, onClosed func(*Channel)) error {
	if handler == nil || onClosed == nil {
		return fmt.Errorf("enrollment handler and disconnect callback required")
	}
	channel, err := New(ctx, conn)
	if err != nil {
		return err
	}
	defer onClosed(channel)
	defer func() { _ = channel.Close() }()
	h := handler(channel)
	if h == nil {
		return fmt.Errorf("enrollment handler required")
	}
	c := &httpConn{Conn: conn, done: make(chan struct{})}
	l := &singleListener{conn: c}
	stop := context.AfterFunc(ctx, func() { _ = c.Close() })
	defer stop()
	state := conn.ConnectionState()
	deadline, _ := ctx.Deadline() // New already requires a deadline within one hour.
	s := &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { r.TLS = &state; h.ServeHTTP(w, r) }),
		ReadHeaderTimeout: 5 * time.Second, ReadTimeout: 10 * time.Second, WriteTimeout: 10 * time.Second, IdleTimeout: time.Until(deadline), MaxHeaderBytes: 8 << 10,
		BaseContext: func(net.Listener) context.Context { return ctx }, ErrorLog: log.New(io.Discard, "", 0)}
	err = s.Serve(l)
	shutdown, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	shutdownErr := s.Shutdown(shutdown)
	if err == http.ErrServerClosed || err == net.ErrClosed {
		err = nil
	}
	if err != nil {
		return err
	}
	return shutdownErr
}

type httpConn struct {
	net.Conn
	once sync.Once
	done chan struct{}
}

func (c *httpConn) Close() error {
	var err error
	c.once.Do(func() { _ = c.SetDeadline(time.Now()); err = c.Conn.Close(); close(c.done) })
	return err
}

type singleListener struct {
	conn     *httpConn
	accepted bool
}

func (l *singleListener) Accept() (net.Conn, error) {
	if !l.accepted {
		l.accepted = true
		return l.conn, nil
	}
	<-l.conn.done
	return nil, net.ErrClosed
}
func (l *singleListener) Close() error   { return l.conn.Close() }
func (l *singleListener) Addr() net.Addr { return l.conn.LocalAddr() }

// HTTPClient cannot reconnect, redirect, use an environment proxy, or move an
// enrollment request to another connection. The caller owns conn and must close
// it on completion, rejection, error, or expiry. Requests use absolute HTTPS
// URLs for the exact endpoint selected before creating the connection.
func HTTPClient(conn *tls.Conn, endpoint string, timeout time.Duration) (*http.Client, error) {
	if conn == nil || timeout <= 0 || timeout > time.Minute {
		return nil, fmt.Errorf("bounded enrollment HTTP client required")
	}
	s := conn.ConnectionState()
	if !s.HandshakeComplete || s.Version != tls.VersionTLS13 || s.DidResume || s.NegotiatedProtocol != Protocol {
		return nil, fmt.Errorf("dedicated enrollment TLS required")
	}
	origin, err := http.NewRequest(http.MethodGet, endpoint, nil)
	if err != nil || origin.URL.Scheme != "https" || origin.URL.Hostname() == "" || origin.URL.User != nil || origin.URL.RawQuery != "" || origin.URL.Fragment != "" || (origin.URL.Path != "" && origin.URL.Path != "/") {
		return nil, fmt.Errorf("invalid enrollment origin")
	}
	var mu sync.Mutex
	used := false
	t := &http.Transport{ForceAttemptHTTP2: false, MaxResponseHeaderBytes: 8 << 10, IdleConnTimeout: time.Hour, ResponseHeaderTimeout: timeout,
		DialTLSContext: func(context.Context, string, string) (net.Conn, error) {
			mu.Lock()
			defer mu.Unlock()
			if used {
				return nil, fmt.Errorf("enrollment connection cannot reconnect")
			}
			used = true
			return conn, nil
		}}
	return &http.Client{Transport: &enrollmentRoundTripper{transport: t, origin: origin.URL.Host}, Timeout: timeout, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}, nil
}

type enrollmentRoundTripper struct {
	transport *http.Transport
	origin    string
}

func (t *enrollmentRoundTripper) RoundTrip(r *http.Request) (*http.Response, error) {
	if r.URL == nil || r.URL.Scheme != "https" || r.URL.Host != t.origin || r.URL.User != nil || r.URL.Fragment != "" || r.Host != "" && r.Host != t.origin {
		return nil, fmt.Errorf("enrollment request changed origin")
	}
	return t.transport.RoundTrip(r)
}
func (t *enrollmentRoundTripper) CloseIdleConnections() { t.transport.CloseIdleConnections() }
