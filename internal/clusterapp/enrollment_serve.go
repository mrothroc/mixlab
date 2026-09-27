package clusterapp

import (
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"net"
	"net/http"
	"sync"
	"time"

	"github.com/mrothroc/mixlab/transport/enrollmenttls"
	"github.com/mrothroc/mixlab/trust/enrollment"
)

// ServeEnrollment owns connections on an explicitly supplied listener. It never
// opens a port implicitly; the CLI must select the interface and policy first.
// All children and live request evidence are closed before this call returns.
func ServeEnrollment(ctx context.Context, listener net.Listener, identity enrollmenttls.Identity, service *enrollment.Service, window enrollment.Window, current CurrentTrust, clock func() time.Time) (result error) {
	if listener == nil || service == nil || current == nil || clock == nil {
		return fmt.Errorf("enrollment server dependencies required")
	}
	if err := enrollmenttls.ValidateListener(listener, window.Interface); err != nil {
		return err
	}
	return serveBootstrap(ctx, listener, identity, time.Unix(window.Expires, 0), clock, func(c *enrollmenttls.Channel) http.Handler {
		h, err := EnrollmentHandler(service, window, c, current, clock)
		if err != nil {
			return http.NotFoundHandler()
		}
		return h
	}, func(ctx context.Context, c *enrollmenttls.Channel) error { return service.Disconnect(ctx, c) })
}

func serveBootstrap(ctx context.Context, listener net.Listener, identity enrollmenttls.Identity, expires time.Time, clock func() time.Time, handler func(*enrollmenttls.Channel) http.Handler, disconnected func(context.Context, *enrollmenttls.Channel) error) (result error) {
	config, err := enrollmenttls.ServerConfig(identity, clock)
	if err != nil {
		return err
	}
	remaining := expires.Sub(clock())
	if remaining <= 0 || remaining > time.Hour {
		return fmt.Errorf("bounded active enrollment window required")
	}
	life, cancel := context.WithTimeout(ctx, remaining)
	defer cancel()
	stop := context.AfterFunc(life, func() { _ = listener.Close() })
	defer stop()
	defer func() { _ = listener.Close() }()
	var wg sync.WaitGroup
	cleanupErrors := make(chan error, 1)
	defer func() {
		cancel()
		wg.Wait()
		select {
		case err := <-cleanupErrors:
			result = errors.Join(result, err)
		default:
		}
	}()
	permits := make(chan struct{}, 16)
	for {
		raw, err := listener.Accept()
		if err != nil {
			cancel()
			if life.Err() != nil && (errors.Is(err, net.ErrClosed) || ctx.Err() != nil) {
				return nil
			}
			return err
		}
		select {
		case permits <- struct{}{}:
		default:
			_ = raw.Close()
			continue
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			defer func() { <-permits; _ = raw.Close() }()
			conn := tls.Server(raw, config)
			handshake, done := context.WithTimeout(life, 15*time.Second)
			err := conn.HandshakeContext(handshake)
			done()
			if err != nil {
				return
			}
			_ = enrollmenttls.ServeConnection(life, conn, handler, func(c *enrollmenttls.Channel) {
				cleanup, done := context.WithTimeout(context.Background(), 5*time.Second)
				defer done()
				if err := disconnected(cleanup, c); err != nil {
					select {
					case cleanupErrors <- err:
					default:
					}
					cancel()
				}
			})
		}()
	}
}
