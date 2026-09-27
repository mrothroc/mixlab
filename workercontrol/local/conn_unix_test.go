//go:build darwin || linux

package local

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"net"
	"sync"
	"testing"
	"time"

	wc "github.com/mrothroc/mixlab/workercontrol"
)

func connected(t *testing.T, budget Limits) (*Conn, *net.UnixConn) {
	t.Helper()
	path := directory(t).Dir() + "/unit.sock"
	l, err := net.ListenUnix("unix", &net.UnixAddr{Name: path, Net: "unix"})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = l.Close() }()
	if err := l.SetDeadline(time.Now().Add(time.Second)); err != nil {
		t.Fatal(err)
	}
	client, err := net.DialUnix("unix", nil, &net.UnixAddr{Name: path, Net: "unix"})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = client.Close() })
	server, err := l.AcceptUnix()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = server.Close() })
	return newConn(server, binding(), budget), client
}

func TestConnReceiveFailures(t *testing.T) {
	for _, mode := range []string{"wrong-job", "bad-version", "oversize", "byte-budget", "message-budget", "replay"} {
		t.Run(mode, func(t *testing.T) {
			budget := limits()
			if mode == "byte-budget" {
				budget.Bytes = 5
			}
			if mode == "message-budget" {
				budget.Messages = 1
			}
			c, peer := connected(t, budget)
			e := message(1)
			switch mode {
			case "wrong-job":
				e.JobID = "different"
			case "bad-version":
				e.Version = "unknown"
				body, err := json.Marshal(e)
				if err != nil {
					t.Fatal(err)
				}
				frame := make([]byte, 4+len(body))
				binary.BigEndian.PutUint32(frame, uint32(len(body)))
				copy(frame[4:], body)
				if _, err := peer.Write(frame); err != nil {
					t.Fatal(err)
				}
			case "oversize":
				// A wire header too large for this endpoint is rejected before allocation.
				var header [4]byte
				binary.BigEndian.PutUint32(header[:], budget.FrameBytes+1)
				if _, err := peer.Write(header[:]); err != nil {
					t.Fatal(err)
				}
			default:
			}
			if mode != "bad-version" && mode != "oversize" {
				if err := wc.WriteFrame(peer, e, 4096); err != nil {
					t.Fatal(err)
				}
			}
			if mode == "replay" || mode == "message-budget" {
				if _, err := c.Receive(bounded(t)); err != nil {
					t.Fatal(err)
				}
				if err := wc.WriteFrame(peer, e, 4096); err != nil {
					t.Fatal(err)
				}
			}
			if _, err := c.Receive(bounded(t)); err == nil {
				t.Fatal("invalid message accepted")
			}
			if _, err := c.Receive(bounded(t)); err == nil {
				t.Fatal("session not closed")
			}
		})
	}
}

func TestConnSendFailures(t *testing.T) {
	for _, mode := range []string{"wrong-attempt", "byte-budget", "frame-budget", "message-budget", "replay"} {
		t.Run(mode, func(t *testing.T) {
			budget := limits()
			if mode == "byte-budget" {
				budget.Bytes = 5
			}
			if mode == "frame-budget" {
				budget.FrameBytes = 5
			}
			if mode == "message-budget" {
				budget.Messages = 1
			}
			c, _ := connected(t, budget)
			e := message(1)
			if mode == "wrong-attempt" {
				e.AttemptID = "other"
			}
			if mode == "replay" || mode == "message-budget" {
				if err := c.Send(bounded(t), e); err != nil {
					t.Fatal(err)
				}
			}
			if err := c.Send(bounded(t), e); err == nil {
				t.Fatal("invalid send accepted")
			}
			if err := c.Send(bounded(t), message(2)); err == nil {
				t.Fatal("session not closed")
			}
		})
	}
}

func TestConnDeadlinesCancellationAndClose(t *testing.T) {
	for _, mode := range []string{"missing", "expired", "canceled", "owner-close", "read-timeout", "backpressure"} {
		t.Run(mode, func(t *testing.T) {
			budget := Limits{FrameBytes: wc.MaxFrameBytes, Bytes: 1 << 30, Messages: 10000}
			c, _ := connected(t, budget)
			ctx, cancel := context.WithTimeout(context.Background(), 80*time.Millisecond)
			defer cancel()
			if mode == "missing" {
				ctx = context.Background()
			}
			if mode == "expired" {
				cancel()
			}
			var closer sync.WaitGroup
			if mode == "canceled" || mode == "owner-close" {
				closer.Add(1)
				go func() {
					defer closer.Done()
					time.Sleep(10 * time.Millisecond)
					if mode == "canceled" {
						cancel()
					} else {
						_ = c.Close()
					}
				}()
			}
			start := time.Now()
			var err error
			if mode == "backpressure" {
				for i := uint64(1); err == nil && i < 10000; i++ {
					err = c.Send(ctx, message(i))
				}
			} else {
				_, err = c.Receive(ctx)
			}
			closer.Wait()
			if err == nil || time.Since(start) > time.Second {
				t.Fatalf("unbounded %s: %v", mode, err)
			}
			if mode == "missing" && !errors.Is(err, ErrDeadline) {
				t.Fatal(err)
			}
		})
	}
}

func TestLimitsValidation(t *testing.T) {
	for _, value := range []Limits{{}, {FrameBytes: wc.MaxFrameBytes + 1, Bytes: 10, Messages: 1}, {FrameBytes: 1, Bytes: 3, Messages: 1}, {FrameBytes: 1, Bytes: 10}} {
		if value.validate() == nil {
			t.Fatalf("accepted %+v", value)
		}
	}
}
