//go:build darwin || linux

package local

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"
	"testing"
	"time"

	wc "github.com/mrothroc/mixlab/workercontrol"
)

func dispatchPair(t *testing.T, options DispatcherOptions) (*Dispatcher, *Dispatcher) {
	t.Helper()
	c, peer := connected(t, Limits{FrameBytes: wc.MaxFrameBytes, Bytes: 64 << 20, Messages: 10000})
	a, err := NewDispatcher(context.Background(), c, options)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = a.Close() })
	b, err := NewDispatcher(context.Background(), newConn(peer, binding(), c.limits), options)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = b.Close() })
	return a, b
}

func opaque(kind wc.Kind, n int) wc.Envelope {
	return wc.Envelope{Kind: kind, PayloadKind: "opaque_test_v7", PayloadVersion: 7, Payload: []byte(fmt.Sprintf(`{"number":%d}`, n))}
}

func dispatchEvent(t *testing.T, ch <-chan wc.Envelope) wc.Envelope {
	t.Helper()
	select {
	case e, ok := <-ch:
		if !ok {
			t.Fatal("event channel closed")
		}
		return e
	case <-time.After(3 * time.Second):
		t.Fatal("event timeout")
		return wc.Envelope{}
	}
}

func dispatchDone(t *testing.T, d *Dispatcher) {
	t.Helper()
	select {
	case <-d.Done():
	case <-time.After(3 * time.Second):
		t.Fatal("dispatcher did not stop")
	}
}

func TestDispatcherCorrelatedConcurrentRequests(t *testing.T) {
	a, b := dispatchPair(t, DispatcherOptions{MaxInFlight: 64, SendBuffer: 64, EventBuffer: 64})
	const count = 40
	var wg sync.WaitGroup
	errs := make(chan error, count)
	for i := 0; i < count; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			req := opaque(wc.KindRoundRequest, i)
			r, err := a.Request(context.Background(), req)
			if err == nil && (string(r.Payload) != string(req.Payload) || r.PayloadKind != req.PayloadKind || r.PayloadVersion != 7) {
				err = fmt.Errorf("misrouted reply %s", r.Payload)
			}
			errs <- err
		}()
	}
	requests := make([]wc.Envelope, count)
	for i := range requests {
		requests[i] = dispatchEvent(t, b.Events())
		if requests[i].Sequence != uint64(i+1) {
			t.Fatalf("out of order sequence %d", requests[i].Sequence)
		}
	}
	for i := count - 1; i >= 0; i-- {
		response := requests[i]
		response.Kind = wc.KindRoundResponse
		if err := b.Reply(context.Background(), requests[i], response); err != nil {
			t.Fatal(err)
		}
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}
	if len(a.Events()) != 0 {
		t.Fatal("reply leaked to event queue")
	}
}

func TestDispatcherBidirectionalRequests(t *testing.T) {
	a, b := dispatchPair(t, DispatcherOptions{})
	result := make(chan error, 2)
	for _, d := range []*Dispatcher{a, b} {
		go func() { _, err := d.Request(context.Background(), opaque(wc.KindArtifactRead, 1)); result <- err }()
	}
	for _, d := range []*Dispatcher{a, b} {
		e := dispatchEvent(t, d.Events())
		if err := d.Reply(context.Background(), e, opaque(wc.KindArtifactRead, 2)); err != nil {
			t.Fatal(err)
		}
	}
	for i := 0; i < 2; i++ {
		if err := <-result; err != nil {
			t.Fatal(err)
		}
	}
}

func TestDispatcherCancellationAndLateReply(t *testing.T) {
	for _, mode := range []string{"cancel", "deadline"} {
		t.Run(mode, func(t *testing.T) {
			a, b := dispatchPair(t, DispatcherOptions{MaxInFlight: 1, RequestTimeout: 100 * time.Millisecond})
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			result := make(chan error, 1)
			go func() { _, err := a.Request(ctx, opaque(wc.KindRoundRequest, 1)); result <- err }()
			request := dispatchEvent(t, b.Events())
			if _, err := a.Request(context.Background(), opaque(wc.KindRoundRequest, 2)); !errors.Is(err, ErrBackpressure) {
				t.Fatalf("in-flight bound: %v", err)
			}
			if mode == "cancel" {
				cancel()
			}
			err := <-result
			want := context.DeadlineExceeded
			if mode == "cancel" {
				want = context.Canceled
			}
			if !errors.Is(err, want) {
				t.Fatalf("cancellation: %v", err)
			}
			if err := b.Reply(context.Background(), request, opaque(wc.KindRoundResponse, 1)); err != nil {
				t.Fatal(err)
			}
			if err := b.Reply(context.Background(), request, opaque(wc.KindRoundResponse, 1)); err != nil {
				t.Fatal(err)
			}
			if err := b.Send(context.Background(), opaque(wc.KindProgress, 3)); err != nil {
				t.Fatal(err)
			}
			if e := dispatchEvent(t, a.Events()); string(e.Payload) != `{"number":3}` {
				t.Fatalf("late reply leaked: %+v", e)
			}
			go func() { _, err := a.Request(context.Background(), opaque(wc.KindRoundRequest, 4)); result <- err }()
			next := dispatchEvent(t, b.Events())
			if next.CorrelationID == request.CorrelationID {
				t.Fatal("correlation reused")
			}
			if err := b.Reply(context.Background(), next, opaque(wc.KindRoundResponse, 4)); err != nil {
				t.Fatal(err)
			}
			if err := <-result; err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestDispatcherInboundBoundsAndControlIsolation(t *testing.T) {
	for _, kind := range []wc.Kind{wc.KindProgress, wc.KindHeartbeat} {
		t.Run(string(kind), func(t *testing.T) {
			a, b := dispatchPair(t, DispatcherOptions{EventBuffer: 1, ControlBuffer: 1, EnqueueTimeout: 50 * time.Millisecond})
			if err := b.Send(context.Background(), opaque(kind, 1)); err != nil {
				t.Fatal(err)
			}
			// A full event queue does not stop the independent control queue.
			if kind == wc.KindProgress {
				if err := b.Send(context.Background(), opaque(wc.KindCancellation, 2)); err != nil {
					t.Fatal(err)
				}
				if e := dispatchEvent(t, a.Controls()); e.Kind != wc.KindCancellation {
					t.Fatal(e)
				}
			}
			_ = b.Send(context.Background(), opaque(kind, 3))
			dispatchDone(t, a)
			if !errors.Is(a.Err(), ErrBackpressure) {
				t.Fatalf("overflow: %v", a.Err())
			}
		})
	}
}

func TestDispatcherPriorityAndOutboundBounds(t *testing.T) {
	c, peer := connected(t, Limits{FrameBytes: wc.MaxFrameBytes, Bytes: 64 << 20, Messages: 100})
	if err := c.conn.SetWriteBuffer(1024); err != nil {
		t.Fatal(err)
	}
	d, err := NewDispatcher(context.Background(), c, DispatcherOptions{SendBuffer: 1, ControlSendBuffer: 1, WriteTimeout: 2 * time.Second})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = d.Close() }()
	big := opaque(wc.KindArtifactWrite, 0)
	big.Payload = []byte(`"` + strings.Repeat("x", 256<<10) + `"`)
	op, _, err := d.submit(context.Background(), big, false)
	if err != nil {
		t.Fatal(err)
	}
	// Reading only the frame header proves the large write has started, while
	// keeping it blocked until normal and control work have both been queued.
	if err := peer.SetReadDeadline(time.Now().Add(3 * time.Second)); err != nil {
		t.Fatal(err)
	}
	var header [4]byte
	if _, err := io.ReadFull(peer, header[:]); err != nil {
		t.Fatal(err)
	}
	queuedCtx, cancel := context.WithCancel(context.Background())
	normal, _, err := d.submit(queuedCtx, opaque(wc.KindProgress, 1), false)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := d.submit(context.Background(), opaque(wc.KindProgress, 2), false); !errors.Is(err, ErrBackpressure) {
		t.Fatal(err)
	}
	control, _, err := d.submit(context.Background(), opaque(wc.KindHeartbeat, 3), false)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := d.submit(context.Background(), opaque(wc.KindCancellation, 4), false); !errors.Is(err, ErrBackpressure) {
		t.Fatal(err)
	}
	cancel()
	// Reconstitute the header only in the reader, not on the socket.
	reader := io.MultiReader(bytes.NewReader(header[:]), peer)
	first, err := wc.ReadFrame(reader, wc.MaxFrameBytes)
	if err != nil || first.Sequence != 1 {
		t.Fatalf("first: %v %+v", err, first)
	}
	second, err := wc.ReadFrame(peer, wc.MaxFrameBytes)
	if err != nil || second.Kind != wc.KindHeartbeat || second.Sequence != 2 {
		t.Fatalf("priority: %v %+v", err, second)
	}
	if err := <-op.result; err != nil {
		t.Fatal(err)
	}
	if err := <-control.result; err != nil {
		t.Fatal(err)
	}
	if err := <-normal.result; !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
}

func TestDispatcherBlockedWriteAndSessionShutdown(t *testing.T) {
	for _, mode := range []string{"write-timeout", "read-timeout", "cancel", "close", "disconnect"} {
		t.Run(mode, func(t *testing.T) {
			c, peer := connected(t, Limits{FrameBytes: wc.MaxFrameBytes, Bytes: 64 << 20, Messages: 100})
			if err := c.conn.SetWriteBuffer(1024); err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			o := DispatcherOptions{WriteTimeout: 50 * time.Millisecond}
			if mode == "read-timeout" {
				o.ReadTimeout = 50 * time.Millisecond
			}
			d, err := NewDispatcher(ctx, c, o)
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = d.Close() }()
			result := make(chan error, 1)
			request := opaque(wc.KindRoundRequest, 1)
			if mode == "write-timeout" {
				request.Payload = []byte(`"` + strings.Repeat("x", 256<<10) + `"`)
			}
			go func() { _, err := d.Request(context.Background(), request); result <- err }()
			if mode != "write-timeout" {
				if err := peer.SetReadDeadline(time.Now().Add(time.Second)); err != nil {
					t.Fatal(err)
				}
				if _, err := wc.ReadFrame(peer, wc.MaxFrameBytes); err != nil {
					t.Fatal(err)
				}
			}
			switch mode {
			case "cancel":
				cancel()
			case "close":
				_ = d.Close()
			case "disconnect":
				_ = peer.Close()
			}
			dispatchDone(t, d)
			if err := <-result; err == nil {
				t.Fatal("pending request survived shutdown")
			}
			if d.Err() == nil {
				t.Fatal("missing terminal cause")
			}
		})
	}
}

func TestDispatcherValidationAndSequenceContinuation(t *testing.T) {
	c, peer := connected(t, limits())
	if err := c.Send(bounded(t), message(12)); err != nil {
		t.Fatal(err)
	}
	if _, err := wc.ReadFrame(peer, 4096); err != nil {
		t.Fatal(err)
	}
	if _, err := NewDispatcher(context.Background(), c, DispatcherOptions{MaxInFlight: -1}); !errors.Is(err, ErrDispatcherOptions) {
		t.Fatal(err)
	}
	d, err := NewDispatcher(context.Background(), c, DispatcherOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = d.Close() }()
	bad := opaque(wc.KindProgress, 1)
	bad.JobID = "wrong"
	if err := d.Send(context.Background(), bad); !errors.Is(err, wc.ErrBinding) {
		t.Fatal(err)
	}
	bad = opaque(wc.KindProgress, 1)
	bad.Payload = []byte(`{"duplicate":1,"duplicate":2}`)
	if err := d.Send(context.Background(), bad); !errors.Is(err, wc.ErrEnvelope) {
		t.Fatal(err)
	}
	if err := d.Send(context.Background(), opaque(wc.KindProgress, 2)); err != nil {
		t.Fatal(err)
	}
	e, err := wc.ReadFrame(peer, 4096)
	if err != nil || e.Sequence != 13 {
		t.Fatalf("sequence continuation: %+v %v", e, err)
	}
}

func TestDispatcherActiveWriteCancellationKeepsReceiveAlive(t *testing.T) {
	c, peer := connected(t, Limits{FrameBytes: wc.MaxFrameBytes, Bytes: 64 << 20, Messages: 100})
	if err := c.conn.SetWriteBuffer(1024); err != nil {
		t.Fatal(err)
	}
	d, err := NewDispatcher(context.Background(), c, DispatcherOptions{WriteTimeout: 2 * time.Second})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = d.Close() }()
	request := opaque(wc.KindArtifactWrite, 1)
	request.Payload = []byte(`"` + strings.Repeat("x", 256<<10) + `"`)
	want := string(request.Payload)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	result := make(chan error, 1)
	go func() { result <- d.Send(ctx, request) }()
	if err := peer.SetDeadline(time.Now().Add(3 * time.Second)); err != nil {
		t.Fatal(err)
	}
	var header [4]byte
	if _, err := io.ReadFull(peer, header[:]); err != nil {
		t.Fatal(err)
	}
	cancel()
	if err := <-result; !errors.Is(err, context.Canceled) {
		t.Fatalf("send cancellation: %v", err)
	}
	// The caller may reclaim its payload after Send returns, even though the
	// dispatcher's independent, bounded write has not completed yet.
	for i := 1; i < len(request.Payload)-1; i++ {
		request.Payload[i] = 'y'
	}
	control := message(1)
	control.Kind = wc.KindCancellation
	if err := wc.WriteFrame(peer, control, wc.MaxFrameBytes); err != nil {
		t.Fatal(err)
	}
	if e := dispatchEvent(t, d.Controls()); e.Kind != wc.KindCancellation {
		t.Fatalf("lost control: %+v", e)
	}
	e, err := wc.ReadFrame(io.MultiReader(bytes.NewReader(header[:]), peer), wc.MaxFrameBytes)
	if err != nil || string(e.Payload) != want {
		t.Fatalf("active payload changed: %v", err)
	}
	if err := d.Send(context.Background(), opaque(wc.KindHeartbeat, 2)); err != nil {
		t.Fatal(err)
	}
	e, err = wc.ReadFrame(peer, wc.MaxFrameBytes)
	if err != nil || e.Kind != wc.KindHeartbeat || e.Sequence != 2 {
		t.Fatalf("connection not usable: %+v %v", e, err)
	}
}

func TestDispatcherSequenceExhaustion(t *testing.T) {
	c, peer := connected(t, limits())
	if err := c.Send(bounded(t), message(^uint64(0)-1)); err != nil {
		t.Fatal(err)
	}
	if _, err := wc.ReadFrame(peer, 4096); err != nil {
		t.Fatal(err)
	}
	d, err := NewDispatcher(context.Background(), c, DispatcherOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = d.Close() }()
	if err := d.Send(context.Background(), opaque(wc.KindProgress, 1)); err != nil {
		t.Fatal(err)
	}
	e, err := wc.ReadFrame(peer, 4096)
	if err != nil || e.Sequence != ^uint64(0) {
		t.Fatalf("last sequence: %+v %v", e, err)
	}
	if err := d.Send(context.Background(), opaque(wc.KindProgress, 2)); !errors.Is(err, wc.ErrSequence) {
		t.Fatalf("wrapped sequence: %v", err)
	}
	dispatchDone(t, d)
}

func TestDispatcherTransientInboundBackpressure(t *testing.T) {
	a, b := dispatchPair(t, DispatcherOptions{EventBuffer: 1, EnqueueTimeout: time.Second})
	for i := 0; i < 2; i++ {
		if err := b.Send(context.Background(), opaque(wc.KindArtifactWrite, i)); err != nil {
			t.Fatal(err)
		}
	}
	if err := b.Send(context.Background(), opaque(wc.KindCancellation, 3)); err != nil {
		t.Fatal(err)
	}
	select {
	case <-a.Done():
		t.Fatalf("transient queue pressure closed session: %v", a.Err())
	case <-time.After(20 * time.Millisecond):
	}
	for i := 0; i < 2; i++ {
		e := dispatchEvent(t, a.Events())
		if string(e.Payload) != fmt.Sprintf(`{"number":%d}`, i) {
			t.Fatalf("reordered event %+v", e)
		}
	}
	if e := dispatchEvent(t, a.Controls()); e.Kind != wc.KindCancellation {
		t.Fatal(e)
	}
}

func TestDispatcherReplyBeforePeerClose(t *testing.T) {
	for i := 0; i < 30; i++ {
		a, b := dispatchPair(t, DispatcherOptions{})
		result := make(chan error, 1)
		go func() {
			e, err := a.Request(context.Background(), opaque(wc.KindRoundRequest, 1))
			if err == nil && string(e.Payload) != `{"number":2}` {
				err = fmt.Errorf("wrong reply %s", e.Payload)
			}
			result <- err
		}()
		request := dispatchEvent(t, b.Events())
		if err := b.Reply(context.Background(), request, opaque(wc.KindRoundResponse, 2)); err != nil {
			t.Fatal(err)
		}
		_ = b.Close()
		if err := <-result; err != nil {
			t.Fatalf("lost reply before EOF: %v", err)
		}
		_ = a.Close()
	}
}
