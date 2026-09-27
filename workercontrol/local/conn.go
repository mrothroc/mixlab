package local

import (
	"context"
	"io"
	"net"
	"sync"
	"time"

	wc "github.com/mrothroc/mixlab/workercontrol"
)

// Limits apply independently in each direction, counting complete wire bytes
// and messages after admission. Owners choose budgets for their workload.
type Limits struct {
	FrameBytes uint32
	Bytes      int64
	Messages   uint64
}

func (l Limits) validate() error {
	if l.FrameBytes == 0 || l.FrameBytes > wc.MaxFrameBytes || l.Bytes < 4 || l.Messages == 0 {
		return ErrBudget
	}
	return nil
}

// Conn is authenticated and bound to one attempt. It must not be copied.
type Conn struct {
	conn                        *net.UnixConn
	binding                     wc.Binding
	limits                      Limits
	readMu, writeMu             sync.Mutex
	closeOnce                   sync.Once
	readBytes, writeBytes       int64
	readCount, writeCount       uint64
	readSequence, writeSequence uint64
}

func newConn(conn *net.UnixConn, binding wc.Binding, limits Limits) *Conn {
	return &Conn{conn: conn, binding: binding, limits: limits}
}

func (c *Conn) Binding() wc.Binding { return c.binding }

func (c *Conn) Close() error {
	var err error
	c.closeOnce.Do(func() { err = c.conn.Close() })
	return err
}

func deadline(ctx context.Context) (time.Time, error) {
	if err := ctx.Err(); err != nil {
		return time.Time{}, err
	}
	value, ok := ctx.Deadline()
	if !ok {
		return time.Time{}, ErrDeadline
	}
	return value, nil
}

func checkBinding(e wc.Envelope, b wc.Binding, last uint64) error {
	if e.JobID != b.JobID || e.AttemptID != b.AttemptID {
		return wc.ErrBinding
	}
	if e.Sequence <= last {
		return wc.ErrSequence
	}
	return nil
}

// Send preserves the caller's sequence and correlation ID. Sequence numbers
// must increase; errors are fatal, so callers never retry on this connection.
func (c *Conn) Send(ctx context.Context, e wc.Envelope) (err error) {
	// Cancellation also unblocks a call queued behind the same-direction lock.
	stop := context.AfterFunc(ctx, func() { _ = c.Close() })
	defer stop()
	c.writeMu.Lock()
	defer c.writeMu.Unlock()
	defer func() {
		if err != nil {
			_ = c.Close()
		}
	}()
	d, err := deadline(ctx)
	if err != nil {
		return err
	}
	if err = c.conn.SetWriteDeadline(d); err != nil {
		return err
	}
	if err = checkBinding(e, c.binding, c.writeSequence); err != nil {
		return err
	}
	if c.writeCount >= c.limits.Messages {
		return ErrBudget
	}
	w := &budgetWriter{w: c.conn, remaining: c.limits.Bytes - c.writeBytes}
	if err = wc.WriteFrame(w, e, c.limits.FrameBytes); err != nil {
		return err
	}
	c.writeBytes = c.limits.Bytes - w.remaining
	c.writeCount++
	c.writeSequence = e.Sequence
	return ctx.Err()
}

func (c *Conn) Receive(ctx context.Context) (e wc.Envelope, err error) {
	stop := context.AfterFunc(ctx, func() { _ = c.Close() })
	defer stop()
	c.readMu.Lock()
	defer c.readMu.Unlock()
	defer func() {
		if err != nil {
			_ = c.Close()
		}
	}()
	d, err := deadline(ctx)
	if err != nil {
		return e, err
	}
	if err = c.conn.SetReadDeadline(d); err != nil {
		return e, err
	}
	if c.readCount >= c.limits.Messages || c.readBytes >= c.limits.Bytes {
		return e, ErrBudget
	}
	r := &io.LimitedReader{R: c.conn, N: c.limits.Bytes - c.readBytes}
	e, err = wc.ReadFrame(r, c.limits.FrameBytes)
	c.readBytes = c.limits.Bytes - r.N
	if err != nil {
		return wc.Envelope{}, err
	}
	if err = checkBinding(e, c.binding, c.readSequence); err != nil {
		return wc.Envelope{}, err
	}
	c.readCount++
	c.readSequence = e.Sequence
	if err = ctx.Err(); err != nil {
		return wc.Envelope{}, err
	}
	return e, nil
}

type budgetWriter struct {
	w         io.Writer
	remaining int64
}

func (w *budgetWriter) Write(p []byte) (int, error) {
	if int64(len(p)) > w.remaining {
		return 0, ErrBudget
	}
	n, err := w.w.Write(p)
	w.remaining -= int64(n)
	return n, err
}
