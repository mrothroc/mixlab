package local

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"io"
	"strconv"
	"strings"
	"sync"
	"time"

	wc "github.com/mrothroc/mixlab/workercontrol"
)

var (
	ErrBackpressure      = errors.New("workercontrol/local: dispatcher capacity exhausted")
	ErrDispatcherOptions = errors.New("workercontrol/local: invalid dispatcher options")
	ErrCorrelation       = errors.New("workercontrol/local: reserved dispatcher correlation ID")
)

// DispatcherOptions bounds retained frames and operation lifetimes. Zero fields
// select defaults. Negative values are invalid. Conn's wire budgets still apply.
type DispatcherOptions struct {
	MaxInFlight, EventBuffer, ControlBuffer   int
	SendBuffer, ControlSendBuffer             int
	WriteTimeout, ReadTimeout, RequestTimeout time.Duration
	EnqueueTimeout                            time.Duration // maximum inbound consumer stall
}

func (o DispatcherOptions) normalized() (DispatcherOptions, error) {
	for _, item := range []struct {
		value    *int
		fallback int
	}{
		{&o.MaxInFlight, 16}, {&o.EventBuffer, 64}, {&o.ControlBuffer, 8},
		{&o.SendBuffer, 16}, {&o.ControlSendBuffer, 8},
	} {
		if *item.value < 0 || *item.value > 65536 {
			return o, ErrDispatcherOptions
		}
		if *item.value == 0 {
			*item.value = item.fallback
		}
	}
	for _, item := range []struct {
		value    *time.Duration
		fallback time.Duration
	}{
		{&o.WriteTimeout, 5 * time.Second}, {&o.ReadTimeout, 30 * time.Second},
		{&o.RequestTimeout, 30 * time.Second},
		{&o.EnqueueTimeout, 5 * time.Second},
	} {
		if *item.value < 0 {
			return o, ErrDispatcherOptions
		}
		if *item.value == 0 {
			*item.value = item.fallback
		}
	}
	return o, nil
}

type dispatchSend struct {
	ctx      context.Context
	envelope wc.Envelope
	result   chan error
}

// Dispatcher exclusively owns Conn.Send/Receive after construction. It must not
// be copied. Events and Controls each have one consuming owner; controls contain
// unsolicited heartbeat/cancellation envelopes. Payloads are never interpreted.
//
// Full outbound queues or request slots reject admission with ErrBackpressure.
// Full inbound queues apply backpressure for at most EnqueueTimeout, then fail
// the session with ErrBackpressure. Replies/controls behind a stalled inbound
// frame cannot be read until it is queued; they never wait indefinitely.
// Memory is bounded by configured frame counts times Conn's frame limit, plus
// one active read/write. Consumers own envelopes removed from these queues.
type Dispatcher struct {
	conn                *Conn
	opts                DispatcherOptions
	ctx                 context.Context
	cancel              context.CancelFunc
	done                chan struct{}
	events, controls    chan wc.Envelope
	sends, controlSends chan *dispatchSend
	prefix              string
	sequence            uint64 // writer-owned
	mu                  sync.Mutex
	err                 error
	nextID              uint64
	pending             map[string]chan wc.Envelope
}

// NewDispatcher takes exclusive ownership of c on success. ctx owns the whole
// session, not an individual RPC. Receive uses a fresh ReadTimeout per frame;
// idle/partial-frame timeout is fatal because Conn cannot resume canceled reads.
// Prior synchronous Conn sends are supported; callers must stop all I/O first.
func NewDispatcher(ctx context.Context, c *Conn, options DispatcherOptions) (*Dispatcher, error) {
	o, err := options.normalized()
	if err != nil {
		return nil, err
	}
	if c == nil {
		return nil, wc.ErrClosed
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	var nonce [16]byte
	if _, err := rand.Read(nonce[:]); err != nil {
		return nil, err
	}
	ctx, cancel := context.WithCancel(ctx)
	d := &Dispatcher{conn: c, opts: o, ctx: ctx, cancel: cancel, done: make(chan struct{}),
		events: make(chan wc.Envelope, o.EventBuffer), controls: make(chan wc.Envelope, o.ControlBuffer),
		sends: make(chan *dispatchSend, o.SendBuffer), controlSends: make(chan *dispatchSend, o.ControlSendBuffer),
		prefix: "dispatch-" + hex.EncodeToString(nonce[:]) + "-", pending: make(map[string]chan wc.Envelope)}
	c.writeMu.Lock()
	d.sequence = c.writeSequence
	c.writeMu.Unlock()
	var wg sync.WaitGroup
	wg.Add(2)
	go func() { defer wg.Done(); d.receive() }()
	go func() { defer wg.Done(); d.write() }()
	go func() {
		<-ctx.Done()
		d.fail(ctx.Err())
		wg.Wait()
		// No admissions remain after fail. Release queued payloads on shutdown.
		for len(d.sends) > 0 {
			<-d.sends
		}
		for len(d.controlSends) > 0 {
			<-d.controlSends
		}
		close(d.events)
		close(d.controls)
		close(d.done)
	}()
	return d, nil
}

func (d *Dispatcher) Binding() wc.Binding          { return d.conn.Binding() }
func (d *Dispatcher) Events() <-chan wc.Envelope   { return d.events }
func (d *Dispatcher) Controls() <-chan wc.Envelope { return d.controls }

// Done closes after both I/O owners exit. Buffered inbound events remain drainable.
func (d *Dispatcher) Done() <-chan struct{} { return d.done }
func (d *Dispatcher) Err() error            { d.mu.Lock(); defer d.mu.Unlock(); return d.err }
func (d *Dispatcher) Close() error          { d.fail(wc.ErrClosed); <-d.done; return nil }

func (d *Dispatcher) fail(err error) {
	d.mu.Lock()
	if d.err == nil {
		if cause := d.ctx.Err(); cause != nil {
			err = cause
		}
		d.err = err
		d.pending = nil
	}
	d.mu.Unlock()
	d.cancel()
	_ = d.conn.Close()
}

// Send sends an unsolicited envelope. Empty correlation IDs are generated;
// otherwise the caller's correlation is preserved. Version/binding/sequence are
// dispatcher-owned (empty binding fields are filled, conflicting ones rejected).
// Payload storage is copied before admission. A caller cancellation never closes
// the shared connection: a queued send is skipped, but an active write may finish
// within WriteTimeout. Cancellation therefore has an uncertain delivery outcome.
func (d *Dispatcher) Send(ctx context.Context, e wc.Envelope) error {
	ctx, cancel := context.WithTimeout(ctx, d.opts.WriteTimeout)
	defer cancel()
	op, _, err := d.submit(ctx, e, false)
	if err != nil {
		return err
	}
	return d.awaitSend(ctx, op)
}

// Reply echoes the request correlation without interpreting kind or payload.
func (d *Dispatcher) Reply(ctx context.Context, request, response wc.Envelope) error {
	b := d.Binding()
	if request.JobID != b.JobID || request.AttemptID != b.AttemptID {
		return wc.ErrBinding
	}
	if request.CorrelationID == "" {
		return ErrCorrelation
	}
	response.CorrelationID = request.CorrelationID
	return d.Send(ctx, response)
}

// Request waits for the first envelope echoing a fresh session-unique correlation
// ID. The supplied correlation is replaced; domain idempotency belongs in payload.
// Cancellation/deadline removes the waiter without canceling Receive or emitting
// a domain cancellation. Late/duplicate replies to generated IDs are discarded.
// Streaming messages should use Send/Events with caller-assigned correlation
// IDs, not multiple replies to one Request.
func (d *Dispatcher) Request(ctx context.Context, e wc.Envelope) (wc.Envelope, error) {
	ctx, cancel := context.WithTimeout(ctx, d.opts.RequestTimeout)
	defer cancel()
	op, reply, err := d.submit(ctx, e, true)
	if err != nil {
		return wc.Envelope{}, err
	}
	defer func() { d.mu.Lock(); delete(d.pending, op.envelope.CorrelationID); d.mu.Unlock() }()
	if err := d.awaitSend(ctx, op); err != nil {
		return receivedReplyOrError(reply, err)
	}
	select {
	case e := <-reply:
		return e, nil
	case <-ctx.Done():
		return receivedReplyOrError(reply, ctx.Err())
	case <-d.ctx.Done():
		return receivedReplyOrError(reply, d.sessionError())
	}
}

// Once the reader has delivered a reply, a subsequent EOF must not turn an
// authoritative result back into an uncertain transport failure.
func receivedReplyOrError(reply <-chan wc.Envelope, err error) (wc.Envelope, error) {
	select {
	case e := <-reply:
		return e, nil
	default:
		return wc.Envelope{}, err
	}
}

func (d *Dispatcher) sessionError() error {
	if err := d.Err(); err != nil {
		return err
	}
	return d.ctx.Err()
}

func (d *Dispatcher) awaitSend(ctx context.Context, op *dispatchSend) error {
	select {
	case err := <-op.result:
		return err
	case <-ctx.Done():
		return ctx.Err()
	case <-d.ctx.Done():
		return d.sessionError()
	}
}

func controlKind(kind wc.Kind) bool { return kind == wc.KindHeartbeat || kind == wc.KindCancellation }

func (d *Dispatcher) submit(ctx context.Context, e wc.Envelope, request bool) (*dispatchSend, chan wc.Envelope, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	if d.err != nil {
		return nil, nil, d.err
	}
	if err := d.ctx.Err(); err != nil {
		return nil, nil, err
	}
	queue := d.sends
	if controlKind(e.Kind) {
		queue = d.controlSends
	}
	if len(queue) == cap(queue) || (request && len(d.pending) >= d.opts.MaxInFlight) {
		return nil, nil, ErrBackpressure
	}
	if !request && strings.HasPrefix(e.CorrelationID, d.prefix) {
		return nil, nil, ErrCorrelation
	}
	if request || e.CorrelationID == "" {
		if d.nextID == ^uint64(0) {
			return nil, nil, wc.ErrSequence
		}
		d.nextID++
		e.CorrelationID = d.prefix + strconv.FormatUint(d.nextID, 10)
	}
	b := d.Binding()
	if (e.JobID != "" && e.JobID != b.JobID) || (e.AttemptID != "" && e.AttemptID != b.AttemptID) {
		return nil, nil, wc.ErrBinding
	}
	if e.Version != "" && e.Version != wc.Version {
		return nil, nil, wc.ErrVersion
	}
	e.Version, e.JobID, e.AttemptID, e.Sequence = wc.Version, b.JobID, b.AttemptID, 1
	// Validate against the actual frame limit before retaining a payload. Local
	// construction errors must not poison an otherwise usable connection.
	if err := wc.WriteFrame(io.Discard, e, d.conn.limits.FrameBytes); err != nil {
		return nil, nil, err
	}
	e.Payload = append([]byte(nil), e.Payload...)
	op := &dispatchSend{ctx: ctx, envelope: e, result: make(chan error, 1)}
	var reply chan wc.Envelope
	if request {
		reply = make(chan wc.Envelope, 1)
		d.pending[e.CorrelationID] = reply
	}
	queue <- op // serialized admissions, receiver can only free capacity
	return op, reply, nil
}

func (d *Dispatcher) receive() {
	for {
		ctx, cancel := context.WithTimeout(d.ctx, d.opts.ReadTimeout)
		e, err := d.conn.Receive(ctx)
		cancel()
		if err != nil {
			d.fail(err)
			return
		}
		d.mu.Lock()
		reply := d.pending[e.CorrelationID]
		if reply != nil {
			// A capacity-one mailbox allows the reader to continue if its caller
			// cancels concurrently. Duplicates never replace the first result.
			select {
			case reply <- e:
			default:
			}
			d.mu.Unlock()
			continue
		}
		d.mu.Unlock()
		if strings.HasPrefix(e.CorrelationID, d.prefix) {
			continue
		}
		queue := d.events
		if controlKind(e.Kind) {
			queue = d.controls
		}
		select {
		case queue <- e:
			continue
		default:
		}
		timer := time.NewTimer(d.opts.EnqueueTimeout)
		select {
		case queue <- e:
			timer.Stop()
		case <-d.ctx.Done():
			timer.Stop()
			return
		case <-timer.C:
			d.fail(ErrBackpressure)
			return
		}
	}
}

func (d *Dispatcher) write() {
	for {
		var op *dispatchSend
		// Control frames bypass queued ordinary traffic, but never preempt a
		// frame already on the wire. WriteTimeout bounds that head-of-line delay.
		select {
		case op = <-d.controlSends:
		default:
		}
		if op == nil {
			select {
			case <-d.ctx.Done():
				return
			case op = <-d.controlSends:
			case op = <-d.sends:
			}
		}
		if d.ctx.Err() != nil {
			return
		}
		if err := op.ctx.Err(); err != nil {
			op.result <- err
			continue
		}
		if d.sequence == ^uint64(0) {
			d.fail(wc.ErrSequence)
			return
		}
		d.sequence++
		e := op.envelope
		e.Sequence = d.sequence
		// Detach from per-call cancellation once writing begins: interrupting a
		// partial frame would destroy unrelated requests on this same connection.
		ctx, cancel := context.WithTimeout(d.ctx, d.opts.WriteTimeout)
		err := d.conn.Send(ctx, e)
		cancel()
		op.result <- err
		if err != nil {
			d.fail(err)
			return
		}
	}
}
