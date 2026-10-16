package learneripc

import (
	"context"
	"fmt"
	"io"
	"sync"
	"time"

	"github.com/mrothroc/mixlab/artifact"
	private "github.com/mrothroc/mixlab/diloco/learner"
	"github.com/mrothroc/mixlab/diloco/run"
	wc "github.com/mrothroc/mixlab/workercontrol"
	"github.com/mrothroc/mixlab/workercontrol/local"
)

type Client struct {
	Session *local.Dispatcher
	mu      sync.Mutex
}

func (c *Client) request(ctx context.Context, q request) (response, error) {
	var out response
	if err := q.Validate(); err != nil {
		return out, err
	}
	e, err := envelope(wc.KindRoundRequest, q)
	if err != nil {
		return out, err
	}
	e, err = c.Session.Request(ctx, e)
	if err != nil {
		return out, err
	}
	if err = check(e, wc.KindRoundResponse); err != nil {
		return out, err
	}
	if err = private.Decode(e.Payload, &out); err != nil {
		return out, err
	}
	if out.Error != "" {
		if out.ErrorCode == "unavailable" {
			return out, private.ErrUnavailable
		}
		return out, fmt.Errorf("learner relay: %s", out.Error)
	}
	return out, nil
}
func (c *Client) Poll(ctx context.Context) (private.RelayResponse, error) {
	return c.call(ctx, request{Operation: "poll"})
}
func (c *Client) call(ctx context.Context, q request) (private.RelayResponse, error) {
	r, err := c.request(ctx, q)
	if err != nil {
		return private.RelayResponse{}, err
	}
	if r.Reply == nil || r.Transfer != "" || len(r.Data) != 0 {
		return private.RelayResponse{}, fmt.Errorf("missing semantic reply")
	}
	return *r.Reply, nil
}
func (c *Client) Start(ctx context.Context, s private.Start) error {
	_, err := c.call(ctx, request{Operation: "call", Call: &private.RelayRequest{Format: "mixlab_learner_relay_v1", Binding: s.Binding, WorkerID: s.WorkerID, Operation: "start", Start: &s}})
	return err
}
func (c *Client) Submit(ctx context.Context, u private.Update) (private.Receipt, error) {
	return c.receipt(ctx, u, "submit")
}
func (c *Client) Receipt(ctx context.Context, u private.Update) (private.Receipt, error) {
	return c.receipt(ctx, u, "receipt")
}
func (c *Client) receipt(ctx context.Context, u private.Update, op string) (private.Receipt, error) {
	r, err := c.call(ctx, request{Operation: "call", Call: &private.RelayRequest{Format: "mixlab_learner_relay_v1", Binding: u.Binding, WorkerID: u.WorkerID, Operation: op, Update: &u}})
	if err != nil {
		return private.Receipt{}, err
	}
	if r.Receipt == nil {
		return private.Receipt{}, fmt.Errorf("missing update receipt")
	}
	return *r.Receipt, nil
}
func (c *Client) Evaluate(ctx context.Context, v run.EvaluationResult) error {
	_, err := c.call(ctx, request{Operation: "call", Call: &private.RelayRequest{Format: "mixlab_learner_relay_v1", Binding: v.Assignment.Binding, WorkerID: v.Assignment.WorkerID, Operation: "evaluate", Evaluation: &v}})
	return err
}
func (c *Client) Read(ctx context.Context, a run.Assignment, b run.Blob, w io.Writer) error {
	return c.read(ctx, artifactRequest(a, b, "read"), w)
}
func (c *Client) ReadEvaluation(ctx context.Context, a run.EvaluationAssignment, w io.Writer) error {
	return c.read(ctx, &private.ArtifactRequest{Evaluation: &a, Blob: a.Model, Operation: "read"}, w)
}
func (c *Client) read(ctx context.Context, q *private.ArtifactRequest, w io.Writer) error {
	b := q.Blob
	c.mu.Lock()
	defer c.mu.Unlock()
	ctx, cancel := context.WithTimeout(ctx, time.Minute)
	defer cancel()
	r, err := c.request(ctx, request{Operation: "read_begin", Artifact: q})
	if err != nil {
		return err
	}
	if r.Transfer == "" {
		return fmt.Errorf("missing artifact transfer")
	}
	reader := &chunkReader{ctx: ctx, c: c, id: r.Transfer, size: b.Ref.Bytes}
	if err = artifact.Copy(ctx, w, reader, b.Ref); err != nil {
		return err
	}
	_, err = c.request(ctx, request{Operation: "read_end", Transfer: r.Transfer, Offset: b.Ref.Bytes})
	return err
}

type chunkReader struct {
	ctx          context.Context
	c            *Client
	id           string
	offset, size uint64
	pending      []byte
}

func (r *chunkReader) Read(b []byte) (int, error) {
	if len(r.pending) == 0 {
		if r.offset == r.size {
			return 0, io.EOF
		}
		v, err := r.c.request(r.ctx, request{Operation: "read_chunk", Transfer: r.id, Offset: r.offset})
		if err != nil {
			return 0, err
		}
		if len(v.Data) == 0 || uint64(len(v.Data)) > r.size-r.offset {
			return 0, fmt.Errorf("invalid artifact chunk")
		}
		r.pending = v.Data
		r.offset += uint64(len(v.Data))
	}
	n := copy(b, r.pending)
	r.pending = r.pending[n:]
	return n, nil
}
func (c *Client) Write(ctx context.Context, a run.Assignment, b run.Blob, reader io.Reader) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	ctx, cancel := context.WithTimeout(ctx, time.Minute)
	defer cancel()
	r, err := c.request(ctx, request{Operation: "write_begin", Artifact: artifactRequest(a, b, "create")})
	if err != nil {
		return err
	}
	if r.Transfer == "" {
		return fmt.Errorf("missing artifact transfer")
	}
	w := &chunkWriter{ctx: ctx, c: c, id: r.Transfer}
	if err = artifact.Copy(ctx, w, reader, b.Ref); err != nil {
		return err
	}
	_, err = c.request(ctx, request{Operation: "write_end", Transfer: r.Transfer, Offset: b.Ref.Bytes})
	return err
}

type chunkWriter struct {
	ctx    context.Context
	c      *Client
	id     string
	offset uint64
}

func (w *chunkWriter) Write(b []byte) (int, error) {
	written := 0
	for len(b) > 0 {
		n := min(len(b), ChunkBytes)
		_, err := w.c.request(w.ctx, request{Operation: "write_chunk", Transfer: w.id, Offset: w.offset, Data: b[:n]})
		if err != nil {
			return written, err
		}
		w.offset += uint64(n)
		written += n
		b = b[n:]
	}
	return written, nil
}
