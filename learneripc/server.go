package learneripc

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"sync"
	"time"

	"github.com/mrothroc/mixlab/artifact"
	private "github.com/mrothroc/mixlab/diloco/learner"
	"github.com/mrothroc/mixlab/diloco/run"
	wc "github.com/mrothroc/mixlab/workercontrol"
)

// Remote is implemented by the agent's authenticated coordinator client.
// Signed grants and credential handles are deliberately absent from this port.
type Remote interface {
	Poll(context.Context) (private.RelayResponse, error)
	Call(context.Context, private.RelayRequest) (private.RelayResponse, error)
	Read(context.Context, run.Assignment, run.Blob, io.Writer) error
	Write(context.Context, run.Assignment, run.Blob, io.Reader) error
}
type evaluationReader interface {
	ReadEvaluation(context.Context, run.EvaluationAssignment, io.Writer) error
}
type Server struct {
	Remote   Remote
	MaxBytes uint64
	// OnError receives local diagnostics before the wire error is sanitized.
	// Only validated operation names and the error are passed, never request
	// payloads, artifact bytes or credentials. It must not reenter this server.
	OnError func(operation string, err error)
	mu      sync.Mutex
	stream  *transfer
}
type transfer struct {
	id       string
	file     *os.File
	request  private.ArtifactRequest
	offset   uint64
	deadline time.Time
}

func (s *Server) Close() { s.mu.Lock(); defer s.mu.Unlock(); s.closeStream() }
func (s *Server) closeStream() {
	if s.stream != nil {
		_ = s.stream.file.Close()
		s.stream = nil
	}
}

// Handle processes one bounded RPC. Host/session owns cancellation and ordering;
// heartbeats use the dispatcher's separate control lane, never this mutex.
func (s *Server) Handle(ctx context.Context, e wc.Envelope) (wc.Envelope, error) {
	if err := check(e, wc.KindRoundRequest); err != nil {
		return wc.Envelope{}, err
	}
	var q request
	if err := private.Decode(e.Payload, &q); err != nil {
		return wc.Envelope{}, err
	}
	if s.Remote == nil || s.MaxBytes == 0 || s.MaxBytes > 1<<30 {
		return wc.Envelope{}, fmt.Errorf("bounded learner relay required")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	ctx, cancel := context.WithTimeout(ctx, time.Minute)
	defer cancel()
	r, err := s.handle(ctx, q)
	if err != nil {
		s.closeStream()
		if s.OnError != nil {
			operation := q.Operation
			if q.Call != nil {
				operation += "/" + q.Call.Operation
			} else if q.Artifact != nil && q.Artifact.Evaluation != nil {
				operation += "/evaluation"
			}
			s.OnError(operation, err)
		}
		// A semantic rejection (notably receipt-not-yet-present) is not a
		// transport failure. Never return remote credential diagnostics.
		r = response{Error: "operation rejected", ErrorCode: "rejected"}
		if errors.Is(err, private.ErrUnavailable) {
			r = response{Error: "coordinator unavailable after bounded retry", ErrorCode: "unavailable"}
		}
	}
	return envelope(wc.KindRoundResponse, r)
}
func (s *Server) handle(ctx context.Context, q request) (response, error) {
	var out response
	if q.Operation == "poll" {
		v, err := s.Remote.Poll(ctx)
		out.Reply = &v
		return out, err
	}
	if q.Operation == "call" {
		v, err := s.Remote.Call(ctx, *q.Call)
		out.Reply = &v
		return out, err
	}
	if q.Operation == "read_begin" || q.Operation == "write_begin" {
		if s.stream != nil {
			return out, fmt.Errorf("artifact stream already active")
		}
		a := *q.Artifact
		if a.Blob.Ref.Bytes > s.MaxBytes {
			return out, fmt.Errorf("artifact exceeds agent transfer budget")
		}
		if a.Operation == "create" && a.Blob.Kind != "delta_f32" {
			return out, fmt.Errorf("only delta uploads supported")
		}
		f, err := os.CreateTemp("", "mixlab-learner-relay-*")
		if err != nil {
			return out, err
		}
		if err = os.Remove(f.Name()); err != nil {
			_ = f.Close()
			return out, err
		}
		var id [16]byte
		if _, err = rand.Read(id[:]); err != nil {
			_ = f.Close()
			return out, err
		}
		s.stream = &transfer{id: hex.EncodeToString(id[:]), file: f, request: a, deadline: time.Now().Add(time.Minute)}
		if a.Operation == "read" {
			w := &limitedWriter{writer: f, remaining: a.Blob.Ref.Bytes}
			if a.Evaluation != nil {
				r, ok := s.Remote.(evaluationReader)
				if !ok {
					return out, fmt.Errorf("evaluation transport unavailable")
				}
				err = r.ReadEvaluation(ctx, *a.Evaluation, w)
			} else {
				err = s.Remote.Read(ctx, a.Assignment, a.Blob, w)
			}
			if err != nil {
				return out, err
			}
			if _, err = f.Seek(0, io.SeekStart); err != nil {
				return out, err
			}
			if err = artifact.Copy(ctx, io.Discard, f, a.Blob.Ref); err != nil {
				return out, err
			}
			if _, err = f.Seek(0, io.SeekStart); err != nil {
				return out, err
			}
		}
		out.Transfer = s.stream.id
		return out, nil
	}
	t := s.stream
	if t == nil || t.id != q.Transfer || t.offset != q.Offset || time.Now().After(t.deadline) {
		return out, fmt.Errorf("stale or out-of-order transfer")
	}
	size := t.request.Blob.Ref.Bytes
	switch q.Operation {
	case "read_chunk":
		if t.request.Operation != "read" || t.offset >= size {
			return out, fmt.Errorf("invalid read chunk")
		}
		out.Data = make([]byte, min(uint64(ChunkBytes), size-t.offset))
		n, err := io.ReadFull(t.file, out.Data)
		t.offset += uint64(n)
		return out, err
	case "write_chunk":
		if t.request.Operation != "create" || uint64(len(q.Data)) > size-t.offset {
			return out, fmt.Errorf("invalid write chunk")
		}
		n, err := t.file.Write(q.Data)
		t.offset += uint64(n)
		if err == nil && n != len(q.Data) {
			err = io.ErrShortWrite
		}
		return out, err
	case "read_end", "write_end":
		if t.offset != size || (q.Operation == "read_end") != (t.request.Operation == "read") {
			return out, fmt.Errorf("incomplete transfer")
		}
		if t.request.Operation == "create" {
			if _, err := t.file.Seek(0, io.SeekStart); err != nil {
				return out, err
			}
			if err := artifact.Copy(ctx, io.Discard, t.file, t.request.Blob.Ref); err != nil {
				return out, err
			}
			if _, err := t.file.Seek(0, io.SeekStart); err != nil {
				return out, err
			}
			if err := s.Remote.Write(ctx, t.request.Assignment, t.request.Blob, t.file); err != nil {
				return out, err
			}
		}
		s.closeStream()
		return out, nil
	}
	return out, fmt.Errorf("invalid stream operation")
}

type limitedWriter struct {
	writer    io.Writer
	remaining uint64
}

func (w *limitedWriter) Write(b []byte) (int, error) {
	if uint64(len(b)) > w.remaining {
		return 0, fmt.Errorf("remote artifact exceeded declared size")
	}
	n, err := w.writer.Write(b)
	w.remaining -= uint64(n)
	return n, err
}
