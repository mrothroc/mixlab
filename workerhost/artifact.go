package workerhost

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"

	"github.com/mrothroc/mixlab/artifact"
	artifactlocal "github.com/mrothroc/mixlab/artifact/local"
	"github.com/mrothroc/mixlab/statehome"
	wc "github.com/mrothroc/mixlab/workercontrol"
	"github.com/mrothroc/mixlab/workerjob"
)

type artifactReader struct {
	receive func() (wc.Envelope, error)
	ref     artifact.Ref
	offset  uint64
	pending []byte
	ended   bool
}

func (r *artifactReader) Read(b []byte) (int, error) {
	if len(b) == 0 {
		return 0, nil
	}
	for len(r.pending) == 0 {
		if r.ended {
			return 0, io.EOF
		}
		e, err := r.receive()
		if err != nil {
			if errors.Is(err, io.EOF) {
				return 0, io.ErrUnexpectedEOF
			}
			return 0, err
		}
		if err := workerjob.CheckPayload(e, e.Kind); err != nil {
			return 0, err
		}
		if e.Kind == wc.KindHeartbeat {
			var heartbeat workerjob.Event
			if err := workerjob.DecodePayload(e.Payload, &heartbeat); err != nil {
				return 0, err
			}
			if heartbeat != (workerjob.Event{}) {
				return 0, fmt.Errorf("invalid stream heartbeat")
			}
			continue
		}
		if e.Kind != wc.KindArtifactWrite {
			return 0, fmt.Errorf("artifact interrupted before end")
		}
		var m workerjob.ArtifactMessage
		if err := workerjob.DecodePayload(e.Payload, &m); err != nil {
			return 0, err
		}
		if err := m.Validate(); err != nil {
			return 0, err
		}
		if m.Ref != r.ref || m.Offset != r.offset {
			return 0, fmt.Errorf("artifact stream identity/offset mismatch")
		}
		switch m.Phase {
		case "chunk":
			r.pending = m.Data
			r.offset += uint64(len(m.Data))
		case "end":
			r.ended = true
		default:
			return 0, fmt.Errorf("duplicate artifact begin")
		}
	}
	n := copy(b, r.pending)
	r.pending = r.pending[n:]
	return n, nil
}

func receiveOutputArtifact(ctx context.Context, path statehome.Path, maxBytes uint64, first wc.Envelope, receive func() (wc.Envelope, error)) (artifact.Ref, error) {
	var m workerjob.ArtifactMessage
	if err := workerjob.CheckPayload(first, wc.KindArtifactWrite); err != nil {
		return artifact.Ref{}, err
	}
	if err := workerjob.DecodePayload(first.Payload, &m); err != nil {
		return artifact.Ref{}, err
	}
	if err := m.Validate(); err != nil {
		return artifact.Ref{}, err
	}
	if maxBytes == 0 || m.Ref.Bytes > maxBytes || m.Phase != "begin" {
		return artifact.Ref{}, fmt.Errorf("unapproved output stream")
	}
	s, err := artifactlocal.Open(path)
	if err != nil {
		return artifact.Ref{}, err
	}
	r := &artifactReader{receive: receive, ref: m.Ref}
	if err := s.Put(ctx, m.Ref, r); err != nil {
		return artifact.Ref{}, err
	}
	b, err := json.Marshal(m.Ref)
	if err != nil {
		return artifact.Ref{}, err
	}
	if err := path.CompareAndSwap(workerjob.OutputReceiptFile, nil, b); err != nil {
		return artifact.Ref{}, err
	}
	return m.Ref, nil
}
