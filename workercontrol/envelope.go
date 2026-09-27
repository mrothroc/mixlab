package workercontrol

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"unicode/utf8"

	"github.com/mrothroc/mixlab/internal/strictjson"
)

const (
	Version                   = "mixlab_worker_control_v1"
	MaxFrameBytes      uint32 = 1 << 20
	MaxIdentifierBytes        = 256
	maxJSONDepth              = 64
)

var (
	ErrFrameSize       = errors.New("workercontrol: invalid frame size")
	ErrEnvelope        = errors.New("workercontrol: invalid envelope")
	ErrVersion         = errors.New("workercontrol: unknown envelope version")
	ErrKind            = errors.New("workercontrol: unknown message kind")
	ErrBinding         = errors.New("workercontrol: session binding mismatch")
	ErrPeer            = errors.New("workercontrol: peer identity mismatch")
	ErrProof           = errors.New("workercontrol: invalid capability proof")
	ErrClosed          = errors.New("workercontrol: session closed")
	ErrUnauthenticated = errors.New("workercontrol: session not authenticated")
	ErrSequence        = errors.New("workercontrol: replay or stale sequence")
	ErrStream          = errors.New("workercontrol: invalid stream")
)

type Kind string

const (
	KindAssignment           Kind = "assignment"
	KindReadiness            Kind = "readiness"
	KindHeartbeat            Kind = "heartbeat"
	KindProgress             Kind = "progress"
	KindCancellation         Kind = "cancellation"
	KindRoundRequest         Kind = "round_request"
	KindRoundResponse        Kind = "round_response"
	KindArtifactRead         Kind = "artifact_read"
	KindArtifactWrite        Kind = "artifact_write"
	KindUpdateSubmission     Kind = "update_submission"
	KindUpdateAcknowledgment Kind = "update_acknowledgment"
	KindTerminalOutcome      Kind = "terminal_outcome"
)

// Envelope routes an opaque context-owned JSON value. Version is independent
// of PayloadVersion. Payload must be present; its semantics (including null)
// and payload kind/version are exclusively the owning context's responsibility.
type Envelope struct {
	Version        string          `json:"version"`
	JobID          string          `json:"job_id"`
	AttemptID      string          `json:"attempt_id"`
	Sequence       uint64          `json:"sequence"`
	CorrelationID  string          `json:"correlation_id"`
	Kind           Kind            `json:"kind"`
	PayloadKind    string          `json:"payload_kind"`
	PayloadVersion uint32          `json:"payload_version"`
	Payload        json.RawMessage `json:"payload"`
}

func validID(s string) bool {
	if len(s) == 0 || len(s) > MaxIdentifierBytes {
		return false
	}
	for _, c := range s {
		if c < 0x21 || c > 0x7e {
			return false
		}
	}
	return true
}

func (e Envelope) Validate() error {
	if e.Version != Version {
		return ErrVersion
	}
	switch e.Kind {
	case KindAssignment, KindReadiness, KindHeartbeat, KindProgress,
		KindCancellation, KindRoundRequest, KindRoundResponse, KindArtifactRead,
		KindArtifactWrite, KindUpdateSubmission, KindUpdateAcknowledgment, KindTerminalOutcome:
	default:
		return ErrKind
	}
	if !validID(e.JobID) || !validID(e.AttemptID) || !validID(e.CorrelationID) ||
		!validID(e.PayloadKind) || e.Sequence == 0 || e.PayloadVersion == 0 {
		return ErrEnvelope
	}
	if len(e.Payload) > int(MaxFrameBytes) {
		return ErrFrameSize
	}
	if len(e.Payload) == 0 || !utf8.Valid(e.Payload) {
		return ErrEnvelope
	}
	body, err := json.Marshal(e)
	if err != nil {
		return ErrEnvelope
	}
	if len(body) > int(MaxFrameBytes) {
		return ErrFrameSize
	}
	return strictJSON(body)
}

// ReadFrame reads a four-byte big-endian length followed by exactly one JSON
// envelope. limit must be in [1, MaxFrameBytes]. Length is checked before
// allocation. On any error the caller must discard the connection; resync is
// unsupported. The caller owns read deadlines. Adjacent frames remain unread.
func ReadFrame(r io.Reader, limit uint32) (Envelope, error) {
	var e Envelope
	if limit == 0 || limit > MaxFrameBytes {
		return e, ErrFrameSize
	}
	var header [4]byte
	if _, err := io.ReadFull(r, header[:]); err != nil {
		return e, err
	}
	n := binary.BigEndian.Uint32(header[:])
	if n == 0 || n > limit {
		return e, ErrFrameSize
	}
	body := make([]byte, int(n))
	if _, err := io.ReadFull(r, body); err != nil {
		return e, err
	}
	if err := strictJSON(body); err != nil {
		return e, err
	}
	// Decode exact keys explicitly: encoding/json's struct matching otherwise
	// accepts case-insensitive aliases and null scalar fields.
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil || len(fields) != 9 {
		return e, ErrEnvelope
	}
	for key, dest := range map[string]any{
		"version": &e.Version, "job_id": &e.JobID, "attempt_id": &e.AttemptID,
		"sequence": &e.Sequence, "correlation_id": &e.CorrelationID,
		"kind": &e.Kind, "payload_kind": &e.PayloadKind, "payload_version": &e.PayloadVersion,
		"payload": &e.Payload,
	} {
		raw, ok := fields[key]
		if !ok || (key != "payload" && bytes.Equal(bytes.TrimSpace(raw), []byte("null"))) {
			return Envelope{}, ErrEnvelope
		}
		if err := json.Unmarshal(raw, dest); err != nil {
			return Envelope{}, fmt.Errorf("%w: %s", ErrEnvelope, key)
		}
	}
	if err := e.Validate(); err != nil {
		return Envelope{}, err
	}
	return e, nil
}

// WriteFrame validates before writing. A partial write/error is fatal to the
// connection. It does not mutate session sequence state or set deadlines.
func WriteFrame(w io.Writer, e Envelope, limit uint32) error {
	if limit == 0 || limit > MaxFrameBytes {
		return ErrFrameSize
	}
	if err := e.Validate(); err != nil {
		return err
	}
	body, err := json.Marshal(e)
	if err != nil {
		return fmt.Errorf("%w: encoding", ErrEnvelope)
	}
	if len(body) > int(limit) {
		return ErrFrameSize
	}
	frame := make([]byte, 4+len(body))
	binary.BigEndian.PutUint32(frame, uint32(len(body)))
	copy(frame[4:], body)
	n, err := w.Write(frame)
	if err == nil && n != len(frame) {
		return io.ErrShortWrite
	}
	return err
}

// Reject ambiguous duplicate keys throughout JSON, including opaque payloads,
// without interpreting their schemas. Nesting and UTF-8 are bounded/strict.
func strictJSON(b []byte) error {
	if err := strictjson.Validate(b, maxJSONDepth); err != nil {
		return ErrEnvelope
	}
	return nil
}
