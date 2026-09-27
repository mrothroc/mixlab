package local

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"

	wc "github.com/mrothroc/mixlab/workercontrol"
)

const handshakeLimit = 4096
const ready = "mixlab_worker_ready_v1\n"

var (
	ErrHandshake = errors.New("workercontrol/local: invalid handshake")
	ErrDeadline  = errors.New("workercontrol/local: a live context deadline is required")
	ErrBudget    = errors.New("workercontrol/local: session budget exhausted or invalid")
)

// Metadata is non-secret and canonical JSON. The fixed-size capability follows
// this frame as raw bytes, avoiding JSON/base64 copies of secret material.
type metadata struct {
	Binding wc.Binding      `json:"binding"`
	Agent   wc.PeerIdentity `json:"agent"`
}

func writeHello(w io.Writer, m metadata, proof []byte) error {
	if len(proof) != wc.CapabilityBytes {
		return ErrHandshake
	}
	payload, err := json.Marshal(m)
	if err != nil {
		return ErrHandshake
	}
	e := wc.Envelope{Version: wc.Version, JobID: m.Binding.JobID, AttemptID: m.Binding.AttemptID,
		Sequence: 1, CorrelationID: "session", Kind: wc.KindReadiness,
		PayloadKind: "worker_session_v1", PayloadVersion: 1, Payload: payload}
	if err := wc.WriteFrame(w, e, handshakeLimit); err != nil {
		return err
	}
	return writeAll(w, proof)
}

func readHello(r io.Reader, proof *[wc.CapabilityBytes]byte) (metadata, error) {
	e, err := wc.ReadFrame(r, handshakeLimit)
	if err != nil {
		return metadata{}, err
	}
	var m metadata
	if err := json.Unmarshal(e.Payload, &m); err != nil {
		return m, ErrHandshake
	}
	canonical, err := json.Marshal(m)
	if err != nil || !bytes.Equal(canonical, e.Payload) || m.Binding.Validate() != nil ||
		wc.ValidatePeerIdentity(m.Agent, m.Agent) != nil || e.JobID != m.Binding.JobID ||
		e.AttemptID != m.Binding.AttemptID || e.Sequence != 1 || e.CorrelationID != "session" ||
		e.Kind != wc.KindReadiness || e.PayloadKind != "worker_session_v1" || e.PayloadVersion != 1 {
		return m, ErrHandshake
	}
	_, err = io.ReadFull(r, proof[:])
	return m, err
}

func writeAll(w io.Writer, value []byte) error {
	n, err := w.Write(value)
	if err == nil && n != len(value) {
		return io.ErrShortWrite
	}
	return err
}
