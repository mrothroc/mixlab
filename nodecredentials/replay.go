// Package nodecredentials owns node-side credential replay persistence. It is
// not an authorization service or a worker-facing secret access API.
package nodecredentials

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
)

const filename = "credential-replay.json"
const version = "mixlab_credential_replay_v1"

var ErrConsumed = errors.New("credential envelope already reserved; owning job must reconcile")

type record struct {
	Version  string                `json:"version"`
	Binding  trust.EnvelopeBinding `json:"binding"`
	Digest   string                `json:"digest"`
	Reserved bool                  `json:"reserved"`
}
type storage interface {
	ReadFileLimit(string, int64) ([]byte, error)
	CompareAndSwap(string, []byte, []byte) error
}
type ReplayJournal struct {
	path    storage
	binding trust.EnvelopeBinding
	digest  string
}

// InitializeReplay is an explicit NEW admitted-job transition, not a fallback
// from OpenReplay. The owner supplies a dedicated protected job directory and
// the exact approved manifest binding/digest. There is no reset or unreserve.
func InitializeReplay(path statehome.Path, binding trust.EnvelopeBinding, digest string) (*ReplayJournal, error) {
	j, err := configured(path, binding, digest)
	if err != nil {
		return nil, err
	}
	b, err := json.Marshal(record{Version: version, Binding: binding, Digest: digest})
	if err != nil {
		return nil, err
	}
	if len(b) > 4096 {
		return nil, fmt.Errorf("credential binding too large")
	}
	if err := path.CompareAndSwap(filename, nil, b); err != nil {
		return nil, err
	}
	return j, nil
}

func configured(path statehome.Path, binding trust.EnvelopeBinding, digest string) (*ReplayJournal, error) {
	d, err := hex.DecodeString(digest)
	if err != nil || len(d) != 32 || hex.EncodeToString(d) != digest {
		return nil, fmt.Errorf("exact canonical envelope digest required")
	}
	if err := path.Validate(); err != nil {
		return nil, err
	}
	return &ReplayJournal{path, binding, digest}, nil
}

// OpenReplay fails on missing, corrupt or substituted state. A reserved record
// can be reopened for inspection but must never decrypt again after a restart.
func OpenReplay(path statehome.Path, binding trust.EnvelopeBinding, digest string) (*ReplayJournal, error) {
	j, err := configured(path, binding, digest)
	if err != nil {
		return nil, err
	}
	_, _, err = j.read()
	if err != nil {
		return nil, err
	}
	return j, nil
}
func (j *ReplayJournal) read() ([]byte, record, error) {
	var r record
	b, err := j.path.ReadFileLimit(filename, 4096)
	if err != nil {
		return nil, r, err
	}
	if err := json.Unmarshal(b, &r); err != nil {
		return nil, r, err
	}
	again, err := json.Marshal(r)
	if err != nil || !bytes.Equal(b, again) || r.Version != version || r.Binding != j.binding || r.Digest != j.digest {
		return nil, r, fmt.Errorf("credential replay state binding/encoding mismatch")
	}
	return b, r, nil
}

// ReserveEnvelope implements trust.EnvelopeReplayGuard. Trust calls it only
// after authentication. CAS publishes the permanent reservation before HPKE
// opening; a post-publication error denies use and never restores availability.
func (j *ReplayJournal) ReserveEnvelope(binding trust.EnvelopeBinding, digest string) error {
	if binding != j.binding || digest != j.digest {
		return fmt.Errorf("credential differs from admitted job")
	}
	old, r, err := j.read()
	if err != nil {
		return err
	}
	if r.Reserved {
		return ErrConsumed
	}
	r.Reserved = true
	b, err := json.Marshal(r)
	if err != nil {
		return err
	}
	return j.path.CompareAndSwap(filename, old, b)
}
func (j *ReplayJournal) Reserved() (bool, error) { _, r, err := j.read(); return r.Reserved, err }

var _ trust.EnvelopeReplayGuard = (*ReplayJournal)(nil)
