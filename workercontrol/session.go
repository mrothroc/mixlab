package workercontrol

import (
	"crypto/rand"
	"crypto/subtle"
	"sync"
)

const CapabilityBytes = 32

// PeerIdentity is trusted adapter input, NEVER a decoded message field. PID
// must identify the supervised, still-live child; UID alone is insufficient.
// UID zero is valid. PID zero denotes missing evidence and is rejected.
type PeerIdentity struct {
	PID int
	UID uint32
}

// ValidatePeerIdentity is only a pure evidence-comparison contract. It does
// not query a socket or OS and must not be used with self-reported credentials.
func ValidatePeerIdentity(expected, observed PeerIdentity) error {
	if expected.PID <= 0 || observed.PID <= 0 || expected != observed {
		return ErrPeer
	}
	return nil
}

// Binding holds immutable, already-approved launcher input, not job content.
// AssignmentDigest identifies the exact signed assignment's approved bytes;
// the owner defines canonicalization and verifies the signature before launch.
type Binding struct {
	JobID            string
	AttemptID        string
	BinaryID         string
	BuildID          string
	AssignmentDigest [32]byte
}

func (b Binding) valid() bool {
	return validID(b.JobID) && validID(b.AttemptID) && validID(b.BinaryID) &&
		validID(b.BuildID) && b.AssignmentDigest != [32]byte{}
}

// Validate checks the binding's shape, not executable provenance or approval.
func (b Binding) Validate() error {
	if !b.valid() {
		return ErrBinding
	}
	return nil
}

// Session is an agent-side admission and child-to-agent receive gate. It must
// not be copied. Its zero value is closed. All admission/message failures close
// the gate. Outgoing sequencing and domain idempotency belong to the future
// adapter/owning context; correlation IDs are not deduplicated here.
type Session struct {
	mu            sync.Mutex
	binding       Binding
	peer          PeerIdentity
	capability    [CapabilityBytes]byte
	pending       bool
	authenticated bool
	last          uint64
}

// NewSession generates a fresh CSPRNG capability. The returned secret copy is
// for protected delivery only; the caller must erase/discard it after delivery.
// No reusable secret accessor or injectable insecure randomness is provided.
func NewSession(binding Binding, expected PeerIdentity) (*Session, [CapabilityBytes]byte, error) {
	var capability [CapabilityBytes]byte
	if !binding.valid() {
		return nil, capability, ErrBinding
	}
	if err := ValidatePeerIdentity(expected, expected); err != nil {
		return nil, capability, err
	}
	if _, err := rand.Read(capability[:]); err != nil {
		return nil, [CapabilityBytes]byte{}, err
	}
	s := &Session{binding: binding, peer: expected, capability: capability, pending: true}
	return s, capability, nil
}

// Authenticate consumes the proof on the first attempt, successful or not.
// Call it exactly once, before the first envelope on a protected connection. Supplied binding
// is a claim; observed identity must come independently from the OS adapter.
// Fixed-size proof bytes are compared in constant time (length is public).
func (s *Session) Authenticate(binding Binding, observed PeerIdentity, proof []byte) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.pending {
		s.close()
		return ErrClosed
	}
	var candidate [CapabilityBytes]byte
	copy(candidate[:], proof)
	match := subtle.ConstantTimeCompare(candidate[:], s.capability[:])
	clear(candidate[:])
	s.pending = false
	clear(s.capability[:])
	if len(proof) != CapabilityBytes || match != 1 {
		return ErrProof
	}
	if binding != s.binding {
		return ErrBinding
	}
	if err := ValidatePeerIdentity(s.peer, observed); err != nil {
		return err
	}
	s.authenticated = true
	return nil
}

// Accept validates a received envelope and advances a strictly increasing
// sequence starting at any positive value. Duplicate/stale sequences, including
// conflicting duplicates, are rejected rather than replayed. Gaps are allowed;
// uint64 wraparound cannot pass. Payload semantics must still be validated.
func (s *Session) Accept(e Envelope) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.pending {
		s.close()
		return ErrUnauthenticated
	}
	if !s.authenticated {
		return ErrClosed
	}
	if err := e.Validate(); err != nil {
		s.close()
		return err
	}
	if e.JobID != s.binding.JobID || e.AttemptID != s.binding.AttemptID {
		s.close()
		return ErrBinding
	}
	if e.Sequence <= s.last {
		s.close()
		return ErrSequence
	}
	s.last = e.Sequence
	return nil
}

// Close invalidates unused proof and prevents all future admission/messages.
func (s *Session) Close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.close()
}

func (s *Session) close() {
	clear(s.capability[:])
	s.pending = false
	s.authenticated = false
}
