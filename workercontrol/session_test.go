package workercontrol

import (
	"crypto/sha256"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
)

func testBinding() Binding {
	return Binding{JobID: "job-1", AttemptID: "attempt-1", BinaryID: "mixlab", BuildID: "build-1", AssignmentDigest: sha256.Sum256([]byte("signed assignment"))}
}

var testPeer = PeerIdentity{PID: 123, UID: 501}

func newTestSession(t *testing.T) (*Session, [CapabilityBytes]byte) {
	t.Helper()
	s, secret, err := NewSession(testBinding(), testPeer)
	if err != nil {
		t.Fatal(err)
	}
	return s, secret
}

func authenticatedSession(t *testing.T) *Session {
	t.Helper()
	s, secret := newTestSession(t)
	if err := s.Authenticate(testBinding(), testPeer, secret[:]); err != nil {
		t.Fatal(err)
	}
	if s.capability != [CapabilityBytes]byte{} {
		t.Fatal("retained capability")
	}
	return s
}

func TestPeerIdentityContract(t *testing.T) {
	for _, observed := range []PeerIdentity{{}, {PID: -1, UID: 501}, {PID: 124, UID: 501}, {PID: 123, UID: 502}} {
		if err := ValidatePeerIdentity(testPeer, observed); !errors.Is(err, ErrPeer) {
			t.Fatal(err)
		}
	}
	if err := ValidatePeerIdentity(PeerIdentity{}, PeerIdentity{}); !errors.Is(err, ErrPeer) {
		t.Fatal(err)
	}
	if err := ValidatePeerIdentity(PeerIdentity{PID: 1}, PeerIdentity{PID: 1}); err != nil {
		t.Fatal(err)
	}
	if err := ValidatePeerIdentity(testPeer, testPeer); err != nil {
		t.Fatal(err)
	}
}

func TestCapabilityFreshOneUseAndClose(t *testing.T) {
	s, first := newTestSession(t)
	other, second := newTestSession(t)
	if first == second || first == [CapabilityBytes]byte{} {
		t.Fatal("capability is not fresh")
	}
	if err := s.Authenticate(testBinding(), testPeer, first[:]); err != nil {
		t.Fatal(err)
	}
	if err := s.Accept(testEnvelope()); err != nil {
		t.Fatal(err)
	}
	if err := s.Authenticate(testBinding(), testPeer, first[:]); !errors.Is(err, ErrClosed) {
		t.Fatal(err)
	}
	if err := s.Accept(testEnvelope()); !errors.Is(err, ErrClosed) {
		t.Fatal(err)
	}
	if err := other.Authenticate(testBinding(), testPeer, first[:]); !errors.Is(err, ErrProof) {
		t.Fatal(err)
	}
	if err := other.Authenticate(testBinding(), testPeer, second[:]); !errors.Is(err, ErrClosed) {
		t.Fatal(err)
	}
	for _, active := range []bool{false, true} {
		s, proof := newTestSession(t)
		if active {
			if err := s.Authenticate(testBinding(), testPeer, proof[:]); err != nil {
				t.Fatal(err)
			}
		}
		s.Close()
		s.Close()
		if s.capability != [CapabilityBytes]byte{} {
			t.Fatal("close retained proof")
		}
		if err := s.Authenticate(testBinding(), testPeer, proof[:]); !errors.Is(err, ErrClosed) {
			t.Fatal(err)
		}
		if err := s.Accept(testEnvelope()); !errors.Is(err, ErrClosed) {
			t.Fatal(err)
		}
	}
	var zero Session
	if err := zero.Authenticate(testBinding(), testPeer, first[:]); !errors.Is(err, ErrClosed) {
		t.Fatal(err)
	}
	if err := zero.Accept(testEnvelope()); !errors.Is(err, ErrClosed) {
		t.Fatal(err)
	}
}

func TestEnvelopeBeforeProofInvalidatesSession(t *testing.T) {
	s, proof := newTestSession(t)
	if err := s.Accept(testEnvelope()); !errors.Is(err, ErrUnauthenticated) {
		t.Fatal(err)
	}
	if s.capability != [CapabilityBytes]byte{} {
		t.Fatal("retained capability after protocol violation")
	}
	if err := s.Authenticate(testBinding(), testPeer, proof[:]); !errors.Is(err, ErrClosed) {
		t.Fatal(err)
	}
}

func TestAdmissionRejectsAndConsumesProof(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mutate func(*Binding, *PeerIdentity, *[]byte)
		want   error
	}{
		{"missing proof", func(_ *Binding, _ *PeerIdentity, p *[]byte) { *p = nil }, ErrProof},
		{"short proof", func(_ *Binding, _ *PeerIdentity, p *[]byte) { *p = (*p)[:31] }, ErrProof},
		{"long proof", func(_ *Binding, _ *PeerIdentity, p *[]byte) { *p = append(*p, 0) }, ErrProof},
		{"wrong proof", func(_ *Binding, _ *PeerIdentity, p *[]byte) { (*p)[31] ^= 1 }, ErrProof},
		{"wrong job", func(b *Binding, _ *PeerIdentity, _ *[]byte) { b.JobID = "other" }, ErrBinding},
		{"wrong attempt", func(b *Binding, _ *PeerIdentity, _ *[]byte) { b.AttemptID = "other" }, ErrBinding},
		{"wrong binary", func(b *Binding, _ *PeerIdentity, _ *[]byte) { b.BinaryID = "other" }, ErrBinding},
		{"wrong build", func(b *Binding, _ *PeerIdentity, _ *[]byte) { b.BuildID = "other" }, ErrBinding},
		{"wrong assignment", func(b *Binding, _ *PeerIdentity, _ *[]byte) { b.AssignmentDigest[0] ^= 1 }, ErrBinding},
		{"foreign pid", func(_ *Binding, p *PeerIdentity, _ *[]byte) { p.PID++ }, ErrPeer},
		{"foreign uid", func(_ *Binding, p *PeerIdentity, _ *[]byte) { p.UID++ }, ErrPeer},
		{"missing peer", func(_ *Binding, p *PeerIdentity, _ *[]byte) { *p = PeerIdentity{} }, ErrPeer},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s, secret := newTestSession(t)
			b, peer, proof := testBinding(), testPeer, append([]byte(nil), secret[:]...)
			tc.mutate(&b, &peer, &proof)
			if err := s.Authenticate(b, peer, proof); !errors.Is(err, tc.want) {
				t.Fatal(err)
			}
			if s.capability != [CapabilityBytes]byte{} {
				t.Fatal("failed admission retained proof")
			}
			if err := s.Authenticate(testBinding(), testPeer, secret[:]); !errors.Is(err, ErrClosed) {
				t.Fatal(err)
			}
			if err := s.Accept(testEnvelope()); !errors.Is(err, ErrClosed) {
				t.Fatal(err)
			}
		})
	}
}

func TestSessionBindingAndSequence(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mutate func(*Envelope)
		want   error
	}{
		{"duplicate", func(e *Envelope) { e.Sequence = 2 }, ErrSequence},
		{"stale", func(e *Envelope) { e.Sequence = 1 }, ErrSequence},
		{"conflicting duplicate", func(e *Envelope) { e.Sequence = 2; e.Payload = []byte(`{"different":true}`) }, ErrSequence},
		{"job", func(e *Envelope) { e.JobID = "other" }, ErrBinding},
		{"attempt", func(e *Envelope) { e.AttemptID = "other" }, ErrBinding},
		{"kind", func(e *Envelope) { e.Kind = "unknown" }, ErrKind},
		{"version", func(e *Envelope) { e.Version = "other" }, ErrVersion},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := authenticatedSession(t)
			e := testEnvelope()
			e.Sequence = 2
			if err := s.Accept(e); err != nil {
				t.Fatal(err)
			}
			e.Sequence = 3
			tc.mutate(&e)
			if err := s.Accept(e); !errors.Is(err, tc.want) {
				t.Fatal(err)
			}
			e = testEnvelope()
			e.Sequence = 100
			if err := s.Accept(e); !errors.Is(err, ErrClosed) {
				t.Fatal(err)
			}
		})
	}
	s := authenticatedSession(t)
	e := testEnvelope()
	for _, sequence := range []uint64{1, 2, 9, ^uint64(0)} {
		e.Sequence = sequence
		if err := s.Accept(e); err != nil {
			t.Fatal(err)
		}
	}
	e.Sequence = 1
	if err := s.Accept(e); !errors.Is(err, ErrSequence) {
		t.Fatal(err)
	}
}

func TestInvalidSessionConfiguration(t *testing.T) {
	for _, mutate := range []func(*Binding){
		func(b *Binding) { b.JobID = "" }, func(b *Binding) { b.AttemptID = "" },
		func(b *Binding) { b.BinaryID = "" }, func(b *Binding) { b.BuildID = "" },
		func(b *Binding) { b.AssignmentDigest = [32]byte{} },
	} {
		b := testBinding()
		mutate(&b)
		if s, secret, err := NewSession(b, testPeer); !errors.Is(err, ErrBinding) || s != nil || secret != [CapabilityBytes]byte{} {
			t.Fatal("invalid binding admitted")
		}
	}
	if _, _, err := NewSession(testBinding(), PeerIdentity{}); !errors.Is(err, ErrPeer) {
		t.Fatal(err)
	}
}

func TestSessionConcurrentAdmissionAndReplay(t *testing.T) {
	s, secret := newTestSession(t)
	var wg sync.WaitGroup
	var successes atomic.Int32
	for range 16 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if s.Authenticate(testBinding(), testPeer, secret[:]) == nil {
				successes.Add(1)
			}
		}()
	}
	wg.Wait()
	if successes.Load() != 1 {
		t.Fatalf("admitted %d times", successes.Load())
	}
	s = authenticatedSession(t)
	successes.Store(0)
	for range 16 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if s.Accept(testEnvelope()) == nil {
				successes.Add(1)
			}
		}()
	}
	wg.Wait()
	if successes.Load() != 1 {
		t.Fatalf("accepted %d times", successes.Load())
	}
}
