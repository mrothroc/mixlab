package enrollment

import (
	"bytes"
	"context"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/internal/certificates"
)

// Poll and ConfirmClient require transport-owned live evidence, not merely a
// request ID or knowledge of the publicly displayed phrase.
func (s *Service) Poll(ctx context.Context, id string, channel ChannelPort, v trust.VerifiedSnapshot, now time.Time) (Progress, error) {
	if channel == nil {
		return Progress{}, fmt.Errorf("live request channel required")
	}
	return s.interactiveAction(ctx, id, channel, "", "poll", v, now)
}

func (s *Service) ConfirmClient(ctx context.Context, id, digest string, channel ChannelPort, v trust.VerifiedSnapshot, now time.Time) (Progress, error) {
	if channel == nil {
		return Progress{}, fmt.Errorf("live request channel required")
	}
	return s.interactiveAction(ctx, id, channel, digest, "client", v, now)
}

// Decide is an administrator-local port. It must never be exposed as a remote
// route or authorized by an ordinary TLS principal role.
func (s *Service) Decide(ctx context.Context, id, digest string, approve bool, v trust.VerifiedSnapshot, now time.Time) (Progress, error) {
	action := "reject"
	if approve {
		action = "admin"
	}
	return s.interactiveAction(ctx, id, nil, digest, action, v, now)
}

func (s *Service) interactiveAction(ctx context.Context, id string, channel ChannelPort, digest, action string, v trust.VerifiedSnapshot, now time.Time) (Progress, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out Progress
	err := s.store.WithProcessLock(ctx, lockFile, func() error {
		old, r, err := s.loadForOperation(v, now)
		if err != nil {
			return err
		}
		for i := range r.Interactive {
			e := &r.Interactive[i]
			if e.ID != id {
				continue
			}
			if err := s.liveChannel(*e, channel); err != nil {
				return err
			}
			w, err := activeWindow(r, e.Request.Request.Window, now)
			if err != nil {
				if e.Stage == "pending" || e.Stage == "approved" {
					e.Stage = "expired"
					r.Sequence++
					if writeErr := s.write(old, r); writeErr != nil {
						return writeErr
					}
				}
				delete(s.live, id)
				return err
			}
			if e.Stage == "expired" || e.Stage == "rejected" {
				return fmt.Errorf("enrollment request is terminal")
			}
			if action != "poll" {
				if w.Policy != Verified || !digestOK(digest) || digest != e.PairingDigest {
					return fmt.Errorf("confirmation differs from exact verified request")
				}
				if e.Stage != "pending" {
					return fmt.Errorf("request already approved")
				}
				switch action {
				case "client":
					e.ClientConfirmed = true
				case "admin":
					e.AdminConfirmed = true
				case "reject":
					e.Stage = "rejected"
				default:
					return fmt.Errorf("invalid local enrollment action")
				}
				r.Sequence++
				if err := s.write(old, r); err != nil {
					return err
				}
			}
			if e.Stage == "rejected" {
				out = progress(*e)
				delete(s.live, id)
				return nil
			}
			out, err = s.finishInteractive(id, v, now)
			return err
		}
		return fmt.Errorf("unknown enrollment request")
	})
	if err != nil {
		return Progress{}, err
	}
	return out, nil
}

// Disconnect is called by the transport on EOF/cancellation. It cannot attach
// another connection to a pending request. Issued records remain durable audit
// history, while incomplete requests expire and never issue after reconnect.
func (s *Service) Disconnect(ctx context.Context, channel ChannelPort) error {
	if channel == nil {
		return fmt.Errorf("enrollment channel required")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.store.WithProcessLock(ctx, lockFile, func() error {
		old, r, err := s.read()
		if err != nil {
			return err
		}
		changed := false
		for i := range r.Interactive {
			e := &r.Interactive[i]
			live := s.live[e.ID]
			if live == nil || live.EvidenceID() != channel.EvidenceID() {
				continue
			}
			delete(s.live, e.ID)
			if e.Stage == "pending" || e.Stage == "approved" {
				e.Stage = "expired"
				changed = true
			}
		}
		if !changed {
			return nil
		}
		if r.Sequence == ^uint64(0) {
			return fmt.Errorf("enrollment sequence exhausted")
		}
		r.Sequence++
		return s.write(old, r)
	})
}

// Both the service mutex and process journal lock must be held. Approval is
// durably acknowledged before issuer use; retries cannot allocate another ID.
func (s *Service) finishInteractive(id string, v trust.VerifiedSnapshot, now time.Time) (Progress, error) {
	old, r, err := s.loadForOperation(v, now)
	if err != nil {
		return Progress{}, err
	}
	for i := range r.Interactive {
		e := &r.Interactive[i]
		if e.ID != id {
			continue
		}
		w, err := activeWindow(r, e.Request.Request.Window, now)
		if err != nil {
			return Progress{}, err
		}
		if err := s.liveChannel(*e, nil); err != nil {
			return Progress{}, err
		}
		if !roleAllowed(v, e.Request.Request.Role) {
			return Progress{}, fmt.Errorf("request role no longer eligible")
		}
		if e.Stage == "pending" {
			if !e.ClientConfirmed || !e.AdminConfirmed {
				return progress(*e), nil
			}
			approved := 0
			for _, prior := range r.Interactive {
				if prior.Request.Request.Window == w.ID && prior.Result != nil {
					approved++
				}
			}
			if approved >= w.MaxApprovals {
				return Progress{}, fmt.Errorf("enrollment approval quota exhausted")
			}
			principal, err := certificates.NewID()
			if err != nil {
				return Progress{}, err
			}
			r.Sequence++
			q := e.Request.Request
			evidence := interactiveEvidence(*e)
			a := Approval{approvalVersion, e.ID, string(w.Policy), q.Cluster, q.Fingerprint, q.Audience, q.Purpose, q.Role, principal, e.Pairing.RequestHash, evidence.digest(), r.Sequence, w.Expires}
			proof, err := s.signPrincipal(v, approvalRequest(a), now)
			if err != nil {
				return Progress{}, err
			}
			commit, err := trust.VerifyProof(s.a.Anchor, v, proof, approvalRequest(a), now)
			if err != nil {
				return Progress{}, err
			}
			e.Result = &InteractiveResult{Evidence: evidence, Approval: a, Proof: proof, Commit: commit}
			e.Stage = "approved"
			if err := s.write(old, r); err != nil {
				return Progress{}, err
			}
			return s.finishInteractive(id, v, now)
		}
		if (e.Stage != "approved" && e.Stage != "issued") || e.Result == nil {
			return Progress{}, fmt.Errorf("request is not approved")
		}
		out := *e.Result
		if out.Proof.Evidence.Principal != s.owner {
			return Progress{}, fmt.Errorf("another authority approved request")
		}
		if err := trust.VerifyHistoricalProof(s.a.Anchor, v, out.Proof, approvalRequest(out.Approval), commitJournal{approval: out.Commit}, now); err != nil {
			return Progress{}, err
		}
		if e.Stage == "issued" {
			if err := validatePrincipalChain(s.a.Anchor, v, e.Request.Request.PrincipalRequest, out.Approval.Principal, out.Chain, now); err != nil {
				return Progress{}, err
			}
			if err := s.signInteractiveDelivery(&out, v, now); err != nil {
				return Progress{}, err
			}
			e.Result = &out
			r.Sequence++
			if err := s.write(old, r); err != nil {
				return Progress{}, err
			}
			return progress(*e), nil
		}
		der, err := s.issueCertificate(e.Request.Request.PrincipalRequest, out.Approval.Principal, now)
		if err != nil {
			return Progress{}, err
		}
		out.Chain = [][]byte{der, bytes.Clone(s.a.Issuer), s.a.Anchor.DER()}
		if err := validatePrincipalChain(s.a.Anchor, v, e.Request.Request.PrincipalRequest, out.Approval.Principal, out.Chain, now); err != nil {
			return Progress{}, err
		}
		if err := s.signInteractiveDelivery(&out, v, now); err != nil {
			return Progress{}, err
		}
		e.Result, e.Stage = &out, "issued"
		r.Sequence++
		if err := s.write(old, r); err != nil {
			return Progress{}, err
		}
		return progress(*e), nil
	}
	return Progress{}, fmt.Errorf("unknown enrollment request")
}
