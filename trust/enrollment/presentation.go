package enrollment

import (
	"context"
	"encoding/hex"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/identity"
)

// PendingApproval is public comparison data for an administrator-local UI.
// It carries neither authority nor private key material.
type PendingApproval struct {
	ID, RequestHash, Phrase, Digest, Peer, Interface string
	Purpose                                          Purpose
	Role                                             trust.Role
}

func (s *Service) Pending(ctx context.Context, window string, v trust.VerifiedSnapshot, now time.Time) ([]PendingApproval, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out []PendingApproval
	err := s.store.WithProcessLock(ctx, lockFile, func() error {
		_, r, err := s.loadForOperation(v, now)
		if err != nil {
			return err
		}
		w, err := activeWindow(r, window, now)
		if err != nil {
			return err
		}
		if w.Policy != Verified {
			return fmt.Errorf("only verified policy has pending human approvals")
		}
		for _, e := range r.Interactive {
			if e.Request.Request.Window != window || e.Stage != "pending" || e.AdminConfirmed || s.liveChannel(e, nil) != nil {
				continue
			}
			b, err := hex.DecodeString(e.PairingDigest)
			if err != nil || len(b) != 32 {
				return fmt.Errorf("invalid pairing digest")
			}
			phrase := identity.Confirmation([32]byte(b))
			out = append(out, PendingApproval{e.ID, e.Pairing.RequestHash, phrase, e.PairingDigest, e.Peer, e.Interface, e.Request.Request.Purpose, e.Request.Request.Role})
		}
		return nil
	})
	return out, err
}

// ConfirmPairing derives the presentation locally from the original TLS
// exporter. The server's phrase/digest cannot substitute for this check.
func ConfirmPairing(q SignedInteractiveRequest, p Progress, channel ChannelPort) (string, string, error) {
	if channel == nil || p.ID != p.Pairing.RequestID || !pairingMatches(q, p.Pairing) {
		return "", "", fmt.Errorf("pairing request substitution")
	}
	d, err := p.Pairing.Digest()
	if err != nil {
		return "", "", err
	}
	b, err := channel.Bind(d)
	if err != nil {
		return "", "", err
	}
	defer clear(b)
	phrase, digest, err := PairingPresentation(p.Pairing, b)
	if err != nil {
		return "", "", err
	}
	if digest != p.PairingDigest {
		return "", "", fmt.Errorf("pairing channel substitution")
	}
	return phrase, digest, nil
}

func (w Window) ValidateTarget(a trust.Anchor, endpoint string, policy Policy, purpose Purpose, now time.Time) error {
	if err := w.validate(); err != nil {
		return err
	}
	if w.Closed || now.Unix() < w.Created || now.Unix() >= w.Expires || w.Cluster != a.Cluster() || w.Fingerprint != a.Fingerprint() || w.Endpoint != endpoint || w.Policy != policy || !w.allows(purpose) {
		return fmt.Errorf("enrollment window does not match selected policy, endpoint, root or purpose")
	}
	return nil
}
