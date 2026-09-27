package enrollment

import (
	"bytes"
	"fmt"
	"net/netip"
	"time"

	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/internal/certificates"
)

type InteractiveEvidence struct {
	Window        string `json:"window"`
	Request       string `json:"request"`
	Pairing       string `json:"pairing"`
	Peer          string `json:"peer"`
	Interface     string `json:"interface"`
	Client        bool   `json:"client"`
	Administrator bool   `json:"administrator"`
}

func interactiveEvidence(e interactiveEntry) InteractiveEvidence {
	return InteractiveEvidence{e.Request.Request.Window, e.Pairing.RequestHash, e.PairingDigest, e.Peer, e.Interface, e.ClientConfirmed, e.AdminConfirmed}
}

func (e InteractiveEvidence) digest() string {
	return digestText("mixlab_interactive_approval_evidence_v1", e)
}

func pairingMatches(q SignedInteractiveRequest, p PairingContext) bool {
	r := q.Request
	_, err := p.Digest()
	return err == nil && p.Fingerprint == r.Fingerprint && p.Audience == r.Audience && p.Purpose == r.Purpose &&
		p.Role == r.Role && p.RequestHash == digestText(interactiveVersion, q) && bytes.Equal(p.ClientNonce, r.Nonce)
}

func (s *Service) validateInteractive(r record, keys map[string]bool) error {
	if len(r.Interactive) > maxEntries {
		return fmt.Errorf("interactive enrollment journal capacity reached")
	}
	seen := map[string]bool{}
	windows := map[string]Window{}
	for _, w := range r.Windows {
		windows[w.ID] = w
		seen[w.ID] = true
	}
	for _, e := range r.Entries {
		seen[e.Invitation.ID] = true
	}
	for _, e := range r.Interactive {
		q := e.Request.Request
		w, ok := windows[q.Window]
		peer, err := netip.ParseAddr(e.Peer)
		if !ok || !certificates.ValidID(e.ID) || seen[e.ID] || e.Request.verify() != nil ||
			q.Cluster != r.Cluster || q.Fingerprint != r.Fingerprint || q.Policy != w.Policy || q.Endpoint != w.Endpoint || q.Audience != w.Audience || !w.allows(q.Purpose) ||
			e.Submitted < w.Created || e.Submitted >= w.Expires || err != nil || peer.IsUnspecified() || peer.IsMulticast() || peer.String() != e.Peer || !validAudience(e.Interface) ||
			!digestOK(e.PairingDigest) || e.Pairing.RequestID != e.ID || !pairingMatches(e.Request, e.Pairing) {
			return fmt.Errorf("invalid interactive request binding")
		}
		seen[e.ID] = true
		switch e.Stage {
		case "pending", "rejected":
			if e.Result != nil {
				return fmt.Errorf("unapproved request contains result")
			}
		case "approved", "issued":
			if e.Result == nil {
				return fmt.Errorf("approved request missing result")
			}
		case "expired":
		default:
			return fmt.Errorf("invalid interactive stage")
		}
		if e.Result != nil || (e.Stage != "expired" && e.Stage != "rejected") {
			if keys[string(q.PublicKey)] {
				return fmt.Errorf("duplicate interactive principal key")
			}
			keys[string(q.PublicKey)] = true
		}
		if e.Result != nil {
			if !e.ClientConfirmed || !e.AdminConfirmed {
				return fmt.Errorf("approval lacks both confirmations")
			}
			if err := e.Result.validate(e.Request, e.Pairing, e.PairingDigest, w, r.Sequence); err != nil {
				return err
			}
			if e.Result.Evidence != interactiveEvidence(e) {
				return fmt.Errorf("approval evidence mismatch")
			}
			if (e.Stage == "approved" && len(e.Result.Chain) != 0) || (e.Stage == "issued" && len(e.Result.Chain) != 3) {
				return fmt.Errorf("invalid interactive issuance outcome")
			}
			if e.Stage == "issued" && e.Result.Delivery.Request != interactiveDeliveryRequest(*e.Result) {
				return fmt.Errorf("interactive delivery receipt mismatch")
			}
		}
	}
	return nil
}

func (r InteractiveResult) validate(q SignedInteractiveRequest, p PairingContext, pairingDigest string, w Window, sequence uint64) error {
	a := r.Approval
	e := r.Evidence
	if q.verify() != nil || !pairingMatches(q, p) || !digestOK(pairingDigest) ||
		e.Window != w.ID || e.Request != p.RequestHash || e.Pairing != pairingDigest || !e.Client || !e.Administrator ||
		a.Version != approvalVersion || a.RequestID != p.RequestID || a.Policy != string(w.Policy) || a.Cluster != w.Cluster ||
		a.Fingerprint != w.Fingerprint || a.Audience != w.Audience || a.Purpose != q.Request.Purpose || a.Role != q.Request.Role ||
		a.RequestHash != p.RequestHash || !certificates.ValidID(a.Principal) || a.EvidenceHash != e.digest() || a.Sequence == 0 || a.Sequence > sequence || a.ExpiresAt != w.Expires ||
		r.Proof.Request != approvalRequest(a) || !digestOK(r.Commit.Digest) || r.Commit.Generation == 0 {
		return fmt.Errorf("invalid interactive approval binding")
	}
	return nil
}

// ValidateInteractiveResult is run by the enrollee before publishing any key
// handle or credential. The pin and locally derived pairing context/digest are
// inputs, not trusted claims taken from the response.
func ValidateInteractiveResult(a trust.Anchor, w Window, q SignedInteractiveRequest, p PairingContext, pairingDigest string, r InteractiveResult, now time.Time) (trust.VerifiedSnapshot, error) {
	var zero trust.VerifiedSnapshot
	if err := w.validate(); err != nil {
		return zero, err
	}
	if w.Closed || now.Unix() < w.Created || now.Unix() >= w.Expires || w.Cluster != a.Cluster() || w.Fingerprint != a.Fingerprint() ||
		q.Request.Window != w.ID || q.Request.Policy != w.Policy || q.Request.Endpoint != w.Endpoint || q.Request.Audience != w.Audience ||
		q.Request.Cluster != w.Cluster || q.Request.Fingerprint != w.Fingerprint || !w.allows(q.Request.Purpose) {
		return zero, fmt.Errorf("result differs from accepted window")
	}
	if err := r.validate(q, p, pairingDigest, w, r.Approval.Sequence); err != nil {
		return zero, err
	}
	v, err := trust.VerifySnapshot(a, r.Snapshot, now)
	if err != nil {
		return zero, err
	}
	if !roleAllowed(v, q.Request.Role) {
		return zero, fmt.Errorf("result role is ineligible")
	}
	// Only the authority uses its own commit journal for historical approval
	// verification. Clients require a fresh signed receipt covering that exact
	// approval, commit and chain, never trust a network-supplied journal entry.
	if err := verifyInteractiveDelivery(a, v, r, now); err != nil {
		return zero, err
	}
	if err := validatePrincipalChain(a, v, q.Request.PrincipalRequest, r.Approval.Principal, r.Chain, now); err != nil {
		return zero, err
	}
	return v, nil
}
