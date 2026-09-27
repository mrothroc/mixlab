package enrollment

import (
	"bytes"
	"context"
	"crypto"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/json"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/internal/certificates"
)

const renewalVersion = "mixlab_principal_renewal_v1"

type RenewalRequest struct {
	Version         string `json:"version"`
	Cluster         string `json:"cluster"`
	Principal       string `json:"principal"`
	CertificateHash string `json:"certificate_hash"`
	Nonce           []byte `json:"nonce"`
}
type SignedRenewalRequest struct {
	Request   RenewalRequest `json:"request"`
	Signature []byte         `json:"signature"`
}
type RenewalResult struct {
	RequestHash string               `json:"request_hash"`
	Chain       [][]byte             `json:"chain"`
	Snapshot    trust.SignedSnapshot `json:"snapshot"`
}
type renewalEntry struct {
	Request  SignedRenewalRequest `json:"request"`
	Original [][]byte             `json:"original"`
	Stage    string               `json:"stage"`
	Result   *RenewalResult       `json:"result"`
}

// NewRenewalRequest proves possession of the existing key; it cannot request
// another role, principal, or replacement public key. Rotation is separate.
func NewRenewalRequest(chain [][]byte, key crypto.Signer) (SignedRenewalRequest, error) {
	if len(chain) != 3 || key == nil {
		return SignedRenewalRequest{}, fmt.Errorf("existing principal chain and signer required")
	}
	c, id, err := certificates.Parse(chain[0])
	if err != nil {
		return SignedRenewalRequest{}, err
	}
	if err := certificates.MatchSigner(c, key); err != nil {
		return SignedRenewalRequest{}, err
	}
	q := RenewalRequest{renewalVersion, id.Cluster, id.Principal, digestText(renewalVersion, chain[0]), make([]byte, 32)}
	if _, err := rand.Read(q.Nonce); err != nil {
		return SignedRenewalRequest{}, err
	}
	d := hash(renewalVersion, q)
	sig, err := key.Sign(rand.Reader, d[:], crypto.Hash(0))
	return SignedRenewalRequest{q, sig}, err
}

func (q SignedRenewalRequest) verify(chain [][]byte) error {
	if len(chain) != 3 || q.Request.Version != renewalVersion || len(q.Request.Nonce) != 32 || bytes.Equal(q.Request.Nonce, make([]byte, 32)) {
		return fmt.Errorf("invalid renewal request")
	}
	c, id, err := certificates.Parse(chain[0])
	if err != nil {
		return err
	}
	if id.Profile != certificates.Principal || id.Role == trust.Worker || q.Request.Cluster != id.Cluster || q.Request.Principal != id.Principal || q.Request.CertificateHash != digestText(renewalVersion, chain[0]) {
		return fmt.Errorf("renewal identity mismatch")
	}
	d := hash(renewalVersion, q.Request)
	if !ed25519.Verify(c.PublicKey.(ed25519.PublicKey), d[:], q.Signature) {
		return fmt.Errorf("invalid renewal proof of possession")
	}
	return nil
}

// ValidateFor exposes proof validation to the protected local retry journal.
func (q SignedRenewalRequest) ValidateFor(chain [][]byte) error { return q.verify(chain) }

func ValidateRenewalResult(a trust.Anchor, original [][]byte, q SignedRenewalRequest, r RenewalResult, now time.Time) error {
	if err := q.verify(original); err != nil {
		return err
	}
	if r.RequestHash != digestText(renewalVersion, q) {
		return fmt.Errorf("renewal response request mismatch")
	}
	v, err := trust.VerifySnapshot(a, r.Snapshot, now)
	if err != nil {
		return err
	}
	id, err := trust.AuthenticatePrincipal(a, v, original, now)
	if err != nil {
		return err
	}
	c, profile, err := certificates.Parse(original[0])
	if err != nil {
		return err
	}
	return validatePrincipalChain(a, v, PrincipalRequest{Role: id.Role, PublicKey: c.PublicKey.(ed25519.PublicKey), EnvelopeKey: profile.EnvelopeKey}, id.Principal, r.Chain, now)
}

func validateRenewals(r record) error {
	if len(r.Renewals) > maxEntries {
		return fmt.Errorf("renewal journal capacity reached")
	}
	seen := map[string]bool{}
	for _, e := range r.Renewals {
		h := digestText(renewalVersion, e.Request)
		if e.Request.verify(e.Original) != nil || e.Request.Request.Cluster != r.Cluster || seen[h] {
			return fmt.Errorf("invalid renewal journal request")
		}
		seen[h] = true
		if (e.Stage == "pending" && e.Result != nil) || (e.Stage == "issued" && (e.Result == nil || e.Result.RequestHash != h || len(e.Result.Chain) != 3)) || (e.Stage != "pending" && e.Stage != "issued") {
			return fmt.Errorf("invalid renewal journal outcome")
		}
	}
	return nil
}

// Renew authenticates the chain supplied by the mutual-TLS adapter again at
// operation time. A valid old signature never permits renewal after expiry or
// revocation. Exact retries recover the committed certificate, not a new key.
func (s *Service) Renew(ctx context.Context, authenticatedChain [][]byte, q SignedRenewalRequest, v trust.VerifiedSnapshot, now time.Time) (RenewalResult, error) {
	var out RenewalResult
	b, err := json.Marshal(struct {
		Chain   [][]byte
		Request SignedRenewalRequest
	}{authenticatedChain, q})
	if err != nil || len(b) > 64<<10 {
		return out, fmt.Errorf("oversized renewal request")
	}
	var frozen struct {
		Chain   [][]byte
		Request SignedRenewalRequest
	}
	if err := json.Unmarshal(b, &frozen); err != nil {
		return out, err
	}
	authenticatedChain, q = frozen.Chain, frozen.Request
	if err := q.verify(authenticatedChain); err != nil {
		return out, err
	}
	err = s.store.WithProcessLock(ctx, lockFile, func() error {
		old, r, err := s.loadForOperation(v, now)
		if err != nil {
			return err
		}
		if _, err := trust.AuthenticatePrincipal(s.a.Anchor, v, authenticatedChain, now); err != nil {
			return err
		}
		digest := digestText(renewalVersion, q)
		for _, e := range r.Renewals {
			if digestText(renewalVersion, e.Request) == digest {
				out, err = s.issueRenewal(digest, v, now)
				return err
			}
		}
		if len(r.Renewals) >= maxEntries {
			return fmt.Errorf("renewal journal capacity reached")
		}
		// A certificate can have only one successor; another nonce cannot mint
		// unbounded independent successors from the same authenticated leaf.
		for _, e := range r.Renewals {
			if e.Request.Request.CertificateHash == q.Request.CertificateHash {
				return fmt.Errorf("certificate renewal already pending or issued; retry original request")
			}
		}
		r.Renewals = append(r.Renewals, renewalEntry{Request: q, Original: authenticatedChain, Stage: "pending"})
		r.Sequence++
		if err := s.write(old, r); err != nil {
			return err
		}
		out, err = s.issueRenewal(digest, v, now)
		return err
	})
	if err != nil {
		return RenewalResult{}, err
	}
	return out, nil
}

func (s *Service) issueRenewal(digest string, v trust.VerifiedSnapshot, now time.Time) (RenewalResult, error) {
	old, r, err := s.loadForOperation(v, now)
	if err != nil {
		return RenewalResult{}, err
	}
	for i := range r.Renewals {
		e := &r.Renewals[i]
		if digestText(renewalVersion, e.Request) != digest {
			continue
		}
		identity, err := trust.AuthenticatePrincipal(s.a.Anchor, v, e.Original, now)
		if err != nil {
			return RenewalResult{}, err
		}
		c, id, err := certificates.Parse(e.Original[0])
		if err != nil {
			return RenewalResult{}, err
		}
		p := PrincipalRequest{Role: identity.Role, PublicKey: c.PublicKey.(ed25519.PublicKey), EnvelopeKey: id.EnvelopeKey}
		if e.Stage == "issued" {
			out := *e.Result
			if err := validatePrincipalChain(s.a.Anchor, v, p, identity.Principal, out.Chain, now); err != nil {
				return RenewalResult{}, err
			}
			out.Snapshot, err = snapshotValue(v)
			return out, err
		}
		der, err := s.issueCertificate(p, identity.Principal, now)
		if err != nil {
			return RenewalResult{}, err
		}
		out := RenewalResult{RequestHash: digest, Chain: [][]byte{der, bytes.Clone(s.a.Issuer), s.a.Anchor.DER()}}
		out.Snapshot, err = snapshotValue(v)
		if err != nil {
			return RenewalResult{}, err
		}
		if err := validatePrincipalChain(s.a.Anchor, v, p, identity.Principal, out.Chain, now); err != nil {
			return RenewalResult{}, err
		}
		e.Result, e.Stage = &out, "issued"
		r.Sequence++
		if err := s.write(old, r); err != nil {
			return RenewalResult{}, err
		}
		return out, nil
	}
	return RenewalResult{}, fmt.Errorf("unknown renewal")
}
