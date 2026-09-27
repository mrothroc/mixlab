package enrollment

import (
	"bytes"
	"context"
	"crypto"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"sync"
	"time"

	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/identity"
	"github.com/mrothroc/mixlab/trust/internal/certificates"
)

var ErrConsumed = errors.New("provisioning invitation already consumed")
var ErrExpired = errors.New("provisioning approval expired; explicit new enrollment required")

const stateVersion = "mixlab_provisioned_enrollment_state_v1"
const stateFile = "provisioned-enrollment.json"
const lockFile = "provisioned-enrollment.lock"
const maxEntries = 256
const maxStateBytes = 16 << 20

type journal interface {
	ReadFileLimit(string, int64) ([]byte, error)
	CompareAndSwap(string, []byte, []byte) error
	WithProcessLock(context.Context, string, func() error) error
}

// Authority is supplied by trusted local composition, never by a request. Root
// and snapshot signing keys are deliberately absent. Signing handles can be
// backed by protected files or Keychain without this context reading key bytes.
type Authority struct {
	Anchor           trust.Anchor
	PrincipalChain   [][]byte
	PrincipalKey     crypto.Signer
	CurrentPrincipal func(time.Time) ([][]byte, crypto.Signer, error)
	Issuer           []byte
	IssuerKey        crypto.Signer
}

type Service struct {
	store journal
	a     Authority
	owner string
	mu    sync.Mutex
	live  map[string]ChannelPort
}

type record struct {
	Version     string             `json:"version"`
	Cluster     string             `json:"cluster"`
	Fingerprint string             `json:"fingerprint"`
	Owner       string             `json:"owner"`
	Sequence    uint64             `json:"sequence"`
	Generation  uint64             `json:"trust_generation"`
	Entries     []entry            `json:"entries"`
	Windows     []Window           `json:"windows,omitempty"`
	Interactive []interactiveEntry `json:"interactive,omitempty"`
	Renewals    []renewalEntry     `json:"renewals,omitempty"`
}

type entry struct {
	Invitation Invitation     `json:"invitation"` // Secret is always nil here.
	Verifier   string         `json:"verifier"`
	Stage      string         `json:"stage"`
	Request    *SignedRequest `json:"request"`
	Result     *Result        `json:"result"`
}

func validAudience(s string) bool {
	if s == "" || len(s) > 256 {
		return false
	}
	for _, c := range s {
		if c < 33 || c > 126 {
			return false
		}
	}
	return true
}

func validEndpoint(s string) bool {
	u, err := url.Parse(s)
	return err == nil && len(s) <= 2048 && u.Scheme == "https" && u.Hostname() != "" && u.User == nil &&
		u.RawQuery == "" && u.Fragment == "" && (u.Path == "" || u.Path == "/") && u.String() == s
}

func configured(p statehome.Path, a Authority, v trust.VerifiedSnapshot, now time.Time) (*Service, error) {
	if p.Kind() != statehome.Authority {
		return nil, fmt.Errorf("enrollment requires authority-owned state")
	}
	if err := p.Validate(); err != nil {
		return nil, err
	}
	s := &Service{store: p, a: a}
	id, err := trust.AuthenticatePrincipal(a.Anchor, v, a.PrincipalChain, now)
	if err != nil || id.Role != trust.Authority {
		return nil, fmt.Errorf("active authority principal required")
	}
	s.owner = id.Principal
	if err := s.checkAuthority(v, now); err != nil {
		return nil, err
	}
	// Freeze public caller-owned slices, but retain the protected signing ports.
	s.a.Issuer = bytes.Clone(a.Issuer)
	s.a.PrincipalChain = make([][]byte, len(a.PrincipalChain))
	for i := range a.PrincipalChain {
		s.a.PrincipalChain[i] = bytes.Clone(a.PrincipalChain[i])
	}
	return s, nil
}

// Initialize is explicit first creation. Missing state on Open is an error,
// never permission to initialize or reset spent invitations.
func Initialize(ctx context.Context, p statehome.Path, a Authority, v trust.VerifiedSnapshot, now time.Time) (*Service, error) {
	s, err := configured(p, a, v, now)
	if err != nil {
		return nil, err
	}
	err = p.WithProcessLock(ctx, lockFile, func() error {
		return s.write(nil, record{Version: stateVersion, Cluster: a.Anchor.Cluster(), Fingerprint: a.Anchor.Fingerprint(),
			Owner: s.owner, Sequence: 1, Generation: v.Generation(), Entries: []entry{}})
	})
	if err != nil {
		return nil, err
	}
	return s, nil
}

func Open(p statehome.Path, a Authority, v trust.VerifiedSnapshot, now time.Time) (*Service, error) {
	s, err := configured(p, a, v, now)
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := s.store.WithProcessLock(ctx, lockFile, func() error {
		_, _, err := s.loadForOperation(v, now)
		return err
	}); err != nil {
		return nil, err
	}
	return s, nil
}

func (s *Service) checkAuthority(v trust.VerifiedSnapshot, now time.Time) error {
	_, _, err := s.principalMaterial(v, now)
	if err != nil {
		return err
	}
	issuer, _, err := certificates.Verify(s.a.Anchor, s.a.Issuer, nil, certificates.Issuer, now)
	if err != nil {
		return err
	}
	if err = certificates.MatchSigner(issuer, s.a.IssuerKey); err != nil {
		return err
	}
	snapshot, err := snapshotValue(v)
	if err != nil {
		return err
	}
	for _, active := range snapshot.Payload.Issuers {
		if bytes.Equal(active, s.a.Issuer) {
			return nil
		}
	}
	return fmt.Errorf("enrollment issuer is not active")
}

func (s *Service) principalMaterial(v trust.VerifiedSnapshot, now time.Time) ([][]byte, crypto.Signer, error) {
	chain, key := s.a.PrincipalChain, s.a.PrincipalKey
	if s.a.CurrentPrincipal != nil {
		var err error
		chain, key, err = s.a.CurrentPrincipal(now)
		if err != nil {
			return nil, nil, err
		}
	}
	id, err := trust.AuthenticatePrincipal(s.a.Anchor, v, chain, now)
	if err != nil || id.Role != trust.Authority || id.Principal != s.owner {
		return nil, nil, fmt.Errorf("authority unavailable or ineligible")
	}
	c, _, err := certificates.Parse(chain[0])
	if err != nil {
		return nil, nil, err
	}
	if err = certificates.MatchSigner(c, key); err != nil {
		return nil, nil, err
	}
	return chain, key, nil
}

func (s *Service) signPrincipal(v trust.VerifiedSnapshot, r trust.SignRequest, now time.Time) (trust.SignedProof, error) {
	chain, key, err := s.principalMaterial(v, now)
	if err != nil {
		return trust.SignedProof{}, err
	}
	return trust.SignPrincipalProof(s.a.Anchor, chain, key, v, r, now)
}

func snapshotValue(v trust.VerifiedSnapshot) (trust.SignedSnapshot, error) {
	b, err := v.Bytes()
	if err != nil {
		return trust.SignedSnapshot{}, err
	}
	var snapshot trust.SignedSnapshot
	err = json.Unmarshal(b, &snapshot)
	return snapshot, err
}

func endpointAllowed(v trust.VerifiedSnapshot, endpoint, audience string, role trust.Role) bool {
	s, err := snapshotValue(v)
	if err != nil || s.Payload.Endpoints.Payload.Audience != audience {
		return false
	}
	eligible := false
	for _, r := range s.Payload.EligibleRoles {
		if r == role {
			eligible = true
		}
	}
	if !eligible {
		return false
	}
	for _, u := range s.Payload.Endpoints.Payload.URLs {
		if u == endpoint {
			return true
		}
	}
	return false
}

func roleAllowed(v trust.VerifiedSnapshot, role trust.Role) bool {
	s, err := snapshotValue(v)
	if err != nil {
		return false
	}
	for _, r := range s.Payload.EligibleRoles {
		if r == role {
			return true
		}
	}
	return false
}

func (s *Service) read() ([]byte, record, error) {
	var r record
	b, err := s.store.ReadFileLimit(stateFile, maxStateBytes)
	if err != nil {
		return nil, r, err
	}
	if err := json.Unmarshal(b, &r); err != nil {
		return nil, r, fmt.Errorf("invalid enrollment journal")
	}
	canonical, err := json.Marshal(r)
	if err != nil || !bytes.Equal(canonical, b) {
		return nil, r, fmt.Errorf("noncanonical enrollment journal")
	}
	return b, r, s.validate(r)
}

func (s *Service) validate(r record) error {
	if r.Version != stateVersion || r.Cluster != s.a.Anchor.Cluster() || r.Fingerprint != s.a.Anchor.Fingerprint() || r.Owner != s.owner ||
		r.Sequence == 0 || r.Generation == 0 || len(r.Entries) > maxEntries {
		return fmt.Errorf("enrollment context mismatch")
	}
	seen, keys := map[string]bool{}, map[string]bool{}
	if len(r.Windows) > maxEntries {
		return fmt.Errorf("enrollment window journal capacity reached")
	}
	for _, w := range r.Windows {
		if err := w.validate(); err != nil {
			return err
		}
		if w.Cluster != r.Cluster || w.Fingerprint != r.Fingerprint || w.AuditPrincipal != r.Owner || seen[w.ID] {
			return fmt.Errorf("invalid enrollment window owner or identity")
		}
		seen[w.ID] = true
	}
	for _, e := range r.Entries {
		i := e.Invitation
		if i.Version != invitationVersion || i.Cluster != r.Cluster || i.Fingerprint != r.Fingerprint || !bytes.Equal(i.Root, s.a.Anchor.DER()) ||
			!certificates.ValidID(i.ID) || seen[i.ID] || len(i.Secret) != 0 || !digestOK(e.Verifier) || i.Wordlist != identity.WordlistVersion ||
			i.UseLimit != 1 || i.Purpose.role() == "" || i.Role != i.Purpose.role() || !validEndpoint(i.Endpoint) || !validAudience(i.Audience) ||
			i.IssuedAt <= 0 || i.ExpiresAt <= i.IssuedAt || i.ExpiresAt-i.IssuedAt > int64(maxInvitationTTL/time.Second) {
			return fmt.Errorf("invalid invitation journal")
		}
		seen[i.ID] = true
		if e.Stage == "available" {
			if e.Request != nil || e.Result != nil {
				return fmt.Errorf("available invitation has issuance state")
			}
			continue
		}
		if (e.Stage != "approved" && e.Stage != "issued") || e.Request == nil || e.Result == nil || e.Request.verify() != nil || !matches(i, e.Request.Request) {
			return fmt.Errorf("invalid enrollment stage/request")
		}
		if keys[string(e.Request.Request.PublicKey)] {
			return fmt.Errorf("duplicate enrolled principal key")
		}
		keys[string(e.Request.Request.PublicKey)] = true
		if err := e.Result.validate(e, r.Sequence); err != nil {
			return err
		}
	}
	if err := s.validateInteractive(r, keys); err != nil {
		return err
	}
	return validateRenewals(r)
}

func (s *Service) write(old []byte, r record) error {
	if err := s.validate(r); err != nil {
		return err
	}
	b, err := json.Marshal(r)
	if err != nil {
		return err
	}
	if len(b) > maxStateBytes {
		return fmt.Errorf("enrollment journal capacity reached")
	}
	return s.store.CompareAndSwap(stateFile, old, b)
}

func (s *Service) loadForOperation(v trust.VerifiedSnapshot, now time.Time) ([]byte, record, error) {
	if err := s.checkAuthority(v, now); err != nil {
		return nil, record{}, err
	}
	b, r, err := s.read()
	if err == nil && v.Generation() < r.Generation {
		err = fmt.Errorf("trust generation rollback")
	}
	if err == nil && r.Sequence == ^uint64(0) {
		err = fmt.Errorf("enrollment sequence exhausted")
	}
	r.Generation = v.Generation()
	if err == nil {
		changed := false
		for i := range r.Interactive {
			e := &r.Interactive[i]
			if e.Stage != "pending" && e.Stage != "approved" {
				continue
			}
			if _, windowErr := activeWindow(r, e.Request.Request.Window, now); windowErr != nil {
				e.Stage = "expired"
				// Preserve an approved result as an audit/key reservation: an
				// interrupted issuer call may have produced a certificate.
				changed = true
			}
		}
		if changed {
			r.Sequence++
			if err = s.write(b, r); err == nil {
				b, r, err = s.read()
			}
		}
	}
	return b, r, err
}

// Invite returns the only copy of the one-use secret. A publication error never
// returns that secret, even if journal durability is uncertain.
func (s *Service) Invite(ctx context.Context, purpose Purpose, endpoint, audience string, ttl time.Duration, v trust.VerifiedSnapshot, now time.Time) (Invitation, error) {
	var result Invitation
	err := s.store.WithProcessLock(ctx, lockFile, func() error {
		old, r, err := s.loadForOperation(v, now)
		if err != nil {
			return err
		}
		if purpose.role() == "" || ttl < time.Second || ttl > maxInvitationTTL || !endpointAllowed(v, endpoint, audience, purpose.role()) {
			return fmt.Errorf("invalid provisioning purpose, endpoint, audience, or TTL")
		}
		if len(r.Entries) >= maxEntries {
			return fmt.Errorf("enrollment journal capacity reached")
		}
		id, err := certificates.NewID()
		if err != nil {
			return err
		}
		i := Invitation{invitationVersion, id, s.a.Anchor.Cluster(), s.a.Anchor.DER(), s.a.Anchor.Fingerprint(), identity.WordlistVersion,
			endpoint, audience, purpose, purpose.role(), make([]byte, 32), now.Unix(), now.Add(ttl).Unix(), 1}
		if _, err := rand.Read(i.Secret); err != nil {
			return err
		}
		defer clear(i.Secret)
		verifier := digestText("mixlab_invitation_verifier_v1", i)
		public := i
		public.Secret = nil
		r.Entries = append(r.Entries, entry{Invitation: public, Verifier: verifier, Stage: "available"})
		r.Sequence++
		if err := s.write(old, r); err != nil {
			return err
		}
		result = i
		result.Secret = bytes.Clone(i.Secret)
		return nil
	})
	return result, err
}

func matches(i Invitation, r Request) bool {
	return i.ID == r.Invitation && i.Cluster == r.Cluster && i.Fingerprint == r.Fingerprint && i.Endpoint == r.Endpoint &&
		i.Audience == r.Audience && i.Purpose == r.Purpose && i.Role == r.Role
}

// Consume durably records one approved request before the issuer signs. A
// duplicate consume always fails, including byte-identical replay. Recovery of
// an uncertain outcome uses the authority-local RecoverApproved operation.
func (s *Service) Consume(ctx context.Context, request SignedRequest, secret []byte, v trust.VerifiedSnapshot, now time.Time) (Result, error) {
	var result Result
	if len(secret) != 32 {
		return result, fmt.Errorf("invalid provisioning proof size")
	}
	if err := request.verify(); err != nil {
		return result, err
	}
	// Freeze request bytes before entering the journal transaction.
	b, err := json.Marshal(request)
	if err != nil || len(b) > 8192 {
		return result, fmt.Errorf("oversized enrollment request")
	}
	var frozen SignedRequest
	if err := json.Unmarshal(b, &frozen); err != nil {
		return result, err
	}
	request = frozen
	if err := request.verify(); err != nil {
		return result, err
	}
	secret = bytes.Clone(secret)
	defer clear(secret)
	err = s.store.WithProcessLock(ctx, lockFile, func() error {
		old, r, err := s.loadForOperation(v, now)
		if err != nil {
			return err
		}
		index := -1
		for n, e := range r.Entries {
			if e.Invitation.ID == request.Request.Invitation {
				index = n
				break
			}
		}
		if index < 0 {
			return fmt.Errorf("unknown enrollment invitation")
		}
		e := &r.Entries[index]
		if e.Stage != "available" {
			return ErrConsumed
		}
		i := e.Invitation
		i.Secret = secret
		if _, err := i.ValidateTarget(request.Request.Endpoint, request.Request.Audience, now); err != nil {
			return err
		}
		got := digestText("mixlab_invitation_verifier_v1", i)
		if subtle.ConstantTimeCompare([]byte(got), []byte(e.Verifier)) != 1 || !matches(i, request.Request) {
			return fmt.Errorf("provisioning proof mismatch")
		}
		if !endpointAllowed(v, i.Endpoint, i.Audience, i.Role) {
			return fmt.Errorf("enrollment endpoint or role no longer eligible")
		}
		if err := s.checkNewKey(r, request.Request.PublicKey, v); err != nil {
			return err
		}
		r.Sequence++
		approval, err := s.approve(i, request, r.Sequence, v, now)
		if err != nil {
			return err
		}
		e.Request, e.Result, e.Stage = &request, &approval, "approved"
		if err := s.write(old, r); err != nil {
			return err
		}
		result, err = s.issueLocked(i.ID, v, now)
		return err
	})
	return result, err
}

func (s *Service) issueCertificate(r PrincipalRequest, principal string, now time.Time) ([]byte, error) {
	if r.Role == trust.Node {
		return certificates.IssueNode(s.a.Anchor, s.a.Issuer, s.a.IssuerKey, principal, ed25519.PublicKey(r.PublicKey), r.EnvelopeKey, now)
	}
	return certificates.Issue(s.a.Anchor, s.a.Issuer, s.a.IssuerKey, certificates.Principal, r.Role, principal, ed25519.PublicKey(r.PublicKey), now)
}
