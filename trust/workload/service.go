package workload

import (
	"bytes"
	"context"
	"crypto"
	"crypto/ed25519"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"time"

	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/internal/certificates"
)

const stateFile = "workload-issuance.json"
const claimFile = "workload-issuance.claim"
const readyFile = "workload-issuance.ready"
const lockFile = "workload-issuance.lock"
const maxEntries = 512
const maxStateBytes = 32 << 20

type journal interface {
	ReadFileLimit(string, int64) ([]byte, error)
	CompareAndSwap(string, []byte, []byte) error
	WithProcessLock(context.Context, string, func() error) error
}

// Authority contains only the issuer-purpose signer. No root, snapshot,
// principal, or worker private key is accessible through this application.
type Authority struct {
	Anchor trust.Anchor
	Issuer []byte
	Key    crypto.Signer
}
type Service struct {
	store journal
	a     Authority
}
type entry struct {
	Grant       Grant               `json:"grant"`
	NodeCommit  trust.AcceptedProof `json:"node_commit"`
	GrantCommit trust.AcceptedProof `json:"grant_commit"`
	IssuedAt    int64               `json:"issued_at"`
	Result      *Result             `json:"result"`
}
type record struct {
	Version    string  `json:"version"`
	Cluster    string  `json:"cluster"`
	IssuerHash string  `json:"issuer_hash"`
	Generation uint64  `json:"generation"`
	Entries    []entry `json:"entries"`
}

func configured(path statehome.Path, a Authority, view trust.VerifiedSnapshot, now time.Time) (*Service, error) {
	if path.Kind() != statehome.Authority {
		return nil, fmt.Errorf("workload issuance requires authority-owned state")
	}
	if err := path.Validate(); err != nil {
		return nil, err
	}
	a.Issuer = bytes.Clone(a.Issuer)
	s := &Service{path, a}
	if err := s.checkAuthority(view, now); err != nil {
		return nil, err
	}
	return s, nil
}

func (s *Service) identity() []byte {
	b, _ := json.Marshal(struct{ Version, Cluster, Issuer string }{Version, s.a.Anchor.Cluster(), digest(s.a.Issuer)})
	return b
}

// Initialize is explicit authority-local setup. A durable claim prevents a
// missing published journal from being mistaken for a fresh issuer history.
func Initialize(ctx context.Context, path statehome.Path, a Authority, view trust.VerifiedSnapshot, now time.Time) (*Service, error) {
	s, err := configured(path, a, view, now)
	if err != nil {
		return nil, err
	}
	err = path.WithProcessLock(ctx, lockFile, func() error {
		if _, err := path.ReadFileLimit(readyFile, 512); err == nil {
			return fmt.Errorf("workload issuer already initialized")
		} else if !errors.Is(err, os.ErrNotExist) {
			return err
		}
		claim, err := path.ReadFileLimit(claimFile, 512)
		switch {
		case errors.Is(err, os.ErrNotExist):
			if err := path.CompareAndSwap(claimFile, nil, s.identity()); err != nil {
				return err
			}
		case err != nil:
			return err
		case !bytes.Equal(claim, s.identity()):
			return fmt.Errorf("workload issuer claim changed")
		}
		r := record{Version, a.Anchor.Cluster(), digest(a.Issuer), view.Generation(), []entry{}}
		b, err := path.ReadFileLimit(stateFile, maxStateBytes)
		switch {
		case errors.Is(err, os.ErrNotExist):
			if err := s.save(nil, r); err != nil {
				return err
			}
		case err != nil:
			return err
		default:
			var saved record
			if err := json.Unmarshal(b, &saved); err != nil {
				return err
			}
			again, err := json.Marshal(saved)
			if err != nil || !bytes.Equal(again, b) || s.validate(saved) != nil || len(saved.Entries) != 0 || saved.Generation > view.Generation() {
				return fmt.Errorf("cannot recover nonempty or invalid unpublished workload issuer")
			}
		}
		return path.CompareAndSwap(readyFile, nil, s.identity())
	})
	return s, err
}

func Open(path statehome.Path, a Authority, view trust.VerifiedSnapshot, now time.Time) (*Service, error) {
	s, err := configured(path, a, view, now)
	if err != nil {
		return nil, err
	}
	_, _, err = s.load()
	return s, err
}

func snapshot(view trust.VerifiedSnapshot) (trust.SignedSnapshot, error) {
	b, err := view.Bytes()
	if err != nil {
		return trust.SignedSnapshot{}, err
	}
	var out trust.SignedSnapshot
	err = json.Unmarshal(b, &out)
	return out, err
}

func (s *Service) checkAuthority(view trust.VerifiedSnapshot, now time.Time) error {
	v, err := snapshot(view)
	if err != nil {
		return err
	}
	if _, err := trust.VerifySnapshot(s.a.Anchor, v, now); err != nil {
		return err
	}
	issuer, _, err := certificates.Verify(s.a.Anchor, s.a.Issuer, nil, certificates.Issuer, now)
	if err != nil {
		return err
	}
	if err := certificates.MatchSigner(issuer, s.a.Key); err != nil {
		return err
	}
	active := false
	for _, ca := range v.Payload.Issuers {
		active = active || bytes.Equal(ca, s.a.Issuer)
	}
	if !active {
		return fmt.Errorf("workload issuer is ineligible")
	}
	return nil
}

func (s *Service) load() ([]byte, record, error) {
	for _, file := range []string{claimFile, readyFile} {
		b, err := s.store.ReadFileLimit(file, 512)
		if err != nil || !bytes.Equal(b, s.identity()) {
			return nil, record{}, fmt.Errorf("workload issuer history missing or changed: %w", err)
		}
	}
	b, err := s.store.ReadFileLimit(stateFile, maxStateBytes)
	if err != nil {
		return nil, record{}, err
	}
	var r record
	if err := json.Unmarshal(b, &r); err != nil {
		return nil, r, err
	}
	canonical, err := json.Marshal(r)
	if err != nil || !bytes.Equal(canonical, b) {
		return nil, r, fmt.Errorf("noncanonical workload issuer history")
	}
	return b, r, s.validate(r)
}

func (s *Service) validate(r record) error {
	if r.Version != Version || r.Cluster != s.a.Anchor.Cluster() || r.IssuerHash != digest(s.a.Issuer) || r.Generation == 0 || len(r.Entries) > maxEntries || r.Entries == nil {
		return fmt.Errorf("invalid workload issuance history")
	}
	jobs, keys, principals := map[string]bool{}, map[string]bool{}, map[string]bool{}
	for _, e := range r.Entries {
		g := e.Grant
		scope := g.Request.Request.Scope
		q, err := g.Request.GrantRequest()
		if err != nil || g.Proof.Request != q || g.Proof.Evidence.Principal != scope.Controller || g.Proof.Evidence.Role != trust.Controller || scope.Cluster != r.Cluster || e.IssuedAt < scope.Created || e.IssuedAt >= scope.AdmitUntil || !hexSize(e.NodeCommit.Digest, 32) || !hexSize(e.GrantCommit.Digest, 32) || e.NodeCommit.Generation == 0 || e.GrantCommit.Generation == 0 || e.NodeCommit.Generation > r.Generation || e.GrantCommit.Generation > r.Generation {
			return fmt.Errorf("invalid workload issuance entry")
		}
		public, _ := g.Request.Request.PublicKey()
		job := scope.Job + "/" + scope.Attempt
		if jobs[job] || keys[string(public)] || principals[scope.Workload] {
			return fmt.Errorf("reused workload job, key or principal")
		}
		jobs[job], keys[string(public)], principals[scope.Workload] = true, true, true
		if e.Result != nil && (e.Result.GrantHash != digest(g) || e.Result.Binding != scope.binding(e.IssuedAt) || len(e.Result.Chain) != 3) {
			return fmt.Errorf("invalid saved workload certificate")
		}
	}
	return nil
}

func (s *Service) save(old []byte, r record) error {
	if err := s.validate(r); err != nil {
		return err
	}
	b, err := json.Marshal(r)
	if err != nil || len(b) > maxStateBytes {
		return fmt.Errorf("workload issuer journal capacity exceeded")
	}
	return s.store.CompareAndSwap(stateFile, old, b)
}

type commitJournal struct{ entry entry }

func (j commitJournal) AcceptedGeneration(digest string) (uint64, bool, error) {
	for _, c := range []trust.AcceptedProof{j.entry.NodeCommit, j.entry.GrantCommit} {
		if c.Digest == digest {
			return c.Generation, true, nil
		}
	}
	return 0, false, nil
}

// Issue is called with the actual mutually authenticated controller chain.
// A signed grant is not a bearer credential. The node's key proof and the
// controller's exact admission grant must both be accepted before using the CA.
func (s *Service) Issue(ctx context.Context, callerChain [][]byte, g Grant, view trust.VerifiedSnapshot, now time.Time) (Result, error) {
	var out Result
	b, err := json.Marshal(g)
	if err != nil || len(b) > 2*trust.MaxTrustBytes+8192 {
		return out, fmt.Errorf("oversized workload grant")
	}
	var frozen Grant
	if err := json.Unmarshal(b, &frozen); err != nil {
		return out, err
	}
	g = frozen
	q, err := g.Request.GrantRequest()
	if err != nil {
		return out, err
	}
	scope := g.Request.Request.Scope
	actor, err := trust.AuthenticatePrincipal(s.a.Anchor, view, callerChain, now)
	if err != nil || actor.Role != trust.Controller || actor.Principal != scope.Controller || scope.Cluster != s.a.Anchor.Cluster() || g.Proof.Evidence.Principal != actor.Principal || now.Unix() < scope.Created || now.Unix() >= scope.AdmitUntil {
		return out, fmt.Errorf("workload admission controller or deadline mismatch")
	}
	// Historical commit evidence preserves an exact retry's signatures, but
	// cannot make a revoked/expired node eligible for credential delivery.
	node, err := trust.AuthenticatePrincipal(s.a.Anchor, view, g.Request.Proof.Evidence.Chain, now)
	if err != nil || node.Role != trust.Node || node.Principal != scope.Node {
		return out, fmt.Errorf("current enrolled node required for workload issuance")
	}
	nodeQ, _ := g.Request.Request.SigningRequest()
	err = s.store.WithProcessLock(ctx, lockFile, func() error {
		if err := s.checkAuthority(view, now); err != nil {
			return err
		}
		current, err := snapshot(view)
		if err != nil {
			return err
		}
		eligible := false
		for _, role := range current.Payload.EligibleRoles {
			eligible = eligible || role == trust.Worker
		}
		if !eligible {
			return fmt.Errorf("workload role is ineligible")
		}
		old, r, err := s.load()
		if err != nil {
			return err
		}
		if view.Generation() < r.Generation {
			return fmt.Errorf("workload trust generation rollback")
		}
		r.Generation = view.Generation()
		index := -1
		for i, e := range r.Entries {
			saved := e.Grant.Request.Request.Scope
			if saved.Job == scope.Job && saved.Attempt == scope.Attempt {
				if digest(g) != digest(e.Grant) {
					return fmt.Errorf("workload admission already bound to another request")
				}
				index = i
				break
			}
		}
		if index < 0 {
			if len(r.Entries) >= maxEntries {
				return fmt.Errorf("workload issuer journal capacity reached")
			}
			nc, err := trust.VerifyProof(s.a.Anchor, view, g.Request.Proof, nodeQ, now)
			if err != nil {
				return err
			}
			gc, err := trust.VerifyProof(s.a.Anchor, view, g.Proof, q, now)
			if err != nil {
				return err
			}
			if err := checkNewKey(s.a.Anchor, view, g, callerChain, now); err != nil {
				return err
			}
			r.Entries = append(r.Entries, entry{Grant: g, NodeCommit: nc, GrantCommit: gc, IssuedAt: now.Unix()})
			index = len(r.Entries) - 1
			// Validate enforces job/key/principal uniqueness before CA side effects.
			if err := s.save(old, r); err != nil {
				return err
			}
			old, r, err = s.load()
			if err != nil {
				return err
			}
		} else {
			e := r.Entries[index]
			if err := trust.VerifyHistoricalProof(s.a.Anchor, view, g.Request.Proof, nodeQ, commitJournal{e}, now); err != nil {
				return err
			}
			if err := trust.VerifyHistoricalProof(s.a.Anchor, view, g.Proof, q, commitJournal{e}, now); err != nil {
				return err
			}
		}
		e := &r.Entries[index]
		// Purpose eligibility can change between accepted intent and a retry,
		// including the actual caller's renewed TLS certificate.
		if err := checkNewKey(s.a.Anchor, view, g, callerChain, now); err != nil {
			return err
		}
		if e.Result != nil {
			out = *e.Result
			out.Snapshot = current
			if err := ValidateResult(s.a.Anchor, g, out, now); err != nil {
				return err
			}
			return s.save(old, r)
		}
		public, _ := g.Request.Request.PublicKey()
		binding := scope.binding(e.IssuedAt)
		der, err := certificates.IssueWorkload(s.a.Anchor, s.a.Issuer, s.a.Key, public, binding, time.Unix(scope.Deadline, 0), time.Unix(e.IssuedAt, 0))
		if err != nil {
			return err
		}
		out = Result{digest(g), binding, [][]byte{der, bytes.Clone(s.a.Issuer), s.a.Anchor.DER()}, current}
		if err := ValidateResult(s.a.Anchor, g, out, now); err != nil {
			return err
		}
		e.Result = &out
		return s.save(old, r)
	})
	if err != nil {
		return Result{}, err
	}
	return out, nil
}

func checkNewKey(a trust.Anchor, view trust.VerifiedSnapshot, g Grant, callerChain [][]byte, now time.Time) error {
	public, err := g.Request.Request.PublicKey()
	if err != nil {
		return err
	}
	nodeEnvelope, err := trust.NodeEnvelopeKey(a, view, g.Request.Proof.Evidence.Chain, g.Request.Request.Scope.Node, now)
	if err != nil {
		return err
	}
	if bytes.Equal(public, nodeEnvelope) {
		return fmt.Errorf("workload reuses node envelope key")
	}
	snap, err := snapshot(view)
	if err != nil {
		return err
	}
	certs := [][]byte{a.DER(), snap.SignerCertificate}
	certs = append(certs, snap.Payload.Issuers...)
	certs = append(certs, g.Request.Proof.Evidence.Chain...)
	certs = append(certs, g.Proof.Evidence.Chain...)
	certs = append(certs, callerChain...)
	for _, der := range certs {
		c, _, err := certificates.Parse(der)
		if err != nil {
			return err
		}
		if bytes.Equal(public, c.PublicKey.(ed25519.PublicKey)) {
			return fmt.Errorf("workload transport key must be purpose-separated")
		}
	}
	return nil
}
