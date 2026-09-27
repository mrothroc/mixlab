package nodecredentials

import (
	"bytes"
	"context"
	"crypto"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/mrothroc/mixlab/securekeys"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/keylifecycle"
	"github.com/mrothroc/mixlab/trust/workload"
)

const transportFile = "transport-credential.json"
const transportLock = "transport-credential.lock"
const transportVersion = "mixlab_node_transport_credential_v1"

type transportRecord struct {
	Version         string                  `json:"version"`
	RootFingerprint string                  `json:"root_fingerprint"`
	Scope           workload.Scope          `json:"scope"`
	Backend         string                  `json:"backend"`
	KeyScope        string                  `json:"key_scope"`
	Stage           string                  `json:"stage"`
	KeyClaimed      bool                    `json:"key_claimed"`
	Request         *workload.SignedRequest `json:"request"`
	Grant           *workload.Grant         `json:"grant"`
	Result          *workload.Result        `json:"result"`
}

// Transport owns one job's protected transport key and public credential.
// Its directory is agent-owned and must never be the worker's runtime directory.
// "worker" below is the key's workload role, not the trainer's process owner.
type Transport struct {
	path    statehome.Path
	anchor  trust.Anchor
	scope   workload.Scope
	keys    *securekeys.Store
	manager *keylifecycle.Manager
}

func InitializeTransport(ctx context.Context, path statehome.Path, anchor trust.Anchor, scope workload.Scope, backend string) (*Transport, error) {
	if path.Kind() != statehome.Agent || scope.Cluster != anchor.Cluster() {
		return nil, fmt.Errorf("agent-owned admitted workload context required")
	}
	if err := scope.Validate(); err != nil {
		return nil, err
	}
	backend, err := securekeys.ResolveBackend(backend)
	if err != nil {
		return nil, err
	}
	var nonce [32]byte
	if _, err := rand.Read(nonce[:]); err != nil {
		return nil, err
	}
	r := transportRecord{Version: transportVersion, RootFingerprint: anchor.Fingerprint(), Scope: scope, Backend: backend, KeyScope: hex.EncodeToString(nonce[:]), Stage: "preparing"}
	b, err := json.Marshal(r)
	if err != nil {
		return nil, err
	}
	if err := path.Publish(func(p statehome.Path) error {
		if err := p.CompareAndSwap(transportFile, nil, b); err != nil {
			return err
		}
		keys, err := securekeys.OpenSelected(backend, p, r.KeyScope)
		if err != nil {
			return err
		}
		defer func() { _ = keys.Close() }()
		_, err = keylifecycle.Initialize(ctx, p, keys, "worker")
		return err
	}); err != nil {
		return nil, err
	}
	return OpenTransport(path, anchor, scope)
}

// OpenTransport never creates keys or repairs missing published state.
func OpenTransport(path statehome.Path, anchor trust.Anchor, scope workload.Scope) (*Transport, error) {
	if path.Kind() != statehome.Agent || scope.Cluster != anchor.Cluster() {
		return nil, fmt.Errorf("agent-owned admitted workload context required")
	}
	if err := scope.Validate(); err != nil {
		return nil, err
	}
	c := &Transport{path: path, anchor: anchor, scope: scope}
	_, r, err := c.load()
	if err != nil {
		return nil, err
	}
	c.keys, err = securekeys.OpenSelected(r.Backend, path, r.KeyScope)
	if err != nil {
		return nil, err
	}
	c.manager, err = keylifecycle.Open(path, c.keys, "worker")
	if err != nil {
		_ = c.keys.Close()
		return nil, err
	}
	return c, nil
}

func (c *Transport) Close() error { return c.keys.Close() }

func (c *Transport) load() ([]byte, transportRecord, error) {
	b, err := c.path.ReadFileLimit(transportFile, 4*trust.MaxTrustBytes+16384)
	if err != nil {
		return nil, transportRecord{}, err
	}
	var r transportRecord
	if err := json.Unmarshal(b, &r); err != nil {
		return nil, r, err
	}
	again, err := json.Marshal(r)
	if err != nil || !bytes.Equal(b, again) || r.Version != transportVersion || r.RootFingerprint != c.anchor.Fingerprint() || r.Scope != c.scope || (r.Backend != "file" && r.Backend != "keychain") || len(r.KeyScope) != 64 {
		return nil, r, fmt.Errorf("transport credential context changed")
	}
	keyScope, err := hex.DecodeString(r.KeyScope)
	if err != nil || hex.EncodeToString(keyScope) != r.KeyScope {
		return nil, r, fmt.Errorf("invalid protected transport key scope")
	}
	if r.Request != nil {
		if r.Request.Request.Scope != c.scope {
			return nil, r, fmt.Errorf("transport key request scope changed")
		}
		if _, err := r.Request.GrantRequest(); err != nil {
			return nil, r, err
		}
	}
	if r.Grant != nil && (r.Request == nil || !samePublicValue(r.Grant.Request, *r.Request)) {
		return nil, r, fmt.Errorf("transport grant substituted local request")
	}
	if (r.Grant == nil) != (r.Result == nil) {
		return nil, r, fmt.Errorf("incomplete transport credential publication")
	}
	switch r.Stage {
	case "preparing", "key-intent":
		if r.Request != nil || r.Grant != nil {
			return nil, r, fmt.Errorf("invalid preparing transport credential")
		}
		if r.KeyClaimed != (r.Stage == "key-intent") {
			return nil, r, fmt.Errorf("transport creation claim changed")
		}
	case "requested":
		if r.Request == nil || r.Grant != nil || !r.KeyClaimed {
			return nil, r, fmt.Errorf("invalid requested transport credential")
		}
	case "installed":
		if r.Request == nil || r.Grant == nil || !r.KeyClaimed {
			return nil, r, fmt.Errorf("invalid installed transport credential")
		}
	case "destroying", "destroyed":
	default:
		return nil, r, fmt.Errorf("invalid transport credential stage")
	}
	return b, r, nil
}
func samePublicValue(a, b any) bool {
	x, _ := json.Marshal(a)
	y, _ := json.Marshal(b)
	return bytes.Equal(x, y)
}
func (c *Transport) save(old []byte, r transportRecord) error {
	b, err := json.Marshal(r)
	if err != nil {
		return err
	}
	if len(b) > 4*trust.MaxTrustBytes+16384 {
		return fmt.Errorf("transport credential record too large")
	}
	return c.path.CompareAndSwap(transportFile, old, b)
}

type NodeSigner func(context.Context, trust.SignRequest) (trust.SignedProof, error)

// PrepareRequest completes only the original key intent. A missing key after
// its handle was published is an error; it cannot silently generate a successor.
func (c *Transport) PrepareRequest(ctx context.Context, sign NodeSigner, current trust.VerifiedSnapshot, now time.Time) (workload.SignedRequest, error) {
	var out workload.SignedRequest
	if sign == nil || now.Unix() < c.scope.Created || now.Unix() >= c.scope.AdmitUntil {
		return out, fmt.Errorf("live workload admission and node signer required")
	}
	err := c.path.WithProcessLock(ctx, transportLock, func() error {
		old, r, err := c.load()
		if err != nil {
			return err
		}
		if r.Stage != "preparing" && r.Stage != "key-intent" && r.Stage != "requested" && r.Stage != "installed" {
			return fmt.Errorf("transport credential is terminal")
		}
		var state keylifecycle.Record
		if r.Stage == "preparing" {
			// Claim the one permitted creation before entering the key adapter.
			// An interrupted claim cannot infer a missing intent is a new key.
			r.Stage = "key-intent"
			r.KeyClaimed = true
			if err := c.save(old, r); err != nil {
				return err
			}
			old, r, err = c.load()
			if err != nil {
				return err
			}
			state, err = c.manager.Create(ctx, keylifecycle.Workload)
		} else {
			state, err = c.manager.View(keylifecycle.Workload)
		}
		if err != nil {
			return err
		}
		if state.Stage == "creating" {
			state, err = c.manager.Recover(ctx, keylifecycle.Workload)
			if err != nil {
				return err
			}
		}
		if state.Stage != "active" || state.Active == nil {
			return keylifecycle.ErrRecoveryRequired
		}
		key, err := c.keys.Signer(*state.Active)
		if err != nil {
			return err
		}
		if r.Request != nil {
			public, err := r.Request.Request.PublicKey()
			if err != nil || !samePublicValue(public, key.Public()) {
				return fmt.Errorf("stored transport request differs from protected key")
			}
			out = *r.Request
			return nil
		}
		request, err := workload.NewRequest(c.scope, key)
		if err != nil {
			return err
		}
		q, err := request.SigningRequest()
		if err != nil {
			return err
		}
		proof, err := sign(ctx, q)
		if err != nil {
			return err
		}
		out = workload.SignedRequest{Request: request, Proof: proof}
		if _, err := out.GrantRequest(); err != nil {
			return err
		}
		if _, err := trust.VerifyProof(c.anchor, current, proof, q, now); err != nil {
			return err
		}
		r.Request, r.Stage = &out, "requested"
		return c.save(old, r)
	})
	if err != nil {
		return workload.SignedRequest{}, err
	}
	return out, nil
}

func (c *Transport) Install(ctx context.Context, grant workload.Grant, result workload.Result, current trust.VerifiedSnapshot, now time.Time) error {
	if err := workload.ValidateResult(c.anchor, grant, result, now); err != nil {
		return err
	}
	if err := trust.VerifyWorkload(c.anchor, current, result.Chain, result.Binding, now); err != nil {
		return err
	}
	return c.path.WithProcessLock(ctx, transportLock, func() error {
		old, r, err := c.load()
		if err != nil {
			return err
		}
		if r.Request == nil || !samePublicValue(*r.Request, grant.Request) {
			return fmt.Errorf("workload result does not belong to prepared local key")
		}
		if r.Stage == "installed" {
			if !samePublicValue(*r.Grant, grant) || !samePublicValue(r.Result.Chain, result.Chain) {
				return fmt.Errorf("workload certificate already installed")
			}
			return nil
		}
		if r.Stage != "requested" {
			return fmt.Errorf("transport credential cannot be installed")
		}
		key, err := c.signer()
		if err != nil {
			return err
		}
		public, err := r.Request.Request.PublicKey()
		if err != nil || !samePublicValue(public, key.Public()) {
			return fmt.Errorf("workload installation differs from protected key")
		}
		r.Grant, r.Result, r.Stage = &grant, &result, "installed"
		return c.save(old, r)
	})
}

func (c *Transport) signer() (crypto.Signer, error) {
	r, err := c.manager.View(keylifecycle.Workload)
	if err != nil {
		return nil, err
	}
	if r.Stage != "active" || r.Active == nil {
		return nil, keylifecycle.ErrRecoveryRequired
	}
	return c.keys.Signer(*r.Active)
}

// Identity is consumed by the agent's TLS relay, never sent to the worker.
// Current trust is required on every use; the issuance snapshot is audit data.
func (c *Transport) Identity(current trust.VerifiedSnapshot, now time.Time) (chain [][]byte, signer crypto.Signer, err error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	err = c.path.WithProcessLock(ctx, transportLock, func() error {
		var e error
		chain, signer, e = c.identity(current, now)
		return e
	})
	if err != nil {
		return nil, nil, err
	}
	return chain, signer, nil
}

func (c *Transport) identity(current trust.VerifiedSnapshot, now time.Time) ([][]byte, crypto.Signer, error) {
	_, r, err := c.load()
	if err != nil {
		return nil, nil, err
	}
	if r.Stage != "installed" {
		return nil, nil, fmt.Errorf("transport credential not installed or already terminal")
	}
	// Refresh only the verification view in memory; stored issuance evidence
	// remains immutable. Recheck the complete grant/key/scope binding as well.
	view, err := current.Bytes()
	if err != nil {
		return nil, nil, err
	}
	result := *r.Result
	if err := json.Unmarshal(view, &result.Snapshot); err != nil {
		return nil, nil, err
	}
	if err := workload.ValidateResult(c.anchor, *r.Grant, result, now); err != nil {
		return nil, nil, err
	}
	key, err := c.signer()
	if err != nil {
		return nil, nil, err
	}
	public, err := r.Request.Request.PublicKey()
	if err != nil || !samePublicValue(public, key.Public()) {
		return nil, nil, fmt.Errorf("installed workload key differs from protected handle")
	}
	return r.Result.Chain, &transportSigner{owner: c, public: bytes.Clone(public)}, nil
}

// Signing and destruction share one process lock. A terminal intent invalidates
// previously returned signers even when deletion is interrupted. Private key
// bytes still belong exclusively to the protected storage adapter.
type transportSigner struct {
	owner  *Transport
	public ed25519.PublicKey
}

func (s *transportSigner) Public() crypto.PublicKey { return ed25519.PublicKey(bytes.Clone(s.public)) }

func (s *transportSigner) Sign(random io.Reader, message []byte, opts crypto.SignerOpts) ([]byte, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	var signature []byte
	err := s.owner.path.WithProcessLock(ctx, transportLock, func() error {
		_, r, err := s.owner.load()
		if err != nil {
			return err
		}
		if r.Stage != "installed" {
			return fmt.Errorf("workload signing is no longer active")
		}
		key, err := s.owner.signer()
		if err != nil {
			return err
		}
		if !samePublicValue(key.Public(), s.public) {
			return fmt.Errorf("workload signer changed")
		}
		signature, err = key.Sign(random, message, opts)
		return err
	})
	if err != nil {
		return nil, err
	}
	return signature, nil
}

// Destroy follows relay shutdown and owning-job terminal reconciliation. It
// records terminal intent before touching key storage. Repeated cleanup resumes
// deletion, but neither Open nor PrepareRequest can revive a destroyed key.
func (c *Transport) Destroy(ctx context.Context) error {
	return c.path.WithProcessLock(ctx, transportLock, func() error {
		old, r, err := c.load()
		if err != nil {
			return err
		}
		if r.Stage == "destroyed" {
			return nil
		}
		if r.Stage != "destroying" {
			r.Stage = "destroying"
			if err := c.save(old, r); err != nil {
				return err
			}
			old, r, err = c.load()
			if err != nil {
				return err
			}
		}
		state, err := c.manager.View(keylifecycle.Workload)
		switch {
		case errors.Is(err, os.ErrNotExist):
			if r.KeyClaimed || r.Request != nil {
				return fmt.Errorf("published transport key history missing")
			}
		case err != nil:
			return err
		default:
			switch state.Stage {
			case "active":
				_, err = c.manager.Destroy(ctx, keylifecycle.Workload, state.Active.ID)
			case "creating":
				_, err = c.manager.Abort(ctx, keylifecycle.Workload, state.Candidate.ID)
			case "destroying", "aborting":
				_, err = c.manager.Recover(ctx, keylifecycle.Workload)
			case "destroyed", "aborted":
			default:
				return keylifecycle.ErrRecoveryRequired
			}
			if err != nil {
				return err
			}
		}
		r.Stage = "destroyed"
		return c.save(old, r)
	})
}
