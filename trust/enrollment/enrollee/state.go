// Package enrollee owns local provisional keys and credential publication.
// Network, discovery, and human presentation are supplied by composition.
package enrollee

import (
	"bytes"
	"context"
	"crypto"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/mrothroc/mixlab/securekeys"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/enrollment"
	"github.com/mrothroc/mixlab/trust/keylifecycle"
	"github.com/mrothroc/mixlab/trust/principal"
)

const version = "mixlab_enrollee_v1"
const filename = "enrollee.json"
const operationLock = "enrollee-operation.lock"

type intent struct {
	Version     string     `json:"version"`
	Root        []byte     `json:"root"`
	Fingerprint string     `json:"fingerprint"`
	Role        trust.Role `json:"role"`
	Backend     string     `json:"backend"`
	Scope       string     `json:"scope"`
	Final       string     `json:"final"`
}
type Client struct {
	mu           sync.Mutex
	stage, final statehome.Path
	root         trust.Anchor
	keys         *securekeys.Store
	manager      *keylifecycle.Manager
	intent       intent
	published    bool
}

// Prepare creates a new private staging context after the selected policy has
// accepted the root. A nonempty destination never gets overwritten, and key
// creation never runs before its lifecycle intent has been committed.
func Prepare(ctx context.Context, stage, final statehome.Path, root trust.Anchor, role trust.Role, backend string, now time.Time) (*Client, error) {
	if stage.Kind() != statehome.Enrollment || final.Kind() != statehome.Principal || stage.Dir() == final.Dir() {
		return nil, fmt.Errorf("distinct enrollment staging and principal destination required")
	}
	if role != trust.Node && role != trust.Controller && role != trust.Coordinator {
		return nil, fmt.Errorf("unsupported enrollee role")
	}
	for _, p := range []statehome.Path{stage, final} {
		if _, err := os.Lstat(p.Dir()); !errors.Is(err, os.ErrNotExist) {
			return nil, fmt.Errorf("enrollment destination must be absent: %s", p.Dir())
		}
	}
	if nested(stage.Dir(), final.Dir()) {
		return nil, fmt.Errorf("overlapping enrollment paths")
	}
	if nested(final.Dir(), stage.Dir()) {
		return nil, fmt.Errorf("overlapping enrollment paths")
	}
	backend, err := securekeys.ResolveBackend(backend)
	if err != nil {
		return nil, err
	}
	var scope [32]byte
	if _, err := rand.Read(scope[:]); err != nil {
		return nil, err
	}
	i := intent{version, root.DER(), root.Fingerprint(), role, backend, hex.EncodeToString(scope[:]), final.Dir()}
	b, err := json.Marshal(i)
	if err != nil {
		return nil, err
	}
	if err := stage.Publish(func(p statehome.Path) error {
		if err := p.CompareAndSwap(filename, nil, b); err != nil {
			return err
		}
		keys, err := securekeys.OpenSelected(backend, p, i.Scope)
		if err != nil {
			return err
		}
		defer func() { _ = keys.Close() }()
		_, err = keylifecycle.Initialize(ctx, p, keys, string(role))
		return err
	}); err != nil {
		return nil, err
	}
	c, err := Open(stage, final, root, now)
	if err != nil {
		return nil, err
	}
	if _, err := c.manager.Create(ctx, keylifecycle.Principal); err != nil {
		return c, err
	}
	if role == trust.Node {
		if _, err := c.manager.Create(ctx, keylifecycle.NodeEnvelope); err != nil {
			return c, err
		}
	}
	return c, nil
}

func nested(parent, child string) bool {
	rel, err := filepath.Rel(parent, child)
	return err == nil && !filepath.IsAbs(rel) && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
}

// Open only resumes an explicit staging context; it never generates missing
// keys or adopts a root/target from the staged bytes without the caller's pin.
func Open(stage, final statehome.Path, root trust.Anchor, now time.Time) (*Client, error) {
	if stage.Kind() != statehome.Enrollment || final.Kind() != statehome.Principal {
		return nil, fmt.Errorf("invalid enrollee contexts")
	}
	b, err := stage.ReadFileLimit(filename, 32<<10)
	if err != nil {
		return nil, err
	}
	var i intent
	if err := json.Unmarshal(b, &i); err != nil {
		return nil, err
	}
	again, err := json.Marshal(i)
	if err != nil || !bytes.Equal(b, again) || i.Version != version || i.Final != final.Dir() || i.Fingerprint != root.Fingerprint() || !bytes.Equal(i.Root, root.DER()) {
		return nil, fmt.Errorf("staging identity/target mismatch")
	}
	if _, err := trust.PinRoot(i.Root, i.Fingerprint, now); err != nil {
		return nil, err
	}
	if i.Role != trust.Node && i.Role != trust.Controller && i.Role != trust.Coordinator {
		return nil, fmt.Errorf("invalid enrollee role")
	}
	keys, err := securekeys.OpenSelected(i.Backend, stage, i.Scope)
	if err != nil {
		return nil, err
	}
	m, err := keylifecycle.Open(stage, keys, string(i.Role))
	if err != nil {
		_ = keys.Close()
		return nil, err
	}
	return &Client{stage: stage, final: final, root: root, keys: keys, manager: m, intent: i}, nil
}

func (c *Client) Close() error { c.mu.Lock(); defer c.mu.Unlock(); return c.keys.Close() }

func (c *Client) Keys() (crypto.Signer, []byte, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	r, err := c.manager.View(keylifecycle.Principal)
	if err != nil {
		return nil, nil, err
	}
	if r.Stage != "active" || r.Active == nil {
		return nil, nil, keylifecycle.ErrRecoveryRequired
	}
	k, err := c.keys.Signer(*r.Active)
	if err != nil {
		return nil, nil, err
	}
	var envelope []byte
	if c.intent.Role == trust.Node {
		r, err := c.manager.View(keylifecycle.NodeEnvelope)
		if err != nil {
			return nil, nil, err
		}
		if r.Stage != "active" || r.Active == nil {
			return nil, nil, keylifecycle.ErrRecoveryRequired
		}
		if _, err := c.keys.EnvelopeOpener(*r.Active); err != nil {
			return nil, nil, err
		}
		envelope = bytes.Clone(r.Active.PublicKey)
	}
	return k, envelope, nil
}

func (c *Client) CompleteProvisioned(ctx context.Context, i enrollment.Invitation, q enrollment.SignedRequest, r enrollment.Result, now time.Time) error {
	if _, err := enrollment.ValidateResult(i, q, r, now); err != nil {
		return err
	}
	return c.publish(ctx, r.Approval.Principal, r.Chain, r.Snapshot, now)
}
func (c *Client) CompleteInteractive(ctx context.Context, w enrollment.Window, q enrollment.SignedInteractiveRequest, p enrollment.PairingContext, digest string, r enrollment.InteractiveResult, now time.Time) error {
	if _, err := enrollment.ValidateInteractiveResult(c.root, w, q, p, digest, r, now); err != nil {
		return err
	}
	return c.publish(ctx, r.Approval.Principal, r.Chain, r.Snapshot, now)
}
func (c *Client) publish(ctx context.Context, id string, chain [][]byte, snapshot trust.SignedSnapshot, now time.Time) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.stage.WithProcessLock(ctx, operationLock, func() error { return c.publishLocked(ctx, id, chain, snapshot, now) })
}

func (c *Client) publishLocked(ctx context.Context, id string, chain [][]byte, snapshot trust.SignedSnapshot, now time.Time) error {
	if c.published {
		return fmt.Errorf("enrollment already published")
	}
	r, err := c.manager.View(keylifecycle.Principal)
	if err != nil {
		return err
	}
	if r.Active == nil || r.Stage != "active" {
		return keylifecycle.ErrRecoveryRequired
	}
	s := principal.State{Version: principal.Version, Cluster: c.root.Cluster(), Role: c.intent.Role, Principal: id, Key: *r.Active, Root: c.root.DER(), Fingerprint: c.root.Fingerprint(), Chain: chain, Snapshot: snapshot}
	if c.intent.Role == trust.Node {
		e, err := c.manager.View(keylifecycle.NodeEnvelope)
		if err != nil {
			return err
		}
		if e.Active == nil || e.Stage != "active" {
			return keylifecycle.ErrRecoveryRequired
		}
		s.EnvelopeKey = e.Active
	}
	if err := principal.Install(ctx, c.stage, c.root, s, now); err != nil {
		return err
	}
	// After a promotion attempt, never destroy a possibly published identity.
	// A post-rename durability failure is reconciled by inspecting the target.
	c.published = true
	return c.final.Promote(c.stage)
}
