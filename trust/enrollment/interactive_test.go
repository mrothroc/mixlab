package enrollment

import (
	"bytes"
	"crypto"
	"crypto/ecdh"
	"crypto/rand"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"net/netip"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/trust"
)

type testChannel struct {
	id           [32]byte
	peer         netip.Addr
	iface        string
	closed, used bool
}

func channelFor(t *testing.T) *testChannel {
	t.Helper()
	c := &testChannel{peer: netip.MustParseAddr("192.168.1.5"), iface: "en0"}
	_, err := rand.Read(c.id[:])
	check(t, err)
	return c
}
func (c *testChannel) EvidenceID() [32]byte { return c.id }
func (c *testChannel) Observe() (netip.Addr, string, error) {
	if c.closed {
		return netip.Addr{}, "", fmt.Errorf("closed")
	}
	return c.peer, c.iface, nil
}
func (c *testChannel) Bind(d [32]byte) ([]byte, error) {
	if c.closed || c.used {
		return nil, fmt.Errorf("closed/used")
	}
	c.used = true
	v := sha256.Sum256(append(bytes.Clone(c.id[:]), d[:]...))
	return v[:], nil
}
func (f fixture) window(t *testing.T, policy Policy) Window {
	t.Helper()
	o := WindowOptions{Policy: policy, Endpoint: "https://enrollment.example:7444", Audience: "enrollment", MaxApprovals: 4}
	if policy == TrustedLAN {
		o.Interface = "en0"
		o.CIDRs = []string{"192.168.1.0/24"}
	}
	w, err := f.s.OpenWindow(ctx, o, f.v, now)
	check(t, err)
	return w
}
func interactiveFor(t *testing.T, f fixture, w Window, k crypto.Signer) SignedInteractiveRequest {
	t.Helper()
	if k == nil {
		k = key(t)
	}
	e, err := ecdh.X25519().GenerateKey(rand.Reader)
	check(t, err)
	q, err := NewInteractiveRequest(f.a.Anchor, w, NodeEnrollment, k, e.PublicKey().Bytes(), now)
	check(t, err)
	return q
}

func TestInteractiveBothPolicies(t *testing.T) {
	for _, policy := range []Policy{TrustedLAN, Verified} {
		t.Run(string(policy), func(t *testing.T) {
			f := setup(t)
			w := f.window(t, policy)
			q := interactiveFor(t, f, w, nil)
			c := channelFor(t)
			p, err := f.s.Begin(ctx, q, c, f.v, now)
			check(t, err)
			if policy == Verified {
				if p.Stage != "pending" || p.Result != nil || f.issuer.calls != 0 {
					t.Fatal("issued before confirmation")
				}
				p, err = f.s.Decide(ctx, p.ID, p.PairingDigest, true, f.v, now)
				check(t, err)
				if p.Stage != "pending" || f.issuer.calls != 0 {
					t.Fatal("issued with only administrator confirmation")
				}
				p, err = f.s.ConfirmClient(ctx, p.ID, p.PairingDigest, c, f.v, now)
				check(t, err)
			}
			if p.Stage != "issued" || p.Result == nil || f.issuer.calls != 1 {
				t.Fatal("missing issuance", p.Stage)
			}
			_, err = ValidateInteractiveResult(f.a.Anchor, w, q, p.Pairing, p.PairingDigest, *p.Result, now)
			check(t, err)
			again, err := f.s.Poll(ctx, p.ID, c, f.v, now)
			check(t, err)
			if f.issuer.calls != 1 || !bytes.Equal(again.Result.Chain[0], p.Result.Chain[0]) {
				t.Fatal("poll reissued")
			}
			if _, err := f.s.Begin(ctx, q, channelFor(t), f.v, now); err == nil {
				t.Fatal("duplicate key enrolled")
			}
			raw, err := f.p.ReadFile(stateFile)
			check(t, err)
			d, _ := p.Pairing.Digest()
			exporter := sha256.Sum256(append(bytes.Clone(c.id[:]), d[:]...))
			encoded, _ := json.Marshal(exporter[:])
			if bytes.Contains(raw, encoded) {
				t.Fatal("exporter persisted")
			}
		})
	}
}

func TestInteractiveRequiresExactLiveChannelAndPhrase(t *testing.T) {
	f := setup(t)
	w := f.window(t, Verified)
	q := interactiveFor(t, f, w, nil)
	c := channelFor(t)
	p, err := f.s.Begin(ctx, q, c, f.v, now)
	check(t, err)
	if _, err := f.s.Poll(ctx, p.ID, nil, f.v, now); err == nil {
		t.Fatal("anonymous poll")
	}
	if _, err := f.s.Poll(ctx, p.ID, channelFor(t), f.v, now); err == nil {
		t.Fatal("connection substitution")
	}
	if _, err := f.s.ConfirmClient(ctx, p.ID, strings.Repeat("a", 64), c, f.v, now); err == nil {
		t.Fatal("phrase substitution")
	}
	if _, err := f.s.Decide(ctx, p.ID, strings.Repeat("b", 64), true, f.v, now); err == nil {
		t.Fatal("wrong administrator comparison")
	}
	if _, err := f.s.Begin(ctx, interactiveFor(t, f, w, nil), c, f.v, now); err == nil {
		t.Fatal("channel reused")
	}
	_, err = f.s.ConfirmClient(ctx, p.ID, p.PairingDigest, c, f.v, now)
	check(t, err)
	if f.issuer.calls != 0 {
		t.Fatal("client can self-approve")
	}
	c.closed = true
	if _, err := f.s.Decide(ctx, p.ID, p.PairingDigest, true, f.v, now); err == nil {
		t.Fatal("approved disconnected request")
	}
	if f.issuer.calls != 0 {
		t.Fatal("issued on lost connection")
	}
}

func TestInteractiveTerminalAndRestart(t *testing.T) {
	for _, mode := range []string{"reject", "close-window", "expiry", "restart"} {
		t.Run(mode, func(t *testing.T) {
			f := setup(t)
			w := f.window(t, Verified)
			q := interactiveFor(t, f, w, nil)
			c := channelFor(t)
			p, err := f.s.Begin(ctx, q, c, f.v, now)
			check(t, err)
			at := now
			switch mode {
			case "reject":
				p, err = f.s.Decide(ctx, p.ID, p.PairingDigest, false, f.v, now)
				check(t, err)
				if p.Stage != "rejected" {
					t.Fatal(p.Stage)
				}
			case "close-window":
				check(t, f.s.CloseWindow(ctx, w.ID, f.v, now))
			case "expiry":
				at = now.Add(10 * time.Minute)
			case "restart":
				f.s, err = Open(f.p, f.a, f.v, now)
				check(t, err)
			}
			if _, err := f.s.ConfirmClient(ctx, p.ID, p.PairingDigest, c, f.v, at); err == nil {
				t.Fatal("terminal request confirmed")
			}
			if _, err := f.s.Decide(ctx, p.ID, p.PairingDigest, true, f.v, at); err == nil {
				t.Fatal("terminal request approved")
			}
			if f.issuer.calls != 0 {
				t.Fatal("terminal request issued")
			}
		})
	}
}

func TestInteractiveResultsRejectSubstitution(t *testing.T) {
	f := setup(t)
	w := f.window(t, TrustedLAN)
	q := interactiveFor(t, f, w, nil)
	p, err := f.s.Begin(ctx, q, channelFor(t), f.v, now)
	check(t, err)
	for _, field := range []string{"digest", "window", "policy", "evidence", "chain", "commit", "request"} {
		t.Run(field, func(t *testing.T) {
			b, _ := json.Marshal(p.Result)
			var r InteractiveResult
			check(t, json.Unmarshal(b, &r))
			ww, qq := w, q
			d := p.PairingDigest
			switch field {
			case "digest":
				d = strings.Repeat("0", 64)
			case "window":
				ww.ID = id(t)
			case "policy":
				ww.Policy = Verified
				ww.CIDRs = nil
				ww.Interface = ""
			case "evidence":
				r.Evidence.Peer = "192.168.1.6"
			case "chain":
				r.Chain = f.a.PrincipalChain
			case "commit":
				r.Commit.Generation++
			case "request":
				qq = interactiveFor(t, f, w, nil)
			}
			if _, err := ValidateInteractiveResult(f.a.Anchor, ww, qq, p.Pairing, d, r, now); err == nil {
				t.Fatal("substitution accepted")
			}
		})
	}
}

func TestInteractiveQuotasAndLANBounds(t *testing.T) {
	for _, field := range []string{"public", "interface", "privileged", "quota", "pending", "rate"} {
		t.Run(field, func(t *testing.T) {
			f := setup(t)
			o := WindowOptions{Policy: TrustedLAN, Endpoint: "https://source.example", Audience: "source", MaxApprovals: 1, Interface: "en0", CIDRs: []string{"192.168.1.0/24"}}
			if field == "pending" || field == "rate" {
				o.Policy = Verified
				o.Interface = ""
				o.CIDRs = nil
				o.MaxPending = 1
				o.MaxPerMinute = 1
			}
			w, err := f.s.OpenWindow(ctx, o, f.v, now)
			check(t, err)
			c := channelFor(t)
			q := interactiveFor(t, f, w, nil)
			switch field {
			case "public":
				c.peer = netip.MustParseAddr("8.8.8.8")
			case "interface":
				c.iface = "en1"
			case "privileged":
				q.Request.Purpose = ControllerEnrollment
				q.Request.Role = trust.Controller
			default:
				p, err := f.s.Begin(ctx, q, c, f.v, now)
				check(t, err)
				if field == "rate" {
					_, err = f.s.Decide(ctx, p.ID, p.PairingDigest, false, f.v, now)
					check(t, err)
				}
				q = interactiveFor(t, f, w, nil)
				c = channelFor(t)
			}
			if _, err := f.s.Begin(ctx, q, c, f.v, now); err == nil {
				t.Fatal("invalid request accepted")
			}
		})
	}
}

func TestInteractivePolicyCannotBeDowngraded(t *testing.T) {
	f := setup(t)
	w := f.window(t, TrustedLAN)
	claimed := w
	claimed.Policy = Verified
	claimed.Interface = ""
	claimed.CIDRs = nil
	q := interactiveFor(t, f, claimed, nil)
	if _, err := f.s.Begin(ctx, q, channelFor(t), f.v, now); err == nil {
		t.Fatal("verified request auto-approved")
	}
}

func TestInteractiveConcurrentApprovalSingleIssuance(t *testing.T) {
	f := setup(t)
	w := f.window(t, Verified)
	q := interactiveFor(t, f, w, nil)
	c := channelFor(t)
	p, err := f.s.Begin(ctx, q, c, f.v, now)
	check(t, err)
	_, err = f.s.ConfirmClient(ctx, p.ID, p.PairingDigest, c, f.v, now)
	check(t, err)
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() { defer wg.Done(); _, _ = f.s.Decide(ctx, p.ID, p.PairingDigest, true, f.v, now) }()
	}
	wg.Wait()
	result, err := f.s.Poll(ctx, p.ID, c, f.v, now)
	check(t, err)
	if result.Stage != "issued" || f.issuer.calls != 1 {
		t.Fatal("duplicate issuance", f.issuer.calls)
	}
}

func TestInteractiveAndProvisionedShareKeyReservation(t *testing.T) {
	for _, first := range []string{"interactive", "provisioned"} {
		t.Run(first, func(t *testing.T) {
			f := setup(t)
			w := f.window(t, Verified)
			i := f.invite(t, NodeEnrollment)
			defer i.Clear()
			k := key(t)
			q := interactiveFor(t, f, w, k)
			p, err := NewRequest(i, k, q.Request.EnvelopeKey, now)
			check(t, err)
			if first == "interactive" {
				_, err = f.s.Begin(ctx, q, channelFor(t), f.v, now)
				check(t, err)
				if _, err = f.s.Consume(ctx, p, i.Secret, f.v, now); err == nil {
					t.Fatal("duplicate key across policies")
				}
			} else {
				_, err = f.s.Consume(ctx, p, i.Secret, f.v, now)
				check(t, err)
				if _, err = f.s.Begin(ctx, q, channelFor(t), f.v, now); err == nil {
					t.Fatal("duplicate key across policies")
				}
			}
		})
	}
}
