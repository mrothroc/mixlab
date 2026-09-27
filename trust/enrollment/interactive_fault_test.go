package enrollment

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/trust"
)

type interactiveFault = failingJournal

func TestInteractivePersistenceBeforeIssuance(t *testing.T) {
	for _, stage := range []string{"pending", "approved", "issued"} {
		for _, after := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/after=%v", stage, after), func(t *testing.T) {
				f := setup(t)
				w := f.window(t, TrustedLAN)
				q := interactiveFor(t, f, w, nil)
				c := channelFor(t)
				base := f.s.store
				fault := &interactiveFault{journal: base, stage: stage, after: after}
				f.s.store = fault
				p, err := f.s.Begin(ctx, q, c, f.v, now)
				if err == nil || p.ID != "" || p.Result != nil || !fault.fired {
					t.Fatal("uncertain result escaped", err)
				}
				if (stage == "pending" || stage == "approved") && f.issuer.calls != 0 {
					t.Fatal("issuer called before durable approval")
				}
				f.s.store = base
				_, r, err := f.s.read()
				check(t, err)
				if len(r.Interactive) == 0 {
					return
				}
				e := r.Interactive[0]
				if stage == "pending" {
					if _, err := f.s.Poll(ctx, e.ID, c, f.v, now); err == nil {
						t.Fatal("unacknowledged request recovered remotely")
					}
					return
				}
				p, err = f.s.Poll(ctx, e.ID, c, f.v, now)
				check(t, err)
				if p.Stage != "issued" {
					t.Fatal(p.Stage)
				}
				if e.Result != nil && p.Result.Approval.Principal != e.Result.Approval.Principal {
					t.Fatal("recovery changed principal")
				}
				if stage == "issued" && after && !bytes.Equal(p.Result.Chain[0], e.Result.Chain[0]) {
					t.Fatal("recovery replaced issued certificate")
				}
			})
		}
	}
}

func TestInteractiveDisconnectExpiresDurably(t *testing.T) {
	f := setup(t)
	w := f.window(t, Verified)
	c := channelFor(t)
	q := interactiveFor(t, f, w, nil)
	p, err := f.s.Begin(ctx, q, c, f.v, now)
	check(t, err)
	check(t, f.s.Disconnect(ctx, c))
	_, r, err := f.s.read()
	check(t, err)
	if r.Interactive[0].Stage != "expired" {
		t.Fatal("disconnect not durable")
	}
	if _, err := f.s.Poll(ctx, p.ID, c, f.v, now); err == nil {
		t.Fatal("disconnected request accessible")
	}
	check(t, f.s.Disconnect(ctx, c))
}

func TestInteractiveRevocationBlocksApproval(t *testing.T) {
	f := setup(t)
	w := f.window(t, Verified)
	c := channelFor(t)
	q := interactiveFor(t, f, w, nil)
	p, err := f.s.Begin(ctx, q, c, f.v, now)
	check(t, err)
	_, err = f.s.ConfirmClient(ctx, p.ID, p.PairingDigest, c, f.v, now)
	check(t, err)
	v := f.view(t, 2, now, []trust.Revocation{{Kind: "principal", ID: f.s.owner, FirstGeneration: 2, Mode: "compromise", Reason: "test"}})
	if _, err := f.s.Decide(ctx, p.ID, p.PairingDigest, true, v, now); err == nil {
		t.Fatal("revoked authority approved")
	}
	if f.issuer.calls != 0 {
		t.Fatal("revoked authority issued")
	}
}

func TestWindowValidation(t *testing.T) {
	for _, cidr := range []string{"0.0.0.0/0", "10.0.0.1/8", "192.168.0.0/15", "8.8.8.0/24", "::/0", "::ffff:192.168.1.0/120"} {
		if _, err := privatePrefix(cidr); err == nil {
			t.Fatal("unsafe CIDR", cidr)
		}
	}
	for _, cidr := range []string{"10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "127.0.0.1/32", "fc00::/7", "fe80::/10", "::1/128"} {
		_, err := privatePrefix(cidr)
		check(t, err)
	}
	f := setup(t)
	for _, name := range []string{"missing-limit", "long-ttl", "empty-cidrs", "missing-interface", "privileged", "duplicate-purpose", "unknown-policy"} {
		t.Run(name, func(t *testing.T) {
			o := WindowOptions{Policy: TrustedLAN, Endpoint: "https://enroll.example", Audience: "enroll", Interface: "en0", CIDRs: []string{"10.0.0.0/8"}, MaxApprovals: 2}
			switch name {
			case "missing-limit":
				o.MaxApprovals = 0
			case "long-ttl":
				o.TTL = 2 * time.Hour
			case "empty-cidrs":
				o.CIDRs = nil
			case "missing-interface":
				o.Interface = ""
			case "privileged":
				o.Purposes = []Purpose{ControllerEnrollment}
			case "duplicate-purpose":
				o.Purposes = []Purpose{NodeEnrollment, NodeEnrollment}
			case "unknown-policy":
				o.Policy = "provisioned"
			}
			if _, err := f.s.OpenWindow(ctx, o, f.v, now); err == nil {
				t.Fatal("invalid window accepted")
			}
		})
	}
	w := f.window(t, Verified)
	if w.Expires-w.Created != 600 || w.MaxPending != 16 || w.MaxPerMinute != 16 {
		t.Fatal("window defaults")
	}
}

func TestProvisionedRequestWireLayoutUnchanged(t *testing.T) {
	f := setup(t)
	i := f.invite(t, NodeEnrollment)
	defer i.Clear()
	q := requestFor(t, i).Request
	b, err := json.Marshal(q)
	check(t, err)
	if strings.Contains(string(b), "PrincipalRequest") {
		t.Fatal("embedded request changed wire nesting")
	}
	want := []string{"version", "invitation", "cluster", "fingerprint", "endpoint", "audience", "purpose", "role", "public_key", "envelope_key", "nonce"}
	for _, name := range want {
		if !bytes.Contains(b, []byte(`"`+name+`":`)) {
			t.Fatal("missing wire field", name)
		}
	}
	var decoded Request
	check(t, json.Unmarshal(b, &decoded))
	again, err := json.Marshal(decoded)
	check(t, err)
	if !bytes.Equal(b, again) {
		t.Fatal("wire round trip changed")
	}
}
