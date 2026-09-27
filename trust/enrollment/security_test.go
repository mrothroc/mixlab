package enrollment

import (
	"bytes"
	"encoding/json"
	"errors"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/trust"
)

func TestInvitationBoundsAndTarget(t *testing.T) {
	f := setup(t)
	for _, p := range []Purpose{"", "external-worker-bootstrap", "authority-enrollment"} {
		if _, err := f.s.Invite(ctx, p, "https://authority.example:7443", "authority", time.Minute, f.v, now); err == nil {
			t.Fatal("unsupported purpose accepted")
		}
	}
	for _, ttl := range []time.Duration{0, -time.Second, time.Hour + time.Second} {
		if _, err := f.s.Invite(ctx, NodeEnrollment, "https://authority.example:7443", "authority", ttl, f.v, now); err == nil {
			t.Fatal("invalid TTL accepted")
		}
	}
	for _, ep := range []string{"http://authority.example:7443", "https://authority.example:7443/other", "https://other.example"} {
		if _, err := f.s.Invite(ctx, NodeEnrollment, ep, "authority", time.Minute, f.v, now); err == nil {
			t.Fatal("unsigned endpoint accepted")
		}
	}
	i := f.invite(t, NodeEnrollment)
	defer i.Clear()
	if _, err := i.ValidateTarget("https://other.example", i.Audience, now); err == nil {
		t.Fatal("target mismatch accepted")
	}
	if _, err := i.ValidateTarget(i.Endpoint, "other", now); err == nil {
		t.Fatal("audience mismatch accepted")
	}
	if _, err := i.ValidateTarget(i.Endpoint, i.Audience, now.Add(10*time.Minute)); err == nil {
		t.Fatal("expired file accepted")
	}
	j := i
	j.Secret = make([]byte, 32)
	if _, err := j.ValidateTarget(j.Endpoint, j.Audience, now); err == nil {
		t.Fatal("zero secret accepted")
	}
}

func TestDuplicateAndReservedKeysRejected(t *testing.T) {
	f := setup(t)
	i, j := f.invite(t, ControllerEnrollment), f.invite(t, ControllerEnrollment)
	defer i.Clear()
	defer j.Clear()
	k := key(t)
	q, err := NewRequest(i, k, nil, now)
	check(t, err)
	_, err = f.s.Consume(ctx, q, i.Secret, f.v, now)
	check(t, err)
	q, err = NewRequest(j, k, nil, now)
	check(t, err)
	if _, err = f.s.Consume(ctx, q, j.Secret, f.v, now); err == nil {
		t.Fatal("duplicate principal key accepted")
	}
	q, err = NewRequest(j, f.a.PrincipalKey, nil, now)
	check(t, err)
	if _, err = f.s.Consume(ctx, q, j.Secret, f.v, now); err == nil {
		t.Fatal("authority key accepted as controller")
	}
	q, err = NewRequest(j, f.a.IssuerKey, nil, now)
	check(t, err)
	if _, err = f.s.Consume(ctx, q, j.Secret, f.v, now); err == nil {
		t.Fatal("issuer key accepted as controller")
	}
	if f.issuer.calls != 2 { // One legitimate certificate plus one test PoP signature.
		t.Fatal("rejected requests reached issuer", f.issuer.calls)
	}
}

func TestEnrollmentCurrentTrustAndRevocation(t *testing.T) {
	f := setup(t)
	i := f.invite(t, ControllerEnrollment)
	defer i.Clear()
	q := requestFor(t, i)
	v2 := f.view(t, 2, now, nil)
	_, err := f.s.Invite(ctx, NodeEnrollment, i.Endpoint, i.Audience, time.Minute, v2, now)
	check(t, err)
	if _, err := f.s.Consume(ctx, q, i.Secret, f.v, now); err == nil {
		t.Fatal("rollback snapshot accepted")
	}
	if _, err := Open(f.p, f.a, f.v, now); err == nil {
		t.Fatal("reopen with old trust accepted")
	}
	out, err := f.s.Consume(ctx, q, i.Secret, v2, now)
	check(t, err)
	v3 := f.view(t, 3, now, []trust.Revocation{{Kind: "principal", ID: out.Approval.Principal, Mode: "compromise", Reason: "test", FirstGeneration: 3}})
	if _, err := f.s.RecoverApproved(ctx, i.ID, v3, now); err == nil {
		t.Fatal("revoked issued identity returned as usable")
	}
	v4 := f.view(t, 4, now, []trust.Revocation{{Kind: "principal", ID: f.s.owner, Mode: "compromise", Reason: "test", FirstGeneration: 4}})
	if _, err := f.s.Invite(ctx, NodeEnrollment, i.Endpoint, i.Audience, time.Minute, v4, now); err == nil {
		t.Fatal("revoked authority issued invitation")
	}
	if _, err := f.s.Invite(ctx, NodeEnrollment, i.Endpoint, i.Audience, time.Minute, v2, now.Add(trust.SnapshotLifetime)); err == nil {
		t.Fatal("expired trust permitted invitation")
	}
}

func TestExpiredApprovalCannotIssueOnRecovery(t *testing.T) {
	f := setup(t)
	i := f.invite(t, NodeEnrollment)
	defer i.Clear()
	q := requestFor(t, i)
	f.s.store = &failingJournal{journal: f.p, stage: "approved", after: true}
	if _, err := f.s.Consume(ctx, q, i.Secret, f.v, now); err == nil {
		t.Fatal("fault missing")
	}
	reopened, err := Open(f.p, f.a, f.v, now)
	check(t, err)
	if _, err := reopened.RecoverApproved(ctx, i.ID, f.v, now.Add(10*time.Minute)); !errors.Is(err, ErrExpired) {
		t.Fatal("expired approval recovered", err)
	}
	if f.issuer.calls != 0 {
		t.Fatal("expired approval issued certificate")
	}
}

func TestClientRejectsAlteredEnrollmentResult(t *testing.T) {
	f := setup(t)
	i := f.invite(t, NodeEnrollment)
	defer i.Clear()
	q := requestFor(t, i)
	out, err := f.s.Consume(ctx, q, i.Secret, f.v, now)
	check(t, err)
	raw, err := json.Marshal(out)
	check(t, err)
	for _, name := range []string{"receipt", "approval", "signature", "chain", "snapshot", "key", "principal"} {
		t.Run(name, func(t *testing.T) {
			var copy Result
			check(t, json.Unmarshal(raw, &copy))
			switch name {
			case "receipt":
				copy.Receipt.Sequence++
			case "approval":
				copy.Approval.Audience = "other"
			case "signature":
				copy.ReceiptProof.Signature[0] ^= 1
			case "chain":
				copy.Chain[0] = bytes.Clone(f.a.PrincipalChain[0])
			case "snapshot":
				copy.Snapshot.Signature[0] ^= 1
			case "key":
				copy.Chain[1] = copy.Chain[2]
			case "principal":
				copy.Approval.Principal = id(t)
			}
			if _, err := ValidateResult(i, q, copy, now); err == nil {
				t.Fatal("tampered enrollment result accepted")
			}
		})
	}
}
