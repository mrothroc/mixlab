package enrollment

import (
	"bytes"
	"encoding/json"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/trust"
)

func TestInteractiveDeliveryRefreshKeepsApprovalImmutable(t *testing.T) {
	f := setup(t)
	w, err := f.s.OpenWindow(ctx, WindowOptions{Policy: Verified, Endpoint: "https://source.example", Audience: "source", TTL: time.Hour, MaxApprovals: 1}, f.v, now)
	check(t, err)
	c := channelFor(t)
	q := interactiveFor(t, f, w, nil)
	p, err := f.s.Begin(ctx, q, c, f.v, now)
	check(t, err)
	_, err = f.s.ConfirmClient(ctx, p.ID, p.PairingDigest, c, f.v, now)
	check(t, err)
	p, err = f.s.Decide(ctx, p.ID, p.PairingDigest, true, f.v, now)
	check(t, err)
	original, err := json.Marshal(p.Result.Proof)
	check(t, err)
	later := now.Add(20 * time.Minute)
	v := f.view(t, 2, later, nil)
	if _, err := trust.VerifyProof(f.a.Anchor, v, p.Result.Proof, approvalRequest(p.Result.Approval), later); err == nil {
		t.Fatal("test did not expire the embedded approval snapshot")
	}
	again, err := f.s.Poll(ctx, p.ID, c, v, later)
	check(t, err)
	_, err = ValidateInteractiveResult(f.a.Anchor, w, q, p.Pairing, p.PairingDigest, *again.Result, later)
	check(t, err)
	proof, err := json.Marshal(again.Result.Proof)
	check(t, err)
	if !bytes.Equal(original, proof) || again.Result.Commit != p.Result.Commit || !bytes.Equal(p.Result.Chain[0], again.Result.Chain[0]) || f.issuer.calls != 1 {
		t.Fatal("delivery changed approval or reissued certificate")
	}
	_, saved, err := f.s.read()
	check(t, err)
	if saved.Interactive[0].Result.Delivery.Request != interactiveDeliveryRequest(*again.Result) {
		t.Fatal("delivery not persisted")
	}
	if _, err := f.s.Poll(ctx, p.ID, channelFor(t), v, later); err == nil {
		t.Fatal("delivery refresh allowed channel substitution")
	}
	late := now.Add(time.Hour)
	v = f.view(t, 3, late, nil)
	if _, err := f.s.Poll(ctx, p.ID, c, v, late); err == nil {
		t.Fatal("delivery reopened expired enrollment window")
	}
}
