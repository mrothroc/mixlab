package enrollment

import (
	"testing"
	"time"
)

func TestExpiredInteractiveRepairWithoutLiveChannel(t *testing.T) {
	f := setup(t)
	w := f.window(t, Verified)
	q := interactiveFor(t, f, w, nil)
	p, err := f.s.Begin(ctx, q, channelFor(t), f.v, now)
	check(t, err)
	reopened, err := Open(f.p, f.a, f.v, now.Add(10*time.Minute))
	check(t, err)
	_, r, err := reopened.read()
	check(t, err)
	if len(r.Interactive) != 1 || r.Interactive[0].ID != p.ID || r.Interactive[0].Stage != "expired" {
		t.Fatal("expired window did not repair durable request state")
	}
	if f.issuer.calls != 0 {
		t.Fatal("repair issued a certificate")
	}
}
