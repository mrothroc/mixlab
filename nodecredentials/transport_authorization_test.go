package nodecredentials

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"errors"
	"os"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/keylifecycle"
	"github.com/mrothroc/mixlab/trust/workload"
)

func TestTransportAuthorizationPreservesStaleOriginalAndKey(t *testing.T) {
	f := transportAuthorizationFixture(t)
	ctx := context.Background()
	original, err := f.credential.PrepareRequest(ctx, f.sign(f.view, f.now), f.view, f.now)
	credentialCheck(t, err)
	before, _, err := f.credential.load()
	credentialCheck(t, err)
	keyBefore, err := f.credential.manager.View(keylifecycle.Workload)
	credentialCheck(t, err)
	now := f.now.Add(trust.SnapshotLifetime + time.Second)
	snapshot, err := f.snapshots.Refresh(ctx, now)
	credentialCheck(t, err)
	current, err := trust.VerifySnapshot(f.anchor, snapshot, now)
	credentialCheck(t, err)
	if _, err = trust.VerifyProof(f.anchor, current, original.Proof, original.Proof.Request, now); err == nil {
		t.Fatal("fixture original proof is not stale")
	}
	auth, err := f.credential.AuthorizeRequest(ctx, current, f.sign(current, now), now)
	credentialCheck(t, err)
	credentialCheck(t, workload.VerifyRequestAuthorization(f.anchor, current, original, auth, now))
	after, _, err := f.credential.load()
	credentialCheck(t, err)
	keyAfter, err := f.credential.manager.View(keylifecycle.Workload)
	credentialCheck(t, err)
	if !bytes.Equal(before, after) || !samePublicValue(keyBefore, keyAfter) {
		t.Fatal("fresh supplement rewrote request or protected key lifecycle")
	}
	credentialCheck(t, f.credential.Close())
	f.credential, err = OpenTransport(f.path, f.anchor, original.Request.Scope)
	credentialCheck(t, err)
	replayed, err := f.credential.PrepareRequest(ctx, func(context.Context, trust.SignRequest) (trust.SignedProof, error) {
		t.Error("reauthorization replaced original signing operation")
		return trust.SignedProof{}, errors.New("unexpected signing")
	}, current, now)
	credentialCheck(t, err)
	if !samePublicValue(original, replayed) {
		t.Fatal("reopening changed immutable original request")
	}
	changed := original
	changed.Proof.Signature = []byte("different proof")
	if workload.VerifyRequestAuthorization(f.anchor, current, changed, auth, now) == nil {
		t.Fatal("supplement accepted a changed original proof")
	}
	changed = original
	changed.Request.Scope.Attempt = changed.Request.Scope.Job
	if workload.VerifyRequestAuthorization(f.anchor, current, changed, auth, now) == nil {
		t.Fatal("supplement accepted a changed scope")
	}
}

func TestTransportAuthorizationRejectsInvalidFreshProof(t *testing.T) {
	f := transportAuthorizationFixture(t)
	ctx := context.Background()
	_, err := f.credential.PrepareRequest(ctx, f.sign(f.view, f.now), f.view, f.now)
	credentialCheck(t, err)
	now := f.now.Add(trust.SnapshotLifetime + time.Second)
	snapshot, err := f.snapshots.Refresh(ctx, now)
	credentialCheck(t, err)
	current, err := trust.VerifySnapshot(f.anchor, snapshot, now)
	credentialCheck(t, err)
	for _, mode := range []string{"stale", "forged", "changed_request", "wrong_node", "canceled", "expired"} {
		t.Run(mode, func(t *testing.T) {
			call, cancel := context.WithCancel(ctx)
			defer cancel()
			callNow := now
			if mode == "expired" {
				callNow = time.Unix(f.credential.scope.AdmitUntil, 0)
			}
			sign := func(ctx context.Context, q trust.SignRequest) (trust.SignedProof, error) {
				if mode == "expired" {
					t.Error("expired scope reached signer")
				}
				if mode == "stale" {
					return f.sign(f.view, f.now)(ctx, q)
				}
				p, err := f.sign(current, now)(ctx, q)
				switch mode {
				case "forged":
					p.Signature[0] ^= 1
				case "changed_request":
					p.Request.Digest = "changed"
				case "wrong_node":
					p.Evidence.Principal = f.credential.scope.Controller
				case "canceled":
					cancel()
				}
				return p, err
			}
			out, err := f.credential.AuthorizeRequest(call, current, sign, callNow)
			if err == nil || out.Version != "" {
				t.Fatal("invalid supplemental proof escaped")
			}
		})
	}
	valid, err := f.credential.AuthorizeRequest(ctx, current, f.sign(current, now), now)
	credentialCheck(t, err)
	revoked, err := f.snapshots.Revoke(ctx, "principal", f.credential.scope.Node, "prospective", "test revocation", now)
	credentialCheck(t, err)
	current, err = trust.VerifySnapshot(f.anchor, revoked, now)
	credentialCheck(t, err)
	if _, err = f.credential.AuthorizeRequest(ctx, current, func(context.Context, trust.SignRequest) (trust.SignedProof, error) {
		return valid.Proof, nil
	}, now); err == nil {
		t.Fatal("revoked signer reused an otherwise valid supplement")
	}
}

func TestTransportAuthorizationCannotCreateOrRepairKey(t *testing.T) {
	for _, mode := range []string{"no_request", "missing_key", "different_key", "destroyed", "changed_scope"} {
		t.Run(mode, func(t *testing.T) {
			f := transportAuthorizationFixture(t)
			ctx := context.Background()
			if mode != "no_request" {
				_, err := f.credential.PrepareRequest(ctx, f.sign(f.view, f.now), f.view, f.now)
				credentialCheck(t, err)
			}
			switch mode {
			case "missing_key":
				state, err := f.credential.manager.View(keylifecycle.Workload)
				credentialCheck(t, err)
				credentialCheck(t, os.Remove(f.path.Dir()+"/key-"+state.Active.ID+".json"))
			case "different_key":
				old, r, err := f.credential.load()
				credentialCheck(t, err)
				key := ed25519.NewKeyFromSeed(bytes.Repeat([]byte{7}, ed25519.SeedSize))
				r.Request.Request, err = workload.NewRequest(r.Scope, key)
				credentialCheck(t, err)
				r.Request.Proof.Request, err = r.Request.Request.SigningRequest()
				credentialCheck(t, err)
				credentialCheck(t, f.credential.save(old, r))
			case "destroyed":
				credentialCheck(t, f.credential.Destroy(ctx))
			case "changed_scope":
				f.credential.scope.Attempt = f.credential.scope.Job
			}
			before, err := f.path.ReadFile(transportFile)
			credentialCheck(t, err)
			_, err = f.credential.AuthorizeRequest(ctx, f.view, func(context.Context, trust.SignRequest) (trust.SignedProof, error) {
				t.Error("missing/changed credential reached signer")
				return trust.SignedProof{}, nil
			}, f.now)
			if err == nil {
				t.Fatal("reauthorization repaired unavailable credential")
			}
			after, err := f.path.ReadFile(transportFile)
			credentialCheck(t, err)
			if !bytes.Equal(before, after) {
				t.Fatal("failed reauthorization rewrote credential state")
			}
			if mode == "no_request" {
				_, err := f.credential.manager.View(keylifecycle.Workload)
				if !errors.Is(err, os.ErrNotExist) {
					t.Fatal("reauthorization created a key intent", err)
				}
			}
		})
	}
}

func credentialCheck(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}
