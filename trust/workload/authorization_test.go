package workload

import (
	"bytes"
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/trust"
)

func (f *fixture) authorize(t *testing.T, g Grant) AuthorizedGrant {
	t.Helper()
	nq, err := g.Request.ReauthorizationRequest()
	check(t, err)
	np, err := trust.SignPrincipalProof(f.a.Anchor, f.node, f.nodeKey, f.view, nq, f.now)
	check(t, err)
	gq, err := g.ReauthorizationRequest()
	check(t, err)
	gp, err := trust.SignPrincipalProof(f.a.Anchor, f.controller, f.controllerKey, f.view, gq, f.now)
	check(t, err)
	return AuthorizedGrant{Version: AuthorizedGrantVersion, Grant: g, NodeAuthorization: RequestAuthorization{Version: RequestAuthorizationVersion, Proof: np}, ControllerAuthorization: gp}
}

func (f *fixture) advance(t *testing.T, d time.Duration) {
	t.Helper()
	f.now = f.now.Add(d)
	f.snapshot.Payload.Generation++
	f.snapshot.Payload.IssuedAt, f.snapshot.Payload.ExpiresAt = f.now.Unix(), f.now.Add(trust.SnapshotLifetime).Unix()
	f.refresh(t)
}

func cloneAuthorized(t *testing.T, g AuthorizedGrant) AuthorizedGrant {
	t.Helper()
	b, err := json.Marshal(g)
	check(t, err)
	var out AuthorizedGrant
	check(t, json.Unmarshal(b, &out))
	return out
}

func TestVerifyRequestAuthorizationAdmissionExpiry(t *testing.T) {
	f := newFixture(t)
	scope := f.grant.Request.Request.Scope
	scope.AdmitUntil = f.now.Add(time.Minute).Unix()
	grant := f.sign(t, scope, newKey(t))
	auth := f.authorize(t, grant).NodeAuthorization
	want, err := grant.Request.ReauthorizationRequest()
	check(t, err)
	for _, tc := range []struct {
		name   string
		offset time.Duration
		valid  bool
	}{
		{name: "before_deadline", offset: -time.Second, valid: true},
		{name: "at_deadline"},
		{name: "after_deadline", offset: time.Second},
	} {
		t.Run(tc.name, func(t *testing.T) {
			now := time.Unix(scope.AdmitUntil, 0).Add(tc.offset)
			// Keep the signature, certificate and both snapshots valid so only
			// the request's admission deadline can reject this authorization.
			_, err := trust.VerifyProof(f.a.Anchor, f.view, auth.Proof, want, now)
			check(t, err)
			err = VerifyRequestAuthorization(f.a.Anchor, f.view, grant.Request, auth, now)
			if tc.valid {
				check(t, err)
			} else if err == nil {
				t.Fatal("fresh proof bypassed the request's admission deadline")
			}
		})
	}
}

func TestAuthorizedWorkloadExpiredOriginalAndHistoricalReplay(t *testing.T) {
	f := newFixture(t)
	original := digest(f.grant)
	f.advance(t, 20*time.Minute)
	if _, err := f.s.Issue(context.Background(), f.controller, f.grant, f.view, f.now); err == nil {
		t.Fatal("ordinary issuance accepted expired original evidence")
	}
	auth := f.authorize(t, f.grant)
	check(t, VerifyRequestAuthorization(f.a.Anchor, f.view, f.grant.Request, auth.NodeAuthorization, f.now))
	check(t, VerifyAuthorizedGrant(f.a.Anchor, f.view, auth, f.now))
	out, err := f.s.IssueAuthorized(context.Background(), f.controller, auth, f.view, f.now)
	check(t, err)
	check(t, ValidateResult(f.a.Anchor, f.grant, out, f.now))
	_, saved, err := f.s.load()
	check(t, err)
	e := saved.Entries[0]
	if e.Authorization == nil || e.NodeCommit.Digest != digest(auth.NodeAuthorization.Proof) || e.GrantCommit.Digest != digest(auth.ControllerAuthorization) || e.NodeCommit.Generation != 2 || e.NodeCommit.Digest == digest(f.grant.Request.Proof) || e.GrantCommit.Digest == digest(f.grant.Proof) || digest(e.Grant) != original || out.GrantHash != original {
		t.Fatal("supplemental acceptance rewrote grant or fabricated original-proof receipts")
	}
	f.advance(t, 20*time.Minute)
	if VerifyAuthorizedGrant(f.a.Anchor, f.view, auth, f.now) == nil {
		t.Fatal("fresh-only helper accepted expired supplements")
	}
	f.s, err = Open(f.path, f.a, f.view, f.now)
	check(t, err)
	for _, incoming := range []AuthorizedGrant{auth, f.authorize(t, f.grant)} {
		again, err := f.s.IssueAuthorized(context.Background(), f.controller, incoming, f.view, f.now)
		check(t, err)
		if !bytes.Equal(out.Chain[0], again.Chain[0]) || again.Binding != out.Binding || again.GrantHash != original || f.issuer.calls != 1 {
			t.Fatal("retry changed certificate, timestamp or grant")
		}
		_, current, err := f.s.load()
		check(t, err)
		if digest(current.Entries[0]) != digest(e) {
			t.Fatal("retry replaced accepted evidence/result")
		}
	}
}

func TestAuthorizedWorkloadCannotBackdateFirstAcceptance(t *testing.T) {
	f := newFixture(t)
	auth := f.authorize(t, f.grant)
	f.advance(t, 20*time.Minute)
	_, err := f.s.IssueAuthorized(context.Background(), f.controller, auth, f.view, f.now)
	if err == nil || f.issuer.calls != 0 {
		t.Fatal("expired uncommitted supplements reached CA")
	}
	_, r, err := f.s.load()
	check(t, err)
	if len(r.Entries) != 0 {
		t.Fatal("invalid authorization reserved issuance")
	}
}

func TestAuthorizedWorkloadLegacyJournalInteroperability(t *testing.T) {
	for _, firstAuthorized := range []bool{false, true} {
		t.Run(map[bool]string{false: "legacy_first", true: "authorized_first"}[firstAuthorized], func(t *testing.T) {
			f := newFixture(t)
			var out Result
			var err error
			if firstAuthorized {
				out, err = f.s.IssueAuthorized(context.Background(), f.controller, f.authorize(t, f.grant), f.view, f.now)
			} else {
				out, err = f.s.Issue(context.Background(), f.controller, f.grant, f.view, f.now)
			}
			check(t, err)
			_, before, err := f.s.load()
			check(t, err)
			f.advance(t, 20*time.Minute)
			f.s, err = Open(f.path, f.a, f.view, f.now)
			check(t, err)
			var again Result
			if firstAuthorized {
				again, err = f.s.Issue(context.Background(), f.controller, f.grant, f.view, f.now)
			} else {
				again, err = f.s.IssueAuthorized(context.Background(), f.controller, f.authorize(t, f.grant), f.view, f.now)
			}
			check(t, err)
			_, after, err := f.s.load()
			check(t, err)
			if digest(before.Entries) != digest(after.Entries) || !bytes.Equal(out.Chain[0], again.Chain[0]) || f.issuer.calls != 1 {
				t.Fatal("cross-path retry migrated receipts or reissued")
			}
		})
	}
}

func TestAuthorizedWorkloadRejectsTampering(t *testing.T) {
	for _, mode := range []string{"node_signature", "controller_signature", "original_node_proof", "original_grant_proof", "scope", "csr", "node_purpose", "controller_purpose", "version", "node_version", "wrong_caller", "expired", "stale_view"} {
		t.Run(mode, func(t *testing.T) {
			f := newFixture(t)
			auth := cloneAuthorized(t, f.authorize(t, f.grant))
			chain, view := f.controller, f.view
			switch mode {
			case "node_signature":
				auth.NodeAuthorization.Proof.Signature[0] ^= 1
			case "controller_signature":
				auth.ControllerAuthorization.Signature[0] ^= 1
			case "original_node_proof":
				auth.Grant.Request.Proof.Signature[0] ^= 1
			case "original_grant_proof":
				auth.Grant.Proof.Signature[0] ^= 1
			case "scope":
				auth.Grant.Request.Request.Scope.Run = randomID(t)
			case "csr":
				auth.Grant.Request.Request.CSR[10] ^= 1
			case "node_purpose":
				auth.NodeAuthorization.Proof.Request.Purpose = trust.WorkloadKeyRequest
			case "controller_purpose":
				auth.ControllerAuthorization.Request.Purpose = trust.WorkloadGrant
			case "version":
				auth.Version = "unknown"
			case "node_version":
				auth.NodeAuthorization.Version = "unknown"
			case "wrong_caller":
				chain = f.node
			case "expired":
				f.advance(t, time.Hour)
				view = f.view
			case "stale_view":
				f.now = f.now.Add(20 * time.Minute)
			}
			if _, err := f.s.IssueAuthorized(context.Background(), chain, auth, view, f.now); err == nil || f.issuer.calls != 0 {
				t.Fatalf("%s reached issuance: %v", mode, err)
			}
		})
	}
}

func TestWorkloadAuthorizationPurposeSeparation(t *testing.T) {
	f := newFixture(t)
	nq, err := f.grant.Request.ReauthorizationRequest()
	check(t, err)
	gq, err := f.grant.ReauthorizationRequest()
	check(t, err)
	if nq.Purpose == f.grant.Request.Proof.Request.Purpose || gq.Purpose == f.grant.Proof.Request.Purpose || nq.Digest != digest(f.grant.Request) || gq.Digest != digest(f.grant) {
		t.Fatal("authorization domains do not bind exact original envelopes")
	}
	if _, err := trust.SignPrincipalProof(f.a.Anchor, f.node, f.nodeKey, f.view, gq, f.now); err == nil {
		t.Fatal("node authorized grant")
	}
	if _, err := trust.SignPrincipalProof(f.a.Anchor, f.controller, f.controllerKey, f.view, nq, f.now); err == nil {
		t.Fatal("controller authorized node request")
	}
}
