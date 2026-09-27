package trust

import (
	"testing"
	"time"
)

func TestSnapshotAdvanceRejectsEndpointRollbackAndConflict(t *testing.T) {
	f := fixture(t)
	old := f.view.signed.Payload.Endpoints
	at := testNow.Add(time.Minute)
	endpoint := old.Payload
	endpoint.IssuedAt = at.Unix()
	endpoint.URLs = []string{"https://new-authority.example"}
	newEndpoints, err := SignAuthorityEndpoints(f.anchor, endpoint, f.rootKey, at)
	requireOK(t, err)
	p := cloneSnapshot(t, f.view.signed).Payload
	p.Generation++
	p.IssuedAt = at.Unix()
	p.ExpiresAt = at.Add(SnapshotLifetime).Unix()
	p.Endpoints = newEndpoints
	signed, err := SignSnapshot(f.anchor, p, f.signer, f.signerKey, at)
	requireOK(t, err)
	current, err := f.view.Advance(signed, at)
	requireOK(t, err)
	for _, scenario := range []string{"unchanged", "older-root-signature", "same-time-different-URLs"} {
		t.Run(scenario, func(t *testing.T) {
			p.Generation++
			p.Endpoints = newEndpoints
			switch scenario {
			case "older-root-signature":
				p.Endpoints = old
			case "same-time-different-URLs":
				conflict := newEndpoints.Payload
				conflict.URLs = []string{"https://another-authority.example"}
				p.Endpoints, err = SignAuthorityEndpoints(f.anchor, conflict, f.rootKey, at)
				requireOK(t, err)
			}
			next, err := SignSnapshot(f.anchor, p, f.signer, f.signerKey, at)
			requireOK(t, err)
			_, err = current.Advance(next, at)
			if (err == nil) != (scenario == "unchanged") {
				t.Fatal("unexpected endpoint advancement", err)
			}
		})
	}
}
