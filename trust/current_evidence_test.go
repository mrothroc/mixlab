package trust

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"
)

func currentEvidenceProof(t *testing.T, f trustFixture, v VerifiedSnapshot) SignedProof {
	t.Helper()
	p := f.principal(t, Coordinator)
	proof, err := SignPrincipalProof(f.anchor, p.chain, p.key, v, request(ArtifactGrant), testNow)
	requireOK(t, err)
	return proof
}

func TestCurrentForEvidenceRefreshAndLocalAcceptance(t *testing.T) {
	f := fixture(t)
	peer := f.snapshot(t, 2, testNow, nil)
	// A newer independent local view makes accepting the peer view observable.
	latest := f.snapshot(t, 3, testNow, nil)
	proof := currentEvidenceProof(t, f, peer)
	local := f.view
	var events []string
	got, err := CurrentForEvidence(context.Background(), f.anchor, proof.Evidence.Snapshot,
		func(context.Context) (VerifiedSnapshot, error) {
			events = append(events, "current")
			return local, nil
		}, func(context.Context) error {
			events = append(events, "refresh")
			local = latest
			return nil
		}, func() time.Time { return testNow }, func(v VerifiedSnapshot) error {
			if v.Generation() == peer.Generation() {
				events = append(events, "preverify")
			} else {
				events = append(events, "verify-local")
			}
			_, err := VerifyProof(f.anchor, v, proof, request(ArtifactGrant), testNow)
			return err
		})
	requireOK(t, err)
	if got.Generation() != latest.Generation() || !reflect.DeepEqual(events, []string{"current", "preverify", "refresh", "current", "verify-local"}) {
		t.Fatalf("acceptance did not preverify then reread independent trust: generation=%d events=%v", got.Generation(), events)
	}
}

func TestCurrentForEvidenceBehindNeverInstallsPeer(t *testing.T) {
	for _, mode := range []string{"nil-refresh", "unchanged-refresh", "failed-refresh"} {
		t.Run(mode, func(t *testing.T) {
			f := fixture(t)
			proof := currentEvidenceProof(t, f, f.snapshot(t, 2, testNow, nil))
			fetchErr := errors.New("pinned authority unavailable")
			var refresh func(context.Context) error
			refreshes, reads, verifies := 0, 0, 0
			if mode != "nil-refresh" {
				refresh = func(context.Context) error {
					refreshes++
					if mode == "failed-refresh" {
						return fetchErr
					}
					return nil
				}
			}
			got, err := CurrentForEvidence(context.Background(), f.anchor, proof.Evidence.Snapshot,
				func(context.Context) (VerifiedSnapshot, error) { reads++; return f.view, nil }, refresh,
				func() time.Time { return testNow }, func(v VerifiedSnapshot) error {
					verifies++
					_, err := VerifyProof(f.anchor, v, proof, request(ArtifactGrant), testNow)
					return err
				})
			if !errors.Is(err, ErrTrustBehind) || got.Generation() != 0 || f.view.Generation() != 1 || reads != 2 || verifies != 1 {
				t.Fatalf("peer evidence became local authority: got=%d local=%d reads=%d verifies=%d err=%v", got.Generation(), f.view.Generation(), reads, verifies, err)
			}
			wantRefreshes := 1
			if mode == "nil-refresh" {
				wantRefreshes = 0
			}
			if refreshes != wantRefreshes || (mode == "failed-refresh" && !errors.Is(err, fetchErr)) {
				t.Fatalf("lost refresh outcome: refreshes=%d err=%v", refreshes, err)
			}
		})
	}
}

func TestCurrentForEvidenceConcurrentRefreshSupersedesFailedFetch(t *testing.T) {
	f := fixture(t)
	proof := currentEvidenceProof(t, f, f.snapshot(t, 2, testNow, nil))
	latest := f.snapshot(t, 3, testNow, nil)
	local := f.view
	var mu sync.Mutex
	fetchStarted, installed := make(chan struct{}), make(chan struct{})
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	go func() {
		defer close(installed)
		select {
		case <-fetchStarted:
			mu.Lock()
			local = latest
			mu.Unlock()
		case <-ctx.Done():
		}
	}()
	got, err := CurrentForEvidence(ctx, f.anchor, proof.Evidence.Snapshot,
		func(context.Context) (VerifiedSnapshot, error) {
			mu.Lock()
			defer mu.Unlock()
			return local, nil
		}, func(ctx context.Context) error {
			close(fetchStarted)
			select {
			case <-installed:
				return errors.New("this fetch failed after maintenance installed newer trust")
			case <-ctx.Done():
				return ctx.Err()
			}
		}, func() time.Time { return testNow }, func(v VerifiedSnapshot) error {
			_, err := VerifyProof(f.anchor, v, proof, request(ArtifactGrant), testNow)
			return err
		})
	cancel()
	<-installed
	requireOK(t, err)
	if got.Generation() != latest.Generation() {
		t.Fatalf("concurrent local refresh ignored: generation=%d", got.Generation())
	}
}

func TestCurrentForEvidencePostRefreshRevocationIsPermanent(t *testing.T) {
	for _, mode := range []string{"prospective", "compromise"} {
		t.Run(mode, func(t *testing.T) {
			f := fixture(t)
			proof := currentEvidenceProof(t, f, f.snapshot(t, 2, testNow, nil))
			revoked := f.snapshot(t, 3, testNow, []Revocation{{Kind: "principal", ID: proof.Evidence.Principal, Mode: mode, Reason: "retired signer", FirstGeneration: 3}})
			local := f.view
			verifies, refreshes := 0, 0
			got, err := CurrentForEvidence(context.Background(), f.anchor, proof.Evidence.Snapshot,
				func(context.Context) (VerifiedSnapshot, error) { return local, nil },
				func(context.Context) error { refreshes++; local = revoked; return nil },
				func() time.Time { return testNow }, func(v VerifiedSnapshot) error {
					verifies++
					_, err := VerifyProof(f.anchor, v, proof, request(ArtifactGrant), testNow)
					return err
				})
			if err == nil || errors.Is(err, ErrTrustBehind) || !strings.Contains(err.Error(), "revoked") || got.Generation() != 0 || verifies != 2 || refreshes != 1 {
				t.Fatalf("revocation was accepted or made retryable: got=%d verifies=%d refreshes=%d err=%v", got.Generation(), verifies, refreshes, err)
			}
		})
	}
}

func TestCurrentForEvidenceInvalidEvidenceNeverRefreshes(t *testing.T) {
	f := fixture(t)
	proof := currentEvidenceProof(t, f, f.snapshot(t, 2, testNow, nil))
	for _, name := range []string{"proof-signature", "snapshot-signature", "snapshot-generation", "identity", "expected-digest", "expected-purpose", "expected-context", "expected-audience", "expired-snapshot"} {
		t.Run(name, func(t *testing.T) {
			p := cloneProof(t, proof)
			want, now := request(ArtifactGrant), testNow
			switch name {
			case "proof-signature":
				p.Signature[0] ^= 1
			case "snapshot-signature":
				p.Evidence.Snapshot.Signature[0] ^= 1
			case "snapshot-generation":
				p.Evidence.Snapshot.Payload.Generation++
			case "identity":
				p.Evidence.Principal = newID(t)
			case "expected-digest":
				want.Digest = strings.Repeat("cd", 32)
			case "expected-purpose":
				want.Purpose = RunCommit
			case "expected-context":
				want.Context = "another-job"
			case "expected-audience":
				want.Audience = "another-node"
			case "expired-snapshot":
				now = testNow.Add(SnapshotLifetime)
			}
			// Even at expiry, the independently loaded lower generation is fresh.
			local := f.snapshot(t, 1, now, nil)
			reads, refreshes, verifies := 0, 0, 0
			got, err := CurrentForEvidence(context.Background(), f.anchor, p.Evidence.Snapshot,
				func(context.Context) (VerifiedSnapshot, error) { reads++; return local, nil },
				func(context.Context) error { refreshes++; return nil },
				func() time.Time { return now }, func(v VerifiedSnapshot) error {
					verifies++
					_, err := VerifyProof(f.anchor, v, p, want, now)
					return err
				})
			if err == nil || errors.Is(err, ErrTrustBehind) || refreshes != 0 || reads != 1 || got.Generation() != 0 {
				t.Fatalf("invalid evidence caused refresh/acceptance: reads=%d refreshes=%d got=%d err=%v", reads, refreshes, got.Generation(), err)
			}
			wantVerifies := 1
			if name == "snapshot-signature" || name == "snapshot-generation" || name == "expired-snapshot" {
				wantVerifies = 0
			}
			if verifies != wantVerifies {
				t.Fatalf("snapshot was not verified before the complete-proof callback: verifies=%d want=%d", verifies, wantVerifies)
			}
		})
	}
}

func TestCurrentForEvidenceCurrentOrOlderSkipsRefresh(t *testing.T) {
	f := fixture(t)
	proof := currentEvidenceProof(t, f, f.view)
	for _, generation := range []uint64{1, 2} {
		local := f.snapshot(t, generation, testNow, nil)
		reads, verifies, refreshes := 0, 0, 0
		got, err := CurrentForEvidence(context.Background(), f.anchor, proof.Evidence.Snapshot,
			func(context.Context) (VerifiedSnapshot, error) { reads++; return local, nil },
			func(context.Context) error { refreshes++; return nil },
			func() time.Time { return testNow }, func(v VerifiedSnapshot) error {
				verifies++
				_, err := VerifyProof(f.anchor, v, proof, request(ArtifactGrant), testNow)
				return err
			})
		requireOK(t, err)
		if got.Generation() != generation || reads != 1 || verifies != 1 || refreshes != 0 {
			t.Fatalf("unnecessary refresh: got=%d reads=%d verifies=%d refreshes=%d", got.Generation(), reads, verifies, refreshes)
		}
	}
}

func TestCurrentForEvidencePreservesCancellationAndReadFailures(t *testing.T) {
	for _, stage := range []string{"before-read", "initial-read", "reread", "during-refresh"} {
		t.Run(stage, func(t *testing.T) {
			f := fixture(t)
			peer := f.snapshot(t, 2, testNow, nil)
			proof := currentEvidenceProof(t, f, peer)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			want := errors.New("local trust unavailable")
			if stage == "before-read" || stage == "during-refresh" {
				want = context.Canceled
			}
			if stage == "before-read" {
				cancel()
			}
			reads, refreshes := 0, 0
			local := f.view
			got, err := CurrentForEvidence(ctx, f.anchor, proof.Evidence.Snapshot,
				func(context.Context) (VerifiedSnapshot, error) {
					reads++
					if stage == "initial-read" || (stage == "reread" && reads == 2) {
						return VerifiedSnapshot{}, want
					}
					return local, nil
				}, func(context.Context) error {
					refreshes++
					if stage == "during-refresh" {
						local = peer
						cancel()
					}
					return nil
				}, func() time.Time { return testNow }, func(v VerifiedSnapshot) error {
					_, err := VerifyProof(f.anchor, v, proof, request(ArtifactGrant), testNow)
					return err
				})
			if !errors.Is(err, want) || errors.Is(err, ErrTrustBehind) || got.Generation() != 0 {
				t.Fatalf("lost terminal context/read failure: got=%d err=%v", got.Generation(), err)
			}
			if (stage == "before-read" && reads != 0) || ((stage == "before-read" || stage == "initial-read") && refreshes != 0) {
				t.Fatalf("continued after terminal failure: reads=%d refreshes=%d", reads, refreshes)
			}
		})
	}
}
