package trust

import (
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/trust/internal/certificates"
)

func TestPrincipalPurposeMatrix(t *testing.T) {
	f := fixture(t)
	policies := map[Role]map[Purpose]bool{
		Authority:   {EnrollmentApproval: true, EnrollmentReceipt: true},
		Controller:  {NodeJob: true, SecureTransportPlan: true, CredentialEnvelopePurpose: true, WorkloadGrant: true},
		Coordinator: {RunPlan: true, RunCommit: true, RoundGrant: true, RecoveryGrant: true, ArtifactGrant: true},
	}
	purposes := []Purpose{EnrollmentApproval, EnrollmentReceipt, NodeJob, SecureTransportPlan, CredentialEnvelopePurpose, WorkloadKeyRequest, WorkloadGrant, RunPlan, RunCommit, RoundGrant, RecoveryGrant, ArtifactGrant, "certificate", "snapshot", "unknown"}
	for role, allowed := range policies {
		p := f.principal(t, role)
		for _, purpose := range purposes {
			t.Run(string(role)+"/"+string(purpose), func(t *testing.T) {
				key := &countingSigner{Signer: p.key}
				r := request(purpose)
				proof, err := SignPrincipalProof(f.anchor, p.chain, key, f.view, r, testNow)
				if !allowed[purpose] {
					if err == nil || key.calls != 0 {
						t.Fatal("forbidden purpose reached private key")
					}
					return
				}
				requireOK(t, err)
				accepted, err := VerifyProof(f.anchor, f.view, proof, r, testNow)
				requireOK(t, err)
				if len(accepted.Digest) != 64 || accepted.Generation != 1 || key.calls != 1 {
					t.Fatal("invalid acceptance record")
				}
				b, err := canonical(proof)
				requireOK(t, err)
				roundTrip, err := DecodeProof(b)
				requireOK(t, err)
				_, err = VerifyProof(f.anchor, f.view, roundTrip, r, testNow)
				requireOK(t, err)
			})
		}
	}
}

func TestProofBindsEveryContextAndEvidenceField(t *testing.T) {
	f := fixture(t)
	p := f.principal(t, Controller)
	r := request(NodeJob)
	proof, err := SignPrincipalProof(f.anchor, p.chain, p.key, f.view, r, testNow)
	requireOK(t, err)
	for name, mutate := range map[string]func(*SignedProof){
		"digest":    func(p *SignedProof) { p.Request.Digest = strings.Repeat("cd", 32) },
		"purpose":   func(p *SignedProof) { p.Request.Purpose = SecureTransportPlan },
		"context":   func(p *SignedProof) { p.Request.Context = "job/attempt-2" },
		"audience":  func(p *SignedProof) { p.Request.Audience = "other-node" },
		"version":   func(p *SignedProof) { p.Request.Version = "v0" },
		"cluster":   func(p *SignedProof) { p.Evidence.Cluster = newID(t) },
		"principal": func(p *SignedProof) { p.Evidence.Principal = newID(t) },
		"role":      func(p *SignedProof) { p.Evidence.Role = Authority },
		"serial":    func(p *SignedProof) { p.Evidence.Serial = newID(t) },
		"algorithm": func(p *SignedProof) { p.Evidence.Algorithm = "none" },
		"timestamp": func(p *SignedProof) { p.Evidence.SignedAt-- },
		"chain":     func(p *SignedProof) { p.Evidence.Chain = p.Evidence.Chain[:2] },
		"signature": func(p *SignedProof) { p.Signature[0] ^= 1 },
		"snapshot":  func(p *SignedProof) { p.Evidence.Snapshot.Payload.Generation++ },
	} {
		t.Run(name, func(t *testing.T) {
			q := cloneProof(t, proof)
			mutate(&q)
			if _, err := VerifyProof(f.anchor, f.view, q, r, testNow); err == nil {
				t.Fatal("accepted altered proof")
			}
		})
	}
	for name, mutate := range map[string]func(*SignRequest){
		"purpose":  func(r *SignRequest) { r.Purpose = SecureTransportPlan },
		"audience": func(r *SignRequest) { r.Audience = "other-node" },
		"context":  func(r *SignRequest) { r.Context = "different-job" },
		"digest":   func(r *SignRequest) { r.Digest = strings.Repeat("cd", 32) },
	} {
		t.Run("expected/"+name, func(t *testing.T) {
			want := r
			mutate(&want)
			if _, err := VerifyProof(f.anchor, f.view, proof, want, testNow); err == nil {
				t.Fatal("accepted cross-context proof replay")
			}
		})
	}
	// Caller mutation after signing cannot rewrite the returned evidence.
	p.chain[0][0] ^= 1
	_, err = VerifyProof(f.anchor, f.view, proof, r, testNow)
	requireOK(t, err)
}

func TestProofRejectsInvalidSigningInputs(t *testing.T) {
	f := fixture(t)
	p := f.principal(t, Controller)
	for name, mutate := range map[string]func(*SignRequest){
		"empty digest":      func(r *SignRequest) { r.Digest = "" },
		"uppercase digest":  func(r *SignRequest) { r.Digest = strings.ToUpper(r.Digest) },
		"empty audience":    func(r *SignRequest) { r.Audience = "" },
		"control context":   func(r *SignRequest) { r.Context = "line\nbreak" },
		"oversized context": func(r *SignRequest) { r.Context = strings.Repeat("x", 257) },
	} {
		t.Run(name, func(t *testing.T) {
			r := request(NodeJob)
			mutate(&r)
			key := &countingSigner{Signer: p.key}
			if _, err := SignPrincipalProof(f.anchor, p.chain, key, f.view, r, testNow); err == nil || key.calls != 0 {
				t.Fatal("invalid request reached key")
			}
		})
	}
	for name, check := range map[string]func() error{
		"wrong key": func() error {
			_, e := SignPrincipalProof(f.anchor, p.chain, f.rootKey, f.view, request(NodeJob), testNow)
			return e
		},
		"no key": func() error {
			_, e := SignPrincipalProof(f.anchor, p.chain, nil, f.view, request(NodeJob), testNow)
			return e
		},
		"stale view": func() error {
			_, e := SignPrincipalProof(f.anchor, p.chain, p.key, f.view, request(NodeJob), testNow.Add(SnapshotLifetime))
			return e
		},
		"zero view": func() error {
			_, e := SignPrincipalProof(f.anchor, p.chain, p.key, VerifiedSnapshot{}, request(NodeJob), testNow)
			return e
		},
	} {
		t.Run(name, func(t *testing.T) {
			if check() == nil {
				t.Fatal("accepted invalid signer/view")
			}
		})
	}
}

func TestHistoricalProofRequiresExactOwnerCommit(t *testing.T) {
	f := fixture(t)
	p := f.principal(t, Controller)
	r := request(NodeJob)
	proof, err := SignPrincipalProof(f.anchor, p.chain, p.key, f.view, r, testNow)
	requireOK(t, err)
	accepted, err := VerifyProof(f.anchor, f.view, proof, r, testNow)
	requireOK(t, err)
	j := memoryJournal{accepted.Digest: accepted.Generation}
	later := testNow.Add(31 * 24 * time.Hour)
	v := f.snapshot(t, 2, later, nil)
	if _, err := VerifyProof(f.anchor, v, proof, r, later); err == nil {
		t.Fatal("expired proof newly accepted")
	}
	requireOK(t, VerifyHistoricalProof(f.anchor, v, proof, r, j, later))
	for name, journal := range map[string]CommitJournal{
		"none": nil, "empty": memoryJournal{}, "zero": memoryJournal{accepted.Digest: 0},
		"future": memoryJournal{accepted.Digest: 3}, "error": errorJournal{},
	} {
		t.Run(name, func(t *testing.T) {
			if VerifyHistoricalProof(f.anchor, v, proof, r, journal, later) == nil {
				t.Fatal("accepted without durable commit")
			}
		})
	}
	// A key holder can lie about time and re-sign, but cannot fabricate the
	// owning context's commit for the new exact proof bytes.
	forged := cloneProof(t, proof)
	forged.Evidence.SignedAt = testNow.Add(-time.Hour).Unix()
	forged.Signature, err = sign(ProofVersion, proofBody(forged), p.key)
	requireOK(t, err)
	if VerifyHistoricalProof(f.anchor, v, forged, r, j, later) == nil {
		t.Fatal("backdating forged a historical acceptance")
	}
	if VerifyHistoricalProof(f.anchor, v, proof, r, j, later.Add(SnapshotLifetime)) == nil {
		t.Fatal("history used expired current trust")
	}
}

type errorJournal struct{}

func (errorJournal) AcceptedGeneration(string) (uint64, bool, error) {
	return 0, false, errors.New("journal unavailable")
}

func TestProspectiveAndCompromiseRevocation(t *testing.T) {
	for _, kind := range []string{"principal", "certificate"} {
		t.Run(kind, func(t *testing.T) {
			f := fixture(t)
			p := f.principal(t, Controller)
			r := request(NodeJob)
			proof, err := SignPrincipalProof(f.anchor, p.chain, p.key, f.view, r, testNow)
			requireOK(t, err)
			accepted, err := VerifyProof(f.anchor, f.view, proof, r, testNow)
			requireOK(t, err)
			id := p.id.Principal
			if kind == "certificate" {
				id = p.id.Serial
			}
			rev := Revocation{Kind: kind, ID: id, Mode: "prospective", Reason: "principal retired", FirstGeneration: 2}
			v := f.snapshot(t, 2, testNow, []Revocation{rev})
			if _, err := VerifyProof(f.anchor, v, proof, r, testNow); err == nil {
				t.Fatal("new acceptance after revocation")
			}
			if _, err := SignPrincipalProof(f.anchor, p.chain, p.key, v, r, testNow); err == nil {
				t.Fatal("revoked key signed")
			}
			requireOK(t, VerifyHistoricalProof(f.anchor, v, proof, r, memoryJournal{accepted.Digest: 1}, testNow))
			if VerifyHistoricalProof(f.anchor, v, proof, r, memoryJournal{accepted.Digest: 2}, testNow) == nil {
				t.Fatal("post-revocation commit survived")
			}
			rev.Mode = "compromise"
			v = f.snapshot(t, 3, testNow, []Revocation{rev})
			if VerifyHistoricalProof(f.anchor, v, proof, r, memoryJournal{accepted.Digest: 1}, testNow) == nil {
				t.Fatal("compromised history survived")
			}
		})
	}
}

func TestCurrentTrustControlsAcceptanceGenerationAndEligibility(t *testing.T) {
	f := fixture(t)
	p := f.principal(t, Controller)
	r := request(NodeJob)
	proof, err := SignPrincipalProof(f.anchor, p.chain, p.key, f.view, r, testNow)
	requireOK(t, err)
	v := f.snapshot(t, 2, testNow, nil)
	a, err := VerifyProof(f.anchor, v, proof, r, testNow)
	requireOK(t, err)
	if a.Generation != 2 {
		t.Fatal("acceptance trusted signer's older generation")
	}
	payload := cloneSnapshot(t, v.signed).Payload
	payload.EligibleRoles = []Role{Authority}
	s, err := SignSnapshot(f.anchor, payload, f.signer, f.signerKey, testNow)
	requireOK(t, err)
	v, err = VerifySnapshot(f.anchor, s, testNow)
	requireOK(t, err)
	if _, err := VerifyProof(f.anchor, v, proof, r, testNow); err == nil {
		t.Fatal("ineligible role accepted")
	}
	// Historical verification retains prior authorization, subject to explicit
	// revocation, rather than silently changing committed job authority.
	requireOK(t, VerifyHistoricalProof(f.anchor, v, proof, r, memoryJournal{a.Digest: 2}, testNow))
}

func TestSnapshotKeyCannotMasqueradeAsPrincipal(t *testing.T) {
	f := fixture(t)
	der, err := certificates.Issue(f.anchor, f.issuer, f.issuerKey, certificates.Principal, Controller, newID(t), f.signerKey.Public(), testNow)
	requireOK(t, err)
	chain := [][]byte{der, f.issuer, f.root}
	if _, err := SignPrincipalProof(f.anchor, chain, f.signerKey, f.view, request(NodeJob), testNow); err == nil {
		t.Fatal("snapshot key granted application signing")
	}
}

func TestStrictProofDecode(t *testing.T) {
	f := fixture(t)
	p := f.principal(t, Controller)
	proof, err := SignPrincipalProof(f.anchor, p.chain, p.key, f.view, request(NodeJob), testNow)
	requireOK(t, err)
	b, err := canonical(proof)
	requireOK(t, err)
	for _, invalid := range [][]byte{
		append(b, '\n'),
		[]byte(strings.Replace(string(b), `"purpose":"node_job"`, `"purpose":"node_job","purpose":"node_job"`, 1)),
		append([]byte(`{"extra":0,`), b[1:]...),
		make([]byte, MaxTrustBytes+1),
	} {
		if _, err := DecodeProof(invalid); err == nil {
			t.Fatal("accepted noncanonical proof")
		}
	}
}

func FuzzDecodeProof(f *testing.F) {
	f.Add([]byte(`{}`))
	f.Add([]byte(`null`))
	f.Add([]byte(`{"request":{}}`))
	f.Fuzz(func(t *testing.T, b []byte) { _, _ = DecodeProof(b) })
}
