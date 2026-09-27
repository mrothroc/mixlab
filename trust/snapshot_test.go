package trust

import (
	"bytes"
	"strings"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/trust/internal/certificates"
)

func TestSnapshotValidationAndImmutableView(t *testing.T) {
	f := fixture(t)
	original, err := f.view.Bytes()
	requireOK(t, err)
	s := cloneSnapshot(t, f.view.signed)
	v, err := VerifySnapshot(f.anchor, s, testNow)
	requireOK(t, err)
	s.Payload.Issuers[0][0] ^= 1
	s.Payload.EligibleRoles[0] = "worker"
	s.Payload.Endpoints.Payload.URLs[0] = "https://changed.example"
	s.SignerCertificate[0] ^= 1
	b, err := v.Bytes()
	requireOK(t, err)
	if !bytes.Equal(original, b) {
		t.Fatal("verified snapshot aliases caller memory")
	}
	b[0] ^= 1
	if _, err := DecodeSnapshot(f.anchor, original, testNow); err != nil {
		t.Fatal(err)
	}
	if _, err := DecodeSnapshot(f.anchor, original, testNow.Add(SnapshotLifetime)); err == nil {
		t.Fatal("accepted expired snapshot")
	}
	other := fixture(t)
	if _, err := VerifySnapshot(other.anchor, f.view.signed, testNow); err == nil {
		t.Fatal("accepted other cluster")
	}
	for name, mutate := range map[string]func(*SignedSnapshot){
		"payload tamper":   func(s *SignedSnapshot) { s.Payload.Generation++ },
		"signature tamper": func(s *SignedSnapshot) { s.Signature[0] ^= 1 },
		"issuer as signer": func(s *SignedSnapshot) { s.SignerCertificate = f.issuer },
		"root as signer":   func(s *SignedSnapshot) { s.SignerCertificate = f.root },
		"endpoint tamper":  func(s *SignedSnapshot) { s.Payload.Endpoints.Payload.Audience = "other" },
	} {
		t.Run(name, func(t *testing.T) {
			s := cloneSnapshot(t, f.view.signed)
			mutate(&s)
			if _, err := VerifySnapshot(f.anchor, s, testNow); err == nil {
				t.Fatal("accepted altered signature or purpose")
			}
		})
	}
}

func TestInvalidSnapshotDoesNotUseSigningKey(t *testing.T) {
	f := fixture(t)
	_, issuerID, err := certificates.Parse(f.issuer)
	requireOK(t, err)
	for name, mutate := range map[string]func(*Snapshot){
		"version":                   func(p *Snapshot) { p.Version = "unknown" },
		"cluster":                   func(p *Snapshot) { p.Cluster = newID(t) },
		"zero generation":           func(p *Snapshot) { p.Generation = 0 },
		"future":                    func(p *Snapshot) { p.IssuedAt = testNow.Add(time.Hour).Unix(); p.ExpiresAt = p.IssuedAt + 60 },
		"expired":                   func(p *Snapshot) { p.ExpiresAt = testNow.Unix() },
		"long TTL":                  func(p *Snapshot) { p.ExpiresAt++ },
		"no issuer":                 func(p *Snapshot) { p.Issuers = nil },
		"duplicate issuer":          func(p *Snapshot) { p.Issuers = append(p.Issuers, p.Issuers[0]) },
		"snapshot signer as issuer": func(p *Snapshot) { p.Issuers = [][]byte{f.signer} },
		"unknown role":              func(p *Snapshot) { p.EligibleRoles = []Role{"unknown"} },
		"duplicate role":            func(p *Snapshot) { p.EligibleRoles = []Role{Controller, Controller} },
		"unordered roles":           func(p *Snapshot) { p.EligibleRoles = []Role{Controller, Authority} },
		"bad revocation": func(p *Snapshot) {
			p.Revocations = []Revocation{{Kind: "principal", ID: newID(t), Mode: "delete", Reason: "test", FirstGeneration: 1}}
		},
		"future revocation": func(p *Snapshot) {
			p.Revocations = []Revocation{{Kind: "principal", ID: newID(t), Mode: "prospective", Reason: "test", FirstGeneration: 2}}
		},
		"CA revocation": func(p *Snapshot) {
			p.Revocations = []Revocation{{Kind: "certificate", ID: issuerID.Serial, Mode: "compromise", Reason: "CA compromised", FirstGeneration: 1}}
		},
		"unsigned endpoints": func(p *Snapshot) { p.Endpoints.Signature = nil },
	} {
		t.Run(name, func(t *testing.T) {
			p := cloneSnapshot(t, f.view.signed).Payload
			mutate(&p)
			key := &countingSigner{Signer: f.signerKey}
			if _, err := SignSnapshot(f.anchor, p, f.signer, key, testNow); err == nil {
				t.Fatal("accepted invalid body")
			}
			if key.calls != 0 {
				t.Fatal("invalid body reached private signing key")
			}
			// Even a correctly signed forbidden body must be rejected.
			s := SignedSnapshot{Payload: p, SignerCertificate: f.signer}
			var err error
			s.Signature, err = sign(SnapshotVersion, snapshotBody(s), f.signerKey)
			requireOK(t, err)
			if _, err := VerifySnapshot(f.anchor, s, testNow); err == nil {
				t.Fatal("accepted signed invalid body")
			}
		})
	}
}

func TestAuthorityEndpointsPurposeAndValidation(t *testing.T) {
	f := fixture(t)
	for name, mutate := range map[string]func(*AuthorityEndpoints){
		"http":       func(p *AuthorityEndpoints) { p.URLs = []string{"http://authority.example"} },
		"userinfo":   func(p *AuthorityEndpoints) { p.URLs = []string{"https://user@authority.example"} },
		"query":      func(p *AuthorityEndpoints) { p.URLs = []string{"https://authority.example?token=x"} },
		"path":       func(p *AuthorityEndpoints) { p.URLs = []string{"https://authority.example/arbitrary"} },
		"fragment":   func(p *AuthorityEndpoints) { p.URLs = []string{"https://authority.example/#other"} },
		"duplicates": func(p *AuthorityEndpoints) { p.URLs = []string{"https://a.example", "https://a.example"} },
		"unordered":  func(p *AuthorityEndpoints) { p.URLs = []string{"https://b.example", "https://a.example"} },
		"empty":      func(p *AuthorityEndpoints) { p.URLs = nil },
		"audience":   func(p *AuthorityEndpoints) { p.Audience = "" },
		"cluster":    func(p *AuthorityEndpoints) { p.Cluster = newID(t) },
		"expired":    func(p *AuthorityEndpoints) { p.ExpiresAt = testNow.Unix() },
	} {
		t.Run(name, func(t *testing.T) {
			p := f.endpoints.Payload
			mutate(&p)
			key := &countingSigner{Signer: f.rootKey}
			if _, err := SignAuthorityEndpoints(f.anchor, p, key, testNow); err == nil {
				t.Fatal("accepted invalid endpoint body")
			}
			if key.calls != 0 {
				t.Fatal("invalid endpoints used root key")
			}
		})
	}
	if _, err := SignAuthorityEndpoints(f.anchor, f.endpoints.Payload, f.signerKey, testNow); err == nil {
		t.Fatal("snapshot key signed root endpoints")
	}
	if _, err := SignSnapshot(f.anchor, f.view.signed.Payload, f.issuer, f.issuerKey, testNow); err == nil {
		t.Fatal("issuer signed snapshot")
	}
	p := f.endpoints.Payload
	p.URLs = []string{"https://a.example"}
	e, err := SignAuthorityEndpoints(f.anchor, p, f.rootKey, testNow)
	requireOK(t, err)
	p.URLs[0] = "https://changed.example"
	requireOK(t, validateEndpoints(f.anchor, e, testNow))
}

func TestSnapshotAdvanceRetainsRevocationHistory(t *testing.T) {
	f := fixture(t)
	r := Revocation{Kind: "principal", ID: newID(t), Mode: "prospective", Reason: "operator retired principal", FirstGeneration: 2}
	next := f.snapshot(t, 2, testNow, []Revocation{r})
	v, err := f.view.Advance(next.signed, testNow)
	requireOK(t, err)
	if v.Generation() != 2 {
		t.Fatal("did not advance")
	}
	if _, err := v.Advance(next.signed, testNow); err == nil {
		t.Fatal("accepted replay")
	}
	if _, err := v.Advance(f.view.signed, testNow); err == nil {
		t.Fatal("accepted rollback")
	}
	for name, revs := range map[string][]Revocation{
		"removed":     nil,
		"backdated":   {{Kind: r.Kind, ID: r.ID, Mode: r.Mode, Reason: r.Reason, FirstGeneration: 1}},
		"moved later": {{Kind: r.Kind, ID: r.ID, Mode: r.Mode, Reason: r.Reason, FirstGeneration: 3}},
	} {
		t.Run(name, func(t *testing.T) {
			n := f.snapshot(t, 3, testNow, revs)
			if _, err := v.Advance(n.signed, testNow); err == nil {
				t.Fatal("accepted revocation rollback")
			}
		})
	}
	r.Mode = "compromise"
	n := f.snapshot(t, 3, testNow, []Revocation{r})
	v, err = v.Advance(n.signed, testNow)
	requireOK(t, err)
	r.Mode = "prospective"
	n = f.snapshot(t, 4, testNow, []Revocation{r})
	if _, err := v.Advance(n.signed, testNow); err == nil {
		t.Fatal("downgraded compromise")
	}
	// A client may miss intermediate generations, but not accept a tombstone
	// that contradicts a generation it already verified.
	r.FirstGeneration = 3
	n = f.snapshot(t, 4, testNow, []Revocation{r})
	_, err = f.view.Advance(n.signed, testNow)
	requireOK(t, err)
	r.FirstGeneration = 1
	n = f.snapshot(t, 4, testNow, []Revocation{r})
	if _, err := f.view.Advance(n.signed, testNow); err == nil {
		t.Fatal("new tombstone contradicts known generation")
	}
	later := testNow.Add(time.Hour)
	n = f.snapshot(t, 4, later, nil)
	_, err = f.view.Advance(n.signed, later)
	requireOK(t, err)
}

func TestStrictCanonicalSnapshotDecode(t *testing.T) {
	f := fixture(t)
	b, err := f.view.Bytes()
	requireOK(t, err)
	for _, invalid := range [][]byte{
		append([]byte(" "), b...),
		[]byte(strings.Replace(string(b), `"generation":1`, `"generation":1,"generation":1`, 1)),
		[]byte(strings.Replace(string(b), `"generation":1`, `"generation":null`, 1)),
		[]byte(strings.Replace(string(b), `"generation":1`, `"Generation":1`, 1)),
		append([]byte(`{"unknown":1,`), b[1:]...),
		make([]byte, MaxTrustBytes+1),
	} {
		if _, err := DecodeSnapshot(f.anchor, invalid, testNow); err == nil {
			t.Fatal("accepted noncanonical snapshot")
		}
	}
}

func TestSnapshotRejectsSharedIssuerAndSigningKey(t *testing.T) {
	f := fixture(t)
	der, err := certificates.Issue(f.anchor, f.root, f.rootKey, certificates.SnapshotSigner, "", "", f.issuerKey.Public(), testNow)
	requireOK(t, err)
	if _, err := SignSnapshot(f.anchor, f.view.signed.Payload, der, f.issuerKey, testNow); err == nil {
		t.Fatal("same key allowed certificate issuance and snapshot signing")
	}
}
