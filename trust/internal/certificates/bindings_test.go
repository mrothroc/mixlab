package certificates

import (
	"bytes"
	"crypto/ecdh"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/asn1"
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func boundFixture(t *testing.T) fixture {
	t.Helper()
	return setup(t)
}

func workload(t *testing.T, f fixture) WorkloadBinding {
	t.Helper()
	return WorkloadBinding{Version: WorkloadBindingVersion, Cluster: f.anchor.Cluster(), Role: Worker,
		Principal: id(t), Participant: id(t), Run: id(t), Job: id(t), Lease: id(t), ManifestHash: strings.Repeat("cd", 32), Attempt: id(t), Audience: "ddp/control",
		IssuedAt: testNow.Unix(), ExpiresAt: testNow.Add(30 * time.Minute).Unix(),
		Group: id(t), Generation: 1, MembershipHash: strings.Repeat("ab", 32), Member: id(t), Rank: 0}
}

func TestBoundProfiles(t *testing.T) {
	f := boundFixture(t)
	encryptionKey, err := ecdh.X25519().GenerateKey(rand.Reader)
	must(t, encryptionKey, err)
	nodeID := id(t)
	der, err := IssueNode(f.anchor, f.issuer, f.issuerKey, nodeID, f.leafKey.Public(), encryptionKey.PublicKey().Bytes(), testNow)
	must(t, der, err)
	c, got, err := Verify(f.anchor, der, f.issuer, Principal, testNow)
	must(t, c, err)
	if got.Role != Node || got.Principal != nodeID || !bytes.Equal(got.EnvelopeKey, encryptionKey.PublicKey().Bytes()) {
		t.Fatal("node binding lost")
	}
	if _, _, err := Parse(der); err != nil {
		t.Fatal("bound certificate required external configuration:", err)
	}
	b := workload(t, f)
	der, err = IssueWorkload(f.anchor, f.issuer, f.issuerKey, f.leafKey.Public(), b, testNow.Add(time.Hour), testNow)
	must(t, der, err)
	c, got, err = Verify(f.anchor, der, f.issuer, Principal, testNow)
	must(t, c, err)
	if got.Role != Worker || got.Workload == nil || *got.Workload != b || len(c.UnhandledCriticalExtensions) != 0 {
		t.Fatal("workload binding lost")
	}
	if _, _, err := Verify(f.anchor, der, f.issuer, Principal, time.Unix(b.ExpiresAt, 0)); err == nil {
		t.Fatal("accepted expired workload")
	}
}

func TestWorkloadBindingRejectsMalformedCertificates(t *testing.T) {
	f := boundFixture(t)
	parent, _, _ := Parse(f.issuer)
	for _, name := range []string{"version", "cluster", "role", "principal", "participant", "run", "job", "lease", "manifest", "attempt", "audience", "issue", "expiry", "generation", "hash", "group", "member", "rank", "missing", "noncritical", "duplicate", "unknown-critical", "unknown-field"} {
		t.Run(name, func(t *testing.T) {
			b := workload(t, f)
			c, err := template(Principal, f.anchor.Cluster(), Worker, b.Principal, f.leafKey.Public(), testNow)
			must(t, c, err)
			c.NotBefore, c.NotAfter = time.Unix(b.IssuedAt, 0), time.Unix(b.ExpiresAt, 0)
			switch name {
			case "version":
				b.Version = "v0"
			case "cluster":
				b.Cluster = id(t)
			case "role":
				b.Role = Controller
			case "principal":
				b.Principal = id(t)
			case "participant":
				b.Participant = ""
			case "run":
				b.Run = ""
			case "job":
				b.Job = ""
			case "lease":
				b.Lease = ""
			case "manifest":
				b.ManifestHash = ""
			case "attempt":
				b.Attempt = ""
			case "audience":
				b.Audience = "invalid\naudience"
			case "issue":
				b.IssuedAt++
			case "expiry":
				b.ExpiresAt++
			case "generation":
				b.Generation = 0
			case "hash":
				b.MembershipHash = strings.Repeat("AB", 32)
			case "group":
				b.Group = ""
			case "member":
				b.Member = ""
			case "rank":
				b.Rank = -1
			}
			raw, err := json.Marshal(b)
			must(t, raw, err)
			if name == "unknown-field" {
				raw = append(raw[:len(raw)-1], []byte(",\"extra\":0}")...)
			}
			c.URIs = append(c.URIs, bindingURI(Worker, raw))
			switch name {
			case "missing":
				c.URIs = c.URIs[:1]
			case "noncritical":
				setTestSAN(t, c, false)
			case "duplicate":
				c.URIs = append(c.URIs, c.URIs[1])
			case "unknown-critical":
				c.ExtraExtensions = append(c.ExtraExtensions, pkix.Extension{Id: asn1.ObjectIdentifier{1, 3, 6, 1, 4, 1, 32473, 1, 99}, Critical: true, Value: []byte{5, 0}})
			}
			der, err := x509.CreateCertificate(rand.Reader, c, parent, f.leafKey.Public(), f.issuerKey)
			must(t, der, err)
			if _, _, err := Verify(f.anchor, der, f.issuer, Principal, testNow); err == nil {
				t.Fatal("malformed binding accepted")
			}
		})
	}
}

func TestBoundIssuanceLimits(t *testing.T) {
	f := boundFixture(t)
	b := workload(t, f)
	for _, name := range []string{"job-deadline", "ttl", "future", "issuer", "key-reuse"} {
		t.Run(name, func(t *testing.T) {
			copy := b
			a, k, deadline := f.anchor, f.leafKey, testNow.Add(time.Hour)
			switch name {
			case "job-deadline":
				deadline = testNow.Add(time.Minute)
			case "ttl":
				copy.ExpiresAt = testNow.Add(WorkloadLifetime + time.Second).Unix()
				deadline = testNow.Add(2 * WorkloadLifetime)
			case "future":
				copy.IssuedAt++
			case "issuer":
				copy.ExpiresAt = testNow.Add(IssuerLifetime + time.Hour).Unix()
				deadline = testNow.Add(2 * IssuerLifetime)
			case "key-reuse":
				k = f.issuerKey
			}
			if _, err := IssueWorkload(a, f.issuer, f.issuerKey, k.Public(), copy, deadline, testNow); err == nil {
				t.Fatal("bad issuance accepted")
			}
		})
	}
	if _, err := IssueNode(f.anchor, f.issuer, f.issuerKey, id(t), f.leafKey.Public(), make([]byte, 32), testNow); err == nil {
		t.Fatal("low-order key issued")
	}
}

func TestNodeBindingMalformed(t *testing.T) {
	f := boundFixture(t)
	parent, _, _ := Parse(f.issuer)
	k, err := ecdh.X25519().GenerateKey(rand.Reader)
	must(t, k, err)
	for _, name := range []string{"missing", "noncritical", "version", "low-order", "unknown-field", "wrong-role", "tls-reuse"} {
		t.Run(name, func(t *testing.T) {
			role := Node
			if name == "wrong-role" {
				role = Controller
			}
			c, err := template(Principal, f.anchor.Cluster(), role, id(t), f.leafKey.Public(), testNow)
			must(t, c, err)
			b := NodeBinding{NodeBindingVersion, k.PublicKey().Bytes()}
			switch name {
			case "version":
				b.Version = "v0"
			case "low-order":
				b.EnvelopeKey = make([]byte, 32)
			case "tls-reuse":
				b.EnvelopeKey = f.leafKey[32:]
			}
			raw, err := json.Marshal(b)
			must(t, raw, err)
			if name == "unknown-field" {
				raw = append(raw[:len(raw)-1], []byte(",\"extra\":1}")...)
			}
			c.URIs = append(c.URIs, bindingURI(Node, raw))
			if name == "missing" {
				c.URIs = c.URIs[:1]
			}
			if name == "noncritical" {
				setTestSAN(t, c, false)
			}
			der, err := x509.CreateCertificate(rand.Reader, c, parent, f.leafKey.Public(), f.issuerKey)
			must(t, der, err)
			if _, _, err := Verify(f.anchor, der, f.issuer, Principal, testNow); err == nil {
				t.Fatal("malformed node accepted")
			}
		})
	}
}

func setTestSAN(t *testing.T, c *x509.Certificate, critical bool, extra ...asn1.RawValue) {
	t.Helper()
	var names []asn1.RawValue
	for _, u := range c.URIs {
		names = append(names, asn1.RawValue{Class: 2, Tag: 6, Bytes: []byte(u.String())})
	}
	names = append(names, extra...)
	raw, err := asn1.Marshal(names)
	must(t, raw, err)
	c.ExtraExtensions = []pkix.Extension{{Id: subjectAltNameOID(), Critical: critical, Value: raw}}
}
