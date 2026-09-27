package certificates

import (
	"crypto/ecdh"
	"crypto/rand"
	"crypto/x509"
	"encoding/asn1"
	"encoding/json"
	"strings"
	"testing"
)

func TestBindingURIRejectsAmbiguity(t *testing.T) {
	f := setup(t)
	parent, _, err := Parse(f.issuer)
	must(t, parent, err)
	key, err := ecdh.X25519().GenerateKey(rand.Reader)
	must(t, key, err)
	for _, name := range []string{"wrong-role", "unknown-uri-version", "swapped", "padding", "fragment", "query", "invalid-base64", "extra-uri", "hidden-name", "duplicate-json", "oversized", "noncanonical-json"} {
		t.Run(name, func(t *testing.T) {
			c, err := template(Principal, f.anchor.Cluster(), Node, id(t), f.leafKey.Public(), testNow)
			must(t, c, err)
			raw, err := json.Marshal(NodeBinding{NodeBindingVersion, key.PublicKey().Bytes()})
			must(t, raw, err)
			if name == "duplicate-json" {
				raw = append(raw[:len(raw)-1], []byte(`,"version":"mixlab_node_binding_v1"}`)...)
			}
			if name == "oversized" {
				raw = []byte(strings.Repeat("a", 5000))
			}
			if name == "noncanonical-json" {
				raw = append([]byte(" "), raw...)
			}
			u := bindingURI(Node, raw)
			c.URIs = append(c.URIs, u)
			switch name {
			case "wrong-role":
				c.URIs[1] = bindingURI(Worker, raw)
			case "unknown-uri-version":
				u.Opaque = strings.Replace(u.Opaque, ":v1:", ":v2:", 1)
			case "swapped":
				c.URIs[0], c.URIs[1] = c.URIs[1], c.URIs[0]
			case "padding":
				u.Opaque += "="
			case "fragment":
				u.Fragment = "ignored"
			case "query":
				u.RawQuery = "ignored=true"
			case "invalid-base64":
				u.Opaque += "!"
			case "extra-uri":
				c.URIs = append(c.URIs, c.URIs[0])
			case "hidden-name":
				setTestSAN(t, c, true, asn1.RawValue{Class: 2, Tag: 8, Bytes: []byte{42, 3}})
			}
			der, err := x509.CreateCertificate(rand.Reader, c, parent, f.leafKey.Public(), f.issuerKey)
			must(t, der, err)
			if _, _, err := Verify(f.anchor, der, f.issuer, Principal, testNow); err == nil {
				t.Fatal("accepted ambiguous binding URI")
			}
		})
	}
}

func TestBoundSANWireContract(t *testing.T) {
	f := setup(t)
	b := workload(t, f)
	der, err := IssueWorkload(f.anchor, f.issuer, f.issuerKey, f.leafKey.Public(), b, testNow.Add(WorkloadLifetime), testNow)
	must(t, der, err)
	c, err := x509.ParseCertificate(der)
	must(t, c, err)
	if len(c.URIs) != 2 || !strings.HasPrefix(c.URIs[1].String(), "urn:mixlab:binding:v1:worker:") {
		t.Fatal("binding URI wire format changed")
	}
	found := false
	for _, e := range c.Extensions {
		if e.Id.Equal(subjectAltNameOID()) {
			found = true
			if !e.Critical {
				t.Fatal("SAN must be critical")
			}
		}
		if strings.HasPrefix(e.Id.String(), "1.3.6.1.4.1.") {
			t.Fatal("private-enterprise namespace leaked into certificate")
		}
	}
	if !found {
		t.Fatal("SAN missing")
	}
}
