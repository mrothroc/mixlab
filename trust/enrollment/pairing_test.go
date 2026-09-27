package enrollment

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/identity"
)

func TestPairingCanonicalFormula(t *testing.T) {
	c := PairingContext{Version: PairingVersion, Fingerprint: strings.Repeat("ab", 32), Audience: "test", Purpose: NodeEnrollment, Role: trust.Node, RequestID: strings.Repeat("01", 16), RequestHash: strings.Repeat("cd", 32), ClientNonce: bytes.Repeat([]byte{1}, 32), ServerNonce: bytes.Repeat([]byte{2}, 32)}
	b, err := json.Marshal(c)
	if err != nil {
		t.Fatal(err)
	}
	contextDigest := sha256.Sum256(b)
	got, err := c.Digest()
	if err != nil || got != contextDigest {
		t.Fatal("request context formula", err)
	}
	exporter := bytes.Repeat([]byte{3}, 32)
	input := append([]byte("mixlab-enrollment-sas-v1"), exporter...)
	input = append(input, contextDigest[:]...)
	digest := sha256.Sum256(input)
	phrase, evidence, err := PairingPresentation(c, exporter)
	if err != nil || phrase != identity.Confirmation(digest) || evidence != hex.EncodeToString(digest[:]) {
		t.Fatal("SAS formula", err)
	}
	for _, field := range []string{"fingerprint", "audience", "purpose", "role", "id", "hash", "client-nonce", "server-nonce"} {
		changed := c
		switch field {
		case "fingerprint":
			changed.Fingerprint = "bad"
		case "audience":
			changed.Audience = ""
		case "purpose":
			changed.Purpose = "external-worker-bootstrap"
		case "role":
			changed.Role = trust.Authority
		case "id":
			changed.RequestID = "bad"
		case "hash":
			changed.RequestHash = "bad"
		case "client-nonce":
			changed.ClientNonce = make([]byte, 32)
		case "server-nonce":
			changed.ServerNonce = nil
		}
		if _, err := changed.Digest(); err == nil {
			t.Fatal("accepted invalid", field)
		}
	}
	if _, _, err := PairingPresentation(c, make([]byte, 32)); err == nil {
		t.Fatal("empty exporter accepted")
	}
}
