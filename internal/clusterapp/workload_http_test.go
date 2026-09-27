package clusterapp

import (
	"bytes"
	"context"
	"crypto"
	"crypto/ecdh"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/tls"
	"encoding/hex"
	"encoding/json"
	"net"
	"net/http"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/nodecredentials"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/enrollment"
	"github.com/mrothroc/mixlab/trust/principal"
	"github.com/mrothroc/mixlab/trust/workload"
)

func TestWorkloadIssuanceOverManagedTLS(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	check(t, err)
	endpoint := "https://" + l.Addr().String()
	p := initializedAt(t, endpoint)
	check(t, InitializeAuthority(ctx, p, testNow))
	a, err := OpenAuthority(ctx, p, testNow)
	check(t, err)
	defer func() { _ = a.Close() }()
	clock := func() time.Time { return testNow }
	done := make(chan error, 1)
	go func() { done <- ServeAuthority(ctx, l, a, clock) }()
	defer func() { cancel(); check(t, <-done) }()
	cp, err := statehome.Resolve(statehome.Options{ExactDir: filepath.Join(filepath.Dir(p.Dir()), "controller")}, statehome.Context{Kind: statehome.Principal})
	check(t, err)
	controller, err := principal.Open(cp, testNow)
	check(t, err)
	defer func() { _ = controller.Close() }()
	cs, ck, err := controller.Active(testNow)
	check(t, err)
	v, err := a.Current(ctx, testNow)
	check(t, err)
	i, err := a.Enrollment.Invite(ctx, enrollment.NodeEnrollment, endpoint, "authority", time.Minute, v, testNow)
	check(t, err)
	defer i.Clear()
	_, nk, err := ed25519.GenerateKey(rand.Reader)
	check(t, err)
	x, err := ecdh.X25519().GenerateKey(rand.Reader)
	check(t, err)
	q, err := enrollment.NewRequest(i, nk, x.PublicKey().Bytes(), testNow)
	check(t, err)
	node, err := a.Enrollment.Consume(ctx, q, i.Secret, v, testNow)
	check(t, err)
	id := func() string {
		b := make([]byte, 16)
		_, err := rand.Read(b)
		check(t, err)
		return hex.EncodeToString(b)
	}
	scope := workload.Scope{Cluster: a.Anchor.Cluster(), Controller: cs.Principal, Node: node.Approval.Principal, Lease: id(), Run: id(), Job: id(), Attempt: id(), Group: id(), Member: id(), Workload: id(), Generation: 1, Rank: 0, MembershipHash: strings.Repeat("a", 64), ManifestHash: strings.Repeat("b", 64), Created: testNow.Unix(), AdmitUntil: testNow.Add(time.Minute).Unix(), Deadline: testNow.Add(3 * 24 * time.Hour).Unix()}
	transportPath, err := statehome.Resolve(statehome.Options{ExactDir: filepath.Join(filepath.Dir(p.Dir()), "node-job-key")}, statehome.Context{Kind: statehome.Agent})
	check(t, err)
	credential, err := nodecredentials.InitializeTransport(ctx, transportPath, a.Anchor, scope, "file")
	check(t, err)
	defer func() { _ = credential.Close() }()
	signed, err := credential.PrepareRequest(ctx, func(_ context.Context, q trust.SignRequest) (trust.SignedProof, error) {
		return trust.SignPrincipalProof(a.Anchor, node.Chain, nk, v, q, testNow)
	}, v, testNow)
	check(t, err)
	check(t, credential.Close())
	credential, err = nodecredentials.OpenTransport(transportPath, a.Anchor, scope)
	check(t, err)
	retry, err := credential.PrepareRequest(ctx, func(context.Context, trust.SignRequest) (trust.SignedProof, error) {
		t.Fatal("retry regenerated key request")
		return trust.SignedProof{}, nil
	}, v, testNow)
	check(t, err)
	if !bytes.Equal(retry.Request.CSR, signed.Request.CSR) {
		t.Fatal("reopened request changed key")
	}
	if _, _, err := credential.Identity(v, testNow); err == nil {
		t.Fatal("unissued identity escaped")
	}
	sign, err := signed.GrantRequest()
	check(t, err)
	grantProof, err := trust.SignPrincipalProof(a.Anchor, cs.Chain, ck, v, sign, testNow)
	check(t, err)
	grant := workload.Grant{Request: signed, Proof: grantProof}
	first, err := IssueRemoteWorkload(ctx, controller, endpoint, grant, clock)
	check(t, err)
	again, err := IssueRemoteWorkload(ctx, controller, endpoint, grant, clock)
	check(t, err)
	if !bytes.Equal(first.Chain[0], again.Chain[0]) {
		t.Fatal("TLS retry reissued certificate")
	}
	check(t, workload.ValidateResult(a.Anchor, grant, again, testNow))
	check(t, credential.Install(ctx, grant, first, v, testNow))
	check(t, credential.Install(ctx, grant, again, v, testNow))
	chain, handle, err := credential.Identity(v, testNow)
	check(t, err)
	if !bytes.Equal(chain[0], first.Chain[0]) {
		t.Fatal("installed certificate changed")
	}
	message := []byte("protected workload proof")
	signature, err := handle.Sign(rand.Reader, message, crypto.Hash(0))
	check(t, err)
	public, ok := handle.Public().(ed25519.PublicKey)
	if !ok || !ed25519.Verify(public, message, signature) {
		t.Fatal("workload signer does not satisfy Ed25519 TLS contract")
	}
	// An anonymous snapshot client cannot use the workload route.
	raw := &http.Transport{TLSClientConfig: &tls.Config{InsecureSkipVerify: true, MinVersion: tls.VersionTLS13, NextProtos: []string{"http/1.1"}, Time: clock}}
	defer raw.CloseIdleConnections()
	res, err := (&http.Client{Transport: raw, Timeout: time.Second}).Post(endpoint+workloadIssueRoute, "application/json", strings.NewReader("{}"))
	check(t, err)
	_ = res.Body.Close()
	if res.StatusCode != http.StatusUnauthorized {
		t.Fatal("anonymous workload issuance", res.StatusCode)
	}
	_, err = a.Snapshots.Revoke(ctx, "principal", scope.Node, "prospective", "test", testNow)
	check(t, err)
	if _, err := IssueRemoteWorkload(ctx, controller, endpoint, grant, clock); err == nil {
		t.Fatal("revoked node received workload credential")
	}
	// Model a crash after terminal intent but before key deletion. An old
	// signer must already be unusable, even though its protected key exists.
	rawRecord, err := transportPath.ReadFile("transport-credential.json")
	check(t, err)
	var record map[string]json.RawMessage
	check(t, json.Unmarshal(rawRecord, &record))
	if string(record["stage"]) != `"installed"` {
		t.Fatal("unexpected transport stage")
	}
	// Preserve canonical field order instead of re-marshaling an unordered map.
	intent := bytes.Replace(rawRecord, []byte(`"stage":"installed"`), []byte(`"stage":"destroying"`), 1)
	check(t, transportPath.CompareAndSwap("transport-credential.json", rawRecord, intent))
	if _, err := handle.Sign(rand.Reader, message, crypto.Hash(0)); err == nil {
		t.Fatal("terminal intent left retained signer usable")
	}
	check(t, credential.Destroy(ctx))
	check(t, credential.Destroy(ctx))
	if _, _, err := credential.Identity(v, testNow); err == nil {
		t.Fatal("destroyed credential revived")
	}
	if _, err := handle.Sign(rand.Reader, []byte("test"), crypto.Hash(0)); err == nil {
		t.Fatal("deleted protected signer remained usable")
	}
	if _, err := credential.PrepareRequest(ctx, func(context.Context, trust.SignRequest) (trust.SignedProof, error) {
		t.Fatal("recreated destroyed key")
		return trust.SignedProof{}, nil
	}, v, testNow); err == nil {
		t.Fatal("destroyed request resumed")
	}
}
