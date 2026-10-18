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
	"errors"
	"net"
	"net/http"
	"path/filepath"
	"strings"
	"sync/atomic"
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
	workloadIssuanceOverManagedTLS(t, 0, false)
}

func TestWorkloadAuthorizedIssuanceOverManagedTLS(t *testing.T) {
	workloadIssuanceOverManagedTLS(t, 0, false, true)
}

func TestWorkloadIssuanceSlowTLSAndCallerDeadline(t *testing.T) {
	for _, bounded := range []bool{false, true} {
		name := "twenty_second_issuance"
		if bounded {
			name = "caller_deadline"
		}
		t.Run(name, func(t *testing.T) {
			workloadIssuanceOverManagedTLS(t, 20*time.Second, bounded)
		})
	}
}

// IssueRemoteWorkload opens a snapshot connection, then a separate mutual-TLS
// issuance connection. Delay only the latter so the old issuance-only 15s cap
// is tested independently of the public trust refresh budget.
type workloadIssuanceDelayListener struct{ *delayedAuthorityListener }

func (l *workloadIssuanceDelayListener) Accept() (net.Conn, error) {
	c, err := l.Listener.Accept()
	if err != nil {
		return nil, err
	}
	if l.accepted.Add(1) == 2 && l.delay > 0 {
		return &delayedAuthorityConn{Conn: c, listener: l.delayedAuthorityListener, closed: make(chan struct{})}, nil
	}
	return c, nil
}

func workloadIssuanceOverManagedTLS(t *testing.T, delay time.Duration, callerBounded bool, authorized ...bool) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	check(t, err)
	endpoint := "https://" + l.Addr().String()
	p := initializedAt(t, endpoint)
	check(t, InitializeAuthority(ctx, p, testNow))
	a, err := OpenAuthority(ctx, p, testNow)
	check(t, err)
	defer func() { _ = a.Close() }()
	var currentTime atomic.Int64
	currentTime.Store(testNow.Unix())
	clock := func() time.Time { return time.Unix(currentTime.Load(), 0) }
	slow := &workloadIssuanceDelayListener{&delayedAuthorityListener{Listener: l, delay: delay, started: make(chan struct{})}}
	done := make(chan error, 1)
	go func() { done <- ServeAuthority(ctx, slow, a, clock) }()
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
	scope := workload.Scope{Cluster: a.Anchor.Cluster(), Controller: cs.Principal, Node: node.Approval.Principal, Lease: id(), Run: id(), Job: id(), Attempt: id(), Group: id(), Member: id(), Workload: id(), Generation: 1, Rank: 0, MembershipHash: strings.Repeat("a", 64), ManifestHash: strings.Repeat("b", 64), Created: testNow.Unix(), AdmitUntil: testNow.Add(30 * time.Minute).Unix(), Deadline: testNow.Add(3 * 24 * time.Hour).Unix()}
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
	issue := func(call context.Context) (workload.Result, error) {
		return IssueRemoteWorkload(call, controller, endpoint, grant, clock)
	}
	route := workloadIssueRoute
	if len(authorized) != 0 && authorized[0] {
		// The original request and grant both expire before first authority
		// acceptance. Fresh authorization must not replace either object.
		currentTime.Store(testNow.Add(trust.SnapshotLifetime + time.Minute).Unix())
		v, err = a.Current(ctx, clock())
		check(t, err)
		if _, err := trust.VerifyProof(a.Anchor, v, grant.Proof, sign, clock()); err == nil {
			t.Fatal("fixture did not expire original grant evidence")
		}
		nq, err := signed.ReauthorizationRequest()
		check(t, err)
		np, err := trust.SignPrincipalProof(a.Anchor, node.Chain, nk, v, nq, clock())
		check(t, err)
		gq, err := grant.ReauthorizationRequest()
		check(t, err)
		gp, err := trust.SignPrincipalProof(a.Anchor, cs.Chain, ck, v, gq, clock())
		check(t, err)
		envelope := workload.AuthorizedGrant{Version: workload.AuthorizedGrantVersion, Grant: grant, NodeAuthorization: workload.RequestAuthorization{Version: workload.RequestAuthorizationVersion, Proof: np}, ControllerAuthorization: gp}
		issue = func(call context.Context) (workload.Result, error) {
			return IssueRemoteAuthorizedWorkload(call, controller, endpoint, envelope, clock)
		}
		route = workloadIssueAuthorizedRoute
	}
	call := ctx
	if callerBounded {
		var stop context.CancelFunc
		call, stop = context.WithTimeout(ctx, 3*time.Second)
		defer stop()
	}
	started := time.Now()
	first, err := issue(call)
	if callerBounded {
		if !errors.Is(err, context.DeadlineExceeded) || call.Err() != context.DeadlineExceeded || ctx.Err() != nil || len(first.Chain) != 0 {
			t.Fatalf("issuance ignored caller deadline or published credentials: err=%v caller=%v parent=%v", err, call.Err(), ctx.Err())
		}
		select {
		case <-slow.started:
		default:
			t.Fatal("caller deadline did not reach the issuance TLS connection")
		}
		if slow.accepted.Load() != 2 {
			t.Fatalf("timed-out issuance retried: connections=%d", slow.accepted.Load())
		}
		return
	}
	check(t, err)
	if slow.accepted.Load() != 2 || time.Since(started) < delay {
		t.Fatalf("issuance bypassed delay or retried internally: connections=%d elapsed=%v", slow.accepted.Load(), time.Since(started))
	}
	again, err := issue(ctx)
	check(t, err)
	if !bytes.Equal(first.Chain[0], again.Chain[0]) {
		t.Fatal("TLS retry reissued certificate")
	}
	check(t, workload.ValidateResult(a.Anchor, grant, again, clock()))
	check(t, credential.Install(ctx, grant, first, v, clock()))
	check(t, credential.Install(ctx, grant, again, v, clock()))
	chain, handle, err := credential.Identity(v, clock())
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
	res, err := (&http.Client{Transport: raw, Timeout: time.Second}).Post(endpoint+route, "application/json", strings.NewReader("{}"))
	check(t, err)
	_ = res.Body.Close()
	if res.StatusCode != http.StatusUnauthorized {
		t.Fatal("anonymous workload issuance", res.StatusCode)
	}
	_, err = a.Snapshots.Revoke(ctx, "principal", scope.Node, "prospective", "test", clock())
	check(t, err)
	if _, err := issue(ctx); err == nil {
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
