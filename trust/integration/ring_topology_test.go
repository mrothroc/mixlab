package integration

import (
	"context"
	"crypto/tls"
	"net"
	"os"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/distributed"
	"github.com/mrothroc/mixlab/grouptransport"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/transport/ringproxy"
	"github.com/mrothroc/mixlab/transport/ringtls"
	"github.com/mrothroc/mixlab/trust"
)

func TestRingRelayRejectsOtherAdmittedRankAndDuplicateStream(t *testing.T) {
	f := newTLSFixture(t)
	node := id(t)
	m := signedJobFixture(t, f, node, nodeagent.Lease{ID: id(t), Run: id(t)}).Manifest
	members := append(append([]distributed.DDPGroupMember(nil), m.Membership.OrderedMembers...), distributed.DDPGroupMember{MemberID: id(t), Rank: 2})
	var err error
	m.Membership, err = distributed.NewDDPGroupMembership(m.Membership.RunID, m.Membership.GroupID, 1, "ring", members)
	check(t, err)
	m.Members = append(m.Members, nodejob.Member{Node: id(t), MemberID: members[2].MemberID, Rank: 2})
	signed, identities := transportFixture(t, f, m)
	public, local, shim, unused := listenRingTest(t), listenRingTest(t), listenRingTest(t), listenRingTest(t)
	signed.Plan.Members[0].Endpoint = public.Addr().String()
	signed.Plan.LoopbackPorts = []int{local.Addr().(*net.TCPAddr).Port, shim.Addr().(*net.TCPAddr).Port, unused.Addr().(*net.TCPAddr).Port}
	check(t, shim.Close())
	check(t, unused.Close())
	q, err := signed.Plan.SigningRequest()
	check(t, err)
	signed.Proof, err = trust.SignPrincipalProof(f.a, f.client.Chain, f.client.Key, f.view, q, f.now)
	check(t, err)
	actor, err := trust.AuthenticatePrincipal(f.a, f.view, f.client.Chain, f.now)
	check(t, err)
	accepted, err := grouptransport.Accept(f.a, f.view, actor, m, identities[0].Chain, signed, f.now)
	check(t, err)
	r, err := ringproxy.New(ringproxy.Options{Accepted: accepted, LocalRank: 0, Identity: identities[0], CurrentTrust: func(context.Context, time.Time) (trust.Anchor, trust.VerifiedSnapshot, error) {
		return f.a, f.view, nil
	}, Clock: f.clock, Public: public, ConnectTimeout: time.Second, MaxConnections: 2})
	check(t, err)
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- r.Run(ctx) }()
	defer func() {
		cancel()
		_ = r.Close()
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			t.Error("relay did not stop")
		}
	}()
	var connections atomic.Int32
	echoRingTest(t, local, &connections)
	dial := func(rank int) (net.Conn, error) {
		p, err := ringtls.New(identities[rank], func(c [][]byte, now time.Time) error {
			return trust.VerifyWorkload(f.a, f.view, c, signed.Plan.Members[0].Binding, now)
		}, f.clock)
		check(t, err)
		return tls.DialWithDialer(&net.Dialer{Timeout: time.Second}, "tcp", public.Addr().String(), p.ClientConfig())
	}
	rejected := func(c net.Conn, err error) {
		if err != nil {
			return
		}
		defer func() { _ = c.Close() }()
		_ = c.SetDeadline(time.Now().Add(time.Second))
		_, _ = c.Write([]byte("forbidden"))
		var b [1]byte
		_, err = c.Read(b[:])
		if err == nil || os.IsTimeout(err) {
			t.Error("connection not promptly rejected", err)
		}
	}
	rejected(dial(1)) // Rank 1 is admitted, but rank 2 is rank 0's predecessor.
	if connections.Load() != 0 {
		t.Fatal("wrong rank reached MLX")
	}
	good, err := dial(2)
	check(t, err)
	defer func() { _ = good.Close() }()
	check(t, good.SetDeadline(time.Now().Add(3*time.Second)))
	_, err = good.Write([]byte("x"))
	check(t, err)
	var b [1]byte
	_, err = good.Read(b[:])
	check(t, err)
	rejected(dial(2))
	if connections.Load() != 1 {
		t.Fatal("duplicate admitted stream reached MLX")
	}
	// Silent anonymous handshakes must not displace the established stream.
	for range 2 {
		c, err := net.DialTimeout("tcp", public.Addr().String(), time.Second)
		check(t, err)
		defer func() { _ = c.Close() }()
	}
	_, err = good.Write([]byte("y"))
	check(t, err)
	_, err = good.Read(b[:])
	check(t, err)
}
