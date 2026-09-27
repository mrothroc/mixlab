package discovery

import (
	"context"
	"net"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/hashicorp/mdns"
)

func nodeAdvertisement() Advertisement {
	return Advertisement{Service: Node, Instance: "opaque-node", Port: 7443, Claims: Claims{Cluster: "cluster-1", Node: "node-1", DisplayName: "Lab Mac"}}
}

func TestDiscoveryClosedTXT(t *testing.T) {
	for _, a := range []Advertisement{
		nodeAdvertisement(),
		{Service: Enrollment, Instance: "enroll", Port: 7444, Claims: Claims{Cluster: "cluster-1", RootFingerprint: strings.Repeat("a", 64), Role: "coordinator", Audience: "enrollment", Policies: []string{"provisioned", "verified"}}},
		{Service: Authority, Instance: "trust", Port: 7445, Claims: Claims{Cluster: "cluster-1", RootFingerprint: strings.Repeat("a", 64)}},
	} {
		txt, err := a.TXT()
		if err != nil {
			t.Fatal(err)
		}
		claims, err := DecodeTXT(a.Service, a.Port, txt)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(claims, a.Claims) {
			t.Fatalf("roundtrip: %+v != %+v", claims, a.Claims)
		}
		for _, extra := range []string{"v=1", "dataset=secret", "path=/tmp/data", "token=secret", "name=" + strings.Repeat("x", 256)} {
			if _, err := DecodeTXT(a.Service, a.Port, append(append([]string(nil), txt...), extra)); err == nil {
				t.Fatalf("accepted %q", extra)
			}
		}
		if _, err := DecodeTXT(a.Service, a.Port+1, txt); err == nil {
			t.Fatal("accepted mismatched SRV/TXT port")
		}
	}
	a := nodeAdvertisement()
	a.Claims.DisplayName = "name\nspoof"
	if _, err := a.TXT(); err == nil {
		t.Fatal("control character accepted")
	}
	a = nodeAdvertisement()
	a.Claims.Policies = []string{"verified"}
	if _, err := a.TXT(); err == nil {
		t.Fatal("node advertised enrollment policy")
	}
}

func TestExplicitDiscovery(t *testing.T) {
	p := Explicit{Addresses: map[Service][]string{Node: {"node.example:7443", "127.0.0.1:7443", "node.example:7443"}}}
	h, err := p.Browse(context.Background(), Node)
	if err != nil {
		t.Fatal(err)
	}
	if len(h) != 2 || h[0].Endpoint != "127.0.0.1:7443" || h[0].Claims.Cluster != "" {
		t.Fatalf("unexpected hints %+v", h)
	}
	for _, bad := range []string{"http://node:7443", "user@node:7443", "node:7443/path", "0.0.0.0:7443", "224.0.0.251:7443", "[fe80::1]:7443", "node:0", "node:07443", "node:65536"} {
		if _, err := (Explicit{Addresses: map[Service][]string{Node: {bad}}}).Browse(context.Background(), Node); err == nil {
			t.Fatalf("accepted %q", bad)
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := p.Browse(ctx, Node); err == nil {
		t.Fatal("ignored cancellation")
	}
}

func TestMDNSEntryHints(t *testing.T) {
	a := nodeAdvertisement()
	txt, _ := a.TXT()
	e := &mdns.ServiceEntry{Host: "untrusted.example.", Port: a.Port, InfoFields: txt, AddrV4: net.ParseIP("192.168.1.10"), AddrV6IPAddr: &net.IPAddr{IP: net.ParseIP("fe80::1"), Zone: "en0"}}
	h := entryHints(Node, e)
	if len(h) != 2 || h[0].Endpoint != "192.168.1.10:7443" || h[1].Endpoint != "[fe80::1%en0]:7443" {
		t.Fatalf("bad hints %+v", h)
	}
	e.InfoFields = append(append([]string(nil), txt...), "credential=secret")
	if len(entryHints(Node, e)) != 0 {
		t.Fatal("accepted unexpected TXT")
	}
	if len(entryHints(Node, nil)) != 0 {
		t.Fatal("accepted nil entry")
	}
}

func TestFakeDiscoveryLifetimeAndCopies(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var f Fake
	a := nodeAdvertisement()
	first, err := f.Advertise(ctx, a)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = first.Close() }()
	a.Claims.Node = "node-2"
	second, err := f.Advertise(ctx, a)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = second.Close() }()
	h, _ := f.Browse(ctx, Node)
	if len(h) != 1 || h[0].Claims.Node != "node-1" {
		t.Fatal(h)
	}
	if err := first.Close(); err != nil {
		t.Fatal(err)
	}
	h, _ = f.Browse(ctx, Node)
	if len(h) != 1 || h[0].Claims.Node != "node-2" {
		t.Fatal(h)
	}
	cancel()
	deadline := time.Now().Add(time.Second)
	for {
		h, _ = f.Browse(context.Background(), Node)
		if len(h) == 0 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("advertisement leaked after cancellation")
		}
		time.Sleep(time.Millisecond)
	}
}

func TestMDNSRejectsBeforeOpeningSockets(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := (MDNS{}).Browse(ctx, Node); err == nil {
		t.Fatal("ignored cancellation")
	}
	if _, err := (MDNS{}).Advertise(context.Background(), nodeAdvertisement()); err == nil {
		t.Fatal("accepted no local addresses")
	}
	if _, err := (MDNS{Timeout: time.Minute}).Browse(context.Background(), Node); err == nil {
		t.Fatal("unbounded browse")
	}
}
