package discovery

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"net"
	"os"
	"testing"
	"time"
)

func TestMDNSLive(t *testing.T) {
	name := os.Getenv("MIXLAB_TEST_MDNS_INTERFACE")
	if name == "" {
		t.Skip("set MIXLAB_TEST_MDNS_INTERFACE to an authorized local multicast interface")
	}
	iface, err := net.InterfaceByName(name)
	if err != nil {
		t.Fatal(err)
	}
	addresses, err := iface.Addrs()
	if err != nil {
		t.Fatal(err)
	}
	var ip net.IP
	for _, addr := range addresses {
		n, ok := addr.(*net.IPNet)
		if ok && n.IP.To4() != nil {
			ip = n.IP
			break
		}
	}
	if ip == nil {
		t.Fatal("test requires an IPv4 address on selected interface")
	}
	var nonce [16]byte
	if _, err := rand.Read(nonce[:]); err != nil {
		t.Fatal(err)
	}
	a := nodeAdvertisement()
	a.Instance = hex.EncodeToString(nonce[:])
	a.Claims.Node = a.Instance
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	p := MDNS{Interface: iface, IPs: []net.IP{ip}, Timeout: time.Second}
	ad, err := p.Advertise(ctx, a)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := ad.Close(); err != nil {
			t.Error(err)
		}
	}()
	for ctx.Err() == nil {
		hints, err := p.Browse(ctx, Node)
		if err != nil {
			t.Fatal(err)
		}
		for _, h := range hints {
			if h.Claims.Node == a.Claims.Node {
				t.Logf("discovered test instance at %s on %s", h.Endpoint, name)
				return
			}
		}
	}
	t.Fatal("live mDNS advertisement was not discovered")
}
