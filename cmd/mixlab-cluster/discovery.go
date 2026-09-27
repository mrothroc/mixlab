package main

import (
	"context"
	"fmt"
	"io"
	"net"
	"net/url"
	"time"

	"github.com/mrothroc/mixlab/discovery"
)

func discoverEnrollmentEndpoint(ctx context.Context, provider discovery.Provider) (string, error) {
	if provider == nil {
		return "", fmt.Errorf("discovery provider required")
	}
	ctx, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()
	hints, err := provider.Browse(ctx, discovery.Enrollment)
	if err != nil {
		return "", err
	}
	if err := ctx.Err(); err != nil {
		return "", err
	}
	if len(hints) > discovery.MaxHints {
		return "", fmt.Errorf("too many enrollment hints; specify -enrollment-coordinator with discovery off")
	}
	addresses := make([]string, 0, len(hints))
	for _, h := range hints {
		if h.Service != discovery.Enrollment {
			return "", fmt.Errorf("unexpected discovery service")
		}
		addresses = append(addresses, h.Endpoint)
	}
	// Reuse endpoint validation/deduplication, discarding every TXT claim.
	clean, err := (discovery.Explicit{Addresses: map[discovery.Service][]string{discovery.Enrollment: addresses}}).Browse(ctx, discovery.Enrollment)
	if err != nil {
		return "", err
	}
	if len(clean) != 1 {
		return "", fmt.Errorf("discovery found %d enrollment sources; specify -enrollment-coordinator with discovery off", len(clean))
	}
	return (&url.URL{Scheme: "https", Host: clean[0].Endpoint}).String(), nil
}

// Use the actual concrete listener address, never enumerate and publish all
// local interfaces. Discovery conveys no approval or certificate authority.
func advertiseListener(ctx context.Context, mode string, l net.Listener, service discovery.Service, instance string, claims discovery.Claims) (io.Closer, error) {
	addr, ok := l.Addr().(*net.TCPAddr)
	if !ok || addr.IP == nil || addr.IP.IsUnspecified() {
		return nil, fmt.Errorf("discovery requires a concrete TCP listener")
	}
	a := discovery.Advertisement{Service: service, Instance: instance, Port: addr.Port, Claims: claims}
	switch mode {
	case "off":
		return (discovery.Explicit{}).Advertise(ctx, a)
	case "mdns":
		var iface *net.Interface
		if addr.Zone != "" {
			var err error
			iface, err = net.InterfaceByName(addr.Zone)
			if err != nil {
				return nil, err
			}
		}
		return (discovery.MDNS{IPs: []net.IP{addr.IP}, Interface: iface}).Advertise(ctx, a)
	default:
		return nil, fmt.Errorf("advertisement must be mdns or off")
	}
}

func nodeDiscoveryClaims(cluster, node string) discovery.Claims {
	return discovery.Claims{Cluster: cluster, Node: node}
}
