// Package enrollmenttls owns ephemeral enrollment-channel evidence. It does
// not approve requests, authenticate proposed roots, or expose any listener.
package enrollmenttls

import (
	"context"
	"crypto/rand"
	"crypto/tls"
	"fmt"
	"net"
	"net/netip"
	"sync"
	"time"
)

const Protocol = "mixlab-enrollment-v1"
const exporterLabel = "EXPORTER-mixlab-enrollment-v1"

// Network is observed locally from the completed connection, never decoded
// from a request's claims. Interface owns the local destination IP; it is not
// proof of physical ingress on a multihomed host.
type Network struct {
	Peer      netip.Addr
	Local     netip.Addr
	Interface string
}

// Channel is ephemeral and one-use for exporter derivation. Its owner must
// Close it when the connection ends, the request terminates, or the window
// closes. A canceled/deadline-expired context also prevents use immediately.
// It does not expose the connection or store/log exporter material.
type Channel struct {
	mu           sync.Mutex
	ctx          context.Context
	conn         *tls.Conn
	network      Network
	used, closed bool
	id           [32]byte
}

func New(ctx context.Context, conn *tls.Conn) (*Channel, error) {
	if ctx == nil || conn == nil {
		return nil, fmt.Errorf("enrollment requires a live bounded TLS connection")
	}
	deadline, ok := ctx.Deadline()
	if !ok || time.Until(deadline) <= 0 || time.Until(deadline) > time.Hour {
		return nil, fmt.Errorf("enrollment connection deadline must be within one hour")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	state := conn.ConnectionState()
	if !state.HandshakeComplete || state.Version != tls.VersionTLS13 || state.DidResume || state.NegotiatedProtocol != Protocol {
		return nil, fmt.Errorf("enrollment requires a full dedicated TLS 1.3 handshake")
	}
	peer, err := address(conn.RemoteAddr())
	if err != nil {
		return nil, err
	}
	local, err := address(conn.LocalAddr())
	if err != nil {
		return nil, err
	}
	iface, err := localAddressInterface(local)
	if err != nil {
		return nil, err
	}
	var id [32]byte
	if _, err := rand.Read(id[:]); err != nil {
		return nil, err
	}
	return &Channel{ctx: ctx, conn: conn, network: Network{Peer: peer, Local: local, Interface: iface}, id: id}, nil
}

// EvidenceID is a local connection identity, never a network bearer token.
func (c *Channel) EvidenceID() [32]byte { return c.id }

func (c *Channel) Observe() (netip.Addr, string, error) {
	n, err := c.Network()
	return n.Peer, n.Interface, err
}

func address(a net.Addr) (netip.Addr, error) {
	host, _, err := net.SplitHostPort(a.String())
	if err != nil {
		return netip.Addr{}, fmt.Errorf("enrollment peer/local address is not a TCP address")
	}
	ip, err := netip.ParseAddr(host)
	if err != nil || ip.IsUnspecified() || ip.IsMulticast() {
		return netip.Addr{}, fmt.Errorf("invalid observed enrollment address")
	}
	return ip.Unmap().WithZone(""), nil
}

func localAddressInterface(local netip.Addr) (string, error) {
	interfaces, err := net.Interfaces()
	if err != nil {
		return "", err
	}
	name := ""
	for _, iface := range interfaces {
		addresses, err := iface.Addrs()
		if err != nil {
			return "", err
		}
		for _, a := range addresses {
			prefix, err := netip.ParsePrefix(a.String())
			if err != nil {
				continue
			}
			if prefix.Addr().Unmap().WithZone("") != local {
				continue
			}
			if name != "" && name != iface.Name {
				return "", fmt.Errorf("ambiguous enrollment local-address interface")
			}
			name = iface.Name
		}
	}
	if name == "" {
		return "", fmt.Errorf("enrollment local-address interface not found")
	}
	return name, nil
}

// ValidateListener rejects wildcard listeners and interface/address mismatch
// before opening an enrollment window. This is endpoint selection, not an
// OS packet-ingress filter. Every completed channel independently observes it.
func ValidateListener(l net.Listener, wantInterface string) error {
	if l == nil {
		return fmt.Errorf("enrollment listener required")
	}
	local, err := address(l.Addr())
	if err != nil {
		return err
	}
	iface, err := localAddressInterface(local)
	if err != nil {
		return err
	}
	if wantInterface != "" && wantInterface != iface {
		return fmt.Errorf("enrollment listener address does not belong to selected interface")
	}
	return nil
}

func (c *Channel) Network() (Network, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return Network{}, fmt.Errorf("enrollment channel is closed")
	}
	if err := c.ctx.Err(); err != nil {
		return Network{}, err
	}
	return c.network, nil
}

// Bind may succeed only once, even for the same context. The caller must clear
// the returned bytes after computing its SAS/evidence digest. Reconnecting
// requires a new channel, request ID, nonces, and operator confirmations.
func (c *Channel) Bind(contextDigest [32]byte) ([]byte, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed || c.used {
		return nil, fmt.Errorf("enrollment exporter is closed or already consumed")
	}
	if err := c.ctx.Err(); err != nil {
		return nil, err
	}
	if contextDigest == [32]byte{} {
		return nil, fmt.Errorf("empty enrollment request context")
	}
	c.used = true
	state := c.conn.ConnectionState()
	return state.ExportKeyingMaterial(exporterLabel, contextDigest[:], 32)
}

// Close invalidates evidence and closes the connection without waiting on an
// unresponsive peer. The application's connection/request lifecycle must invoke
// it on EOF as well; a silent disconnected peer is bounded by the context TTL.
func (c *Channel) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return nil
	}
	c.closed = true
	_ = c.conn.SetDeadline(time.Now())
	return c.conn.Close()
}
