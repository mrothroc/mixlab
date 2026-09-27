// Package discovery supplies untrusted address hints, never identities or
// capabilities. Callers must authenticate each endpoint before using it.
package discovery

import (
	"context"
	"fmt"
	"io"
	"net"
	"net/netip"
	"sort"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

type Service string

const (
	Node        Service = "_mixlab._tcp"
	Enrollment  Service = "_mixlab-enroll._tcp"
	Authority   Service = "_mixlab-trust._tcp"
	Version             = "1"
	MaxHints            = 256
	maxTXTBytes         = 900
)

func (s Service) valid() bool { return s == Node || s == Enrollment || s == Authority }

// Claims are presentation hints only. Keeping a closed schema prevents local
// callers from accidentally publishing catalogs, paths, jobs or credentials.
type Claims struct {
	Cluster         string
	Node            string
	RootFingerprint string
	Role            string
	Audience        string
	Policies        []string
	DisplayName     string
}

type Hint struct {
	Service  Service
	Endpoint string // host:port; no scheme, path, userinfo or query
	Claims   Claims
}

type Advertisement struct {
	Service  Service
	Instance string
	Port     int
	Claims   Claims
}

type Provider interface {
	Browse(context.Context, Service) ([]Hint, error)
	Advertise(context.Context, Advertisement) (io.Closer, error)
}

func label(s string) bool {
	if len(s) == 0 || len(s) > 64 {
		return false
	}
	for _, c := range s {
		switch {
		case c >= 'a' && c <= 'z', c >= 'A' && c <= 'Z', c >= '0' && c <= '9', c == '-', c == '_':
		default:
			return false
		}
	}
	return true
}

func (a Advertisement) TXT() ([]string, error) {
	if !a.Service.valid() || !label(a.Instance) || a.Port < 1 || a.Port > 65535 {
		return nil, fmt.Errorf("invalid discovery service, instance or port")
	}
	c := a.Claims
	if !label(c.Cluster) || len(c.DisplayName) > 96 || !utf8.ValidString(c.DisplayName) || strings.IndexFunc(c.DisplayName, unicode.IsControl) >= 0 {
		return nil, fmt.Errorf("invalid discovery claims")
	}
	txt := []string{"v=" + Version, "port=" + strconv.Itoa(a.Port), "cluster=" + c.Cluster}
	switch a.Service {
	case Node:
		if !label(c.Node) || c.RootFingerprint != "" || c.Role != "" || c.Audience != "" || len(c.Policies) != 0 {
			return nil, fmt.Errorf("invalid node discovery claims")
		}
		txt = append(txt, "node="+c.Node)
	case Enrollment:
		if c.Node != "" || !label(c.RootFingerprint) || (c.Role != "authority" && c.Role != "coordinator") || !label(c.Audience) || len(c.Policies) < 1 || len(c.Policies) > 3 {
			return nil, fmt.Errorf("invalid enrollment discovery claims")
		}
		policies := append([]string(nil), c.Policies...)
		sort.Strings(policies)
		for i, p := range policies {
			if p != "provisioned" && p != "trusted_lan" && p != "verified" {
				return nil, fmt.Errorf("invalid enrollment policy")
			}
			if i > 0 && policies[i-1] == p {
				return nil, fmt.Errorf("duplicate enrollment policy")
			}
		}
		txt = append(txt, "root="+c.RootFingerprint, "role="+c.Role, "audience="+c.Audience, "policies="+strings.Join(policies, ","))
	case Authority:
		if c.Node != "" || !label(c.RootFingerprint) || c.Role != "" || c.Audience != "" || len(c.Policies) != 0 {
			return nil, fmt.Errorf("invalid authority discovery claims")
		}
		txt = append(txt, "root="+c.RootFingerprint)
	}
	if c.DisplayName != "" {
		txt = append(txt, "name="+c.DisplayName)
	}
	return txt, nil
}

// DecodeTXT rejects unknown fields as well as ambiguous duplicates. Even valid
// claims must not influence certificate pinning or enrollment policy selection.
func DecodeTXT(service Service, port int, txt []string) (Claims, error) {
	var c Claims
	if len(txt) > 9 {
		return c, fmt.Errorf("too many discovery fields")
	}
	m := make(map[string]string, len(txt))
	total := 0
	for _, field := range txt {
		total += len(field) + 1
		if len(field) > 255 || total > maxTXTBytes {
			return c, fmt.Errorf("discovery TXT too large")
		}
		k, v, ok := strings.Cut(field, "=")
		if !ok || v == "" {
			return c, fmt.Errorf("invalid discovery field")
		}
		if _, exists := m[k]; exists {
			return c, fmt.Errorf("duplicate discovery field")
		}
		switch k {
		case "v", "port", "cluster", "node", "root", "role", "audience", "policies", "name":
		default:
			return c, fmt.Errorf("unknown discovery field")
		}
		m[k] = v
	}
	if m["v"] != Version || m["port"] != strconv.Itoa(port) {
		return c, fmt.Errorf("discovery version/port mismatch")
	}
	c = Claims{Cluster: m["cluster"], Node: m["node"], RootFingerprint: m["root"], Role: m["role"], Audience: m["audience"], DisplayName: m["name"]}
	if m["policies"] != "" {
		c.Policies = strings.Split(m["policies"], ",")
	}
	_, err := (Advertisement{Service: service, Instance: "validate", Port: port, Claims: c}).TXT()
	return c, err
}

func validEndpoint(endpoint string) error {
	if len(endpoint) > 320 {
		return fmt.Errorf("discovery endpoint too long")
	}
	host, port, err := net.SplitHostPort(endpoint)
	if err != nil || host == "" {
		return fmt.Errorf("invalid discovery endpoint")
	}
	p, err := strconv.Atoi(port)
	if err != nil || p < 1 || p > 65535 || strconv.Itoa(p) != port {
		return fmt.Errorf("invalid discovery port")
	}
	if ip, err := netip.ParseAddr(host); err == nil {
		if ip.IsUnspecified() || ip.IsMulticast() || ip.IsLinkLocalUnicast() && ip.Is6() && ip.Zone() == "" {
			return fmt.Errorf("unroutable discovery endpoint")
		}
		return nil
	}
	if len(host) > 253 {
		return fmt.Errorf("invalid discovery hostname")
	}
	for _, part := range strings.Split(host, ".") {
		if !label(part) || len(part) > 63 || strings.Contains(part, "_") || part[0] == '-' || part[len(part)-1] == '-' {
			return fmt.Errorf("invalid discovery hostname")
		}
	}
	return nil
}

func ordered(hints []Hint) []Hint {
	sort.SliceStable(hints, func(i, j int) bool { return hints[i].Endpoint < hints[j].Endpoint })
	out := make([]Hint, 0, len(hints))
	for _, h := range hints {
		if len(out) > 0 && out[len(out)-1].Endpoint == h.Endpoint {
			continue
		}
		h.Claims.Policies = append([]string(nil), h.Claims.Policies...)
		out = append(out, h)
	}
	return out
}

// Explicit is the multicast-off provider. It deliberately supplies no claims.
type Explicit struct{ Addresses map[Service][]string }

func (p Explicit) Browse(ctx context.Context, service Service) ([]Hint, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if !service.valid() || len(p.Addresses[service]) > MaxHints {
		return nil, fmt.Errorf("invalid explicit discovery selection")
	}
	var out []Hint
	for _, address := range p.Addresses[service] {
		if err := validEndpoint(address); err != nil {
			return nil, err
		}
		out = append(out, Hint{Service: service, Endpoint: address})
	}
	return ordered(out), nil
}

type noopCloser struct{}

func (noopCloser) Close() error { return nil }

func (Explicit) Advertise(ctx context.Context, a Advertisement) (io.Closer, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if _, err := a.TXT(); err != nil {
		return nil, err
	}
	return noopCloser{}, nil
}
