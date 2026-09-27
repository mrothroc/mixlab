package enrollment

import (
	"context"
	"fmt"
	"net/netip"
	"sort"
	"time"

	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/internal/certificates"
)

type Policy string

const TrustedLAN Policy = "trusted_lan"
const Verified Policy = "verified"

type WindowOptions struct {
	Policy                                 Policy
	Endpoint, Audience                     string
	Purposes                               []Purpose
	TTL                                    time.Duration
	MaxApprovals, MaxPending, MaxPerMinute int
	Interface                              string
	CIDRs                                  []string
}

type Window struct {
	Version        string    `json:"version"`
	ID             string    `json:"id"`
	Cluster        string    `json:"cluster"`
	Fingerprint    string    `json:"fingerprint"`
	Policy         Policy    `json:"policy"`
	Endpoint       string    `json:"endpoint"`
	Audience       string    `json:"audience"`
	Purposes       []Purpose `json:"purposes"`
	Created        int64     `json:"created"`
	Expires        int64     `json:"expires"`
	MaxApprovals   int       `json:"max_approvals"`
	MaxPending     int       `json:"max_pending"`
	MaxPerMinute   int       `json:"max_per_minute"`
	Interface      string    `json:"interface"`
	CIDRs          []string  `json:"cidrs"`
	AuditPrincipal string    `json:"audit_principal"`
	Closed         bool      `json:"closed"`
}

const windowVersion = "mixlab_enrollment_window_v1"

func privatePrefix(s string) (netip.Prefix, error) {
	p, err := netip.ParsePrefix(s)
	if err != nil || p != p.Masked() || p.Addr().Is4In6() {
		return netip.Prefix{}, fmt.Errorf("invalid canonical LAN CIDR")
	}
	for _, allowed := range []string{"10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "169.254.0.0/16", "127.0.0.0/8", "fc00::/7", "fe80::/10", "::1/128"} {
		v := netip.MustParsePrefix(allowed)
		if p.Addr().BitLen() == v.Addr().BitLen() && p.Bits() >= v.Bits() && v.Contains(p.Addr()) {
			return p, nil
		}
	}
	return netip.Prefix{}, fmt.Errorf("trusted LAN CIDR must be entirely private, link-local, or loopback")
}

func (w Window) validate() error {
	if w.Version != windowVersion || !certificates.ValidID(w.ID) || !certificates.ValidID(w.Cluster) || !digestOK(w.Fingerprint) || !certificates.ValidID(w.AuditPrincipal) ||
		!validEndpoint(w.Endpoint) || !validAudience(w.Audience) || (w.Policy != TrustedLAN && w.Policy != Verified) ||
		w.Created <= 0 || w.Expires <= w.Created || w.Expires-w.Created > 3600 || w.MaxApprovals < 1 || w.MaxApprovals > maxEntries || w.MaxPending < 1 || w.MaxPending > 64 || w.MaxPerMinute < 1 || w.MaxPerMinute > 64 || len(w.Purposes) == 0 || len(w.Purposes) > 3 {
		return fmt.Errorf("invalid enrollment window")
	}
	for i, p := range w.Purposes {
		if p.role() == "" || (i > 0 && w.Purposes[i-1] >= p) {
			return fmt.Errorf("invalid window purpose allowlist")
		}
	}
	if w.Policy == TrustedLAN && (len(w.Purposes) != 1 || w.Purposes[0] != NodeEnrollment) {
		return fmt.Errorf("trusted LAN permits node enrollment only")
	}
	if w.Policy == TrustedLAN || w.Interface != "" || len(w.CIDRs) > 0 {
		if !validAudience(w.Interface) || len(w.CIDRs) == 0 || len(w.CIDRs) > 16 {
			return fmt.Errorf("trusted LAN requires node-only purpose, local-address interface and bounded CIDRs")
		}
		for i, cidr := range w.CIDRs {
			if _, err := privatePrefix(cidr); err != nil {
				return err
			}
			if i > 0 && w.CIDRs[i-1] >= cidr {
				return fmt.Errorf("invalid CIDR ordering")
			}
		}
	}
	return nil
}

func (s *Service) OpenWindow(ctx context.Context, o WindowOptions, v trust.VerifiedSnapshot, now time.Time) (Window, error) {
	if o.TTL == 0 {
		o.TTL = 10 * time.Minute
	}
	if o.MaxPending == 0 {
		o.MaxPending = 16
	}
	if o.MaxPerMinute == 0 {
		o.MaxPerMinute = 16
	}
	if len(o.Purposes) == 0 {
		o.Purposes = []Purpose{NodeEnrollment}
	}
	if o.TTL < time.Second || o.TTL > time.Hour {
		return Window{}, fmt.Errorf("invalid enrollment window TTL")
	}
	var result Window
	err := s.store.WithProcessLock(ctx, lockFile, func() error {
		old, r, err := s.loadForOperation(v, now)
		if err != nil {
			return err
		}
		if len(r.Windows) >= maxEntries {
			return fmt.Errorf("enrollment window journal capacity reached")
		}
		id, err := certificates.NewID()
		if err != nil {
			return err
		}
		w := Window{Version: windowVersion, ID: id, Cluster: s.a.Anchor.Cluster(), Fingerprint: s.a.Anchor.Fingerprint(), Policy: o.Policy, Endpoint: o.Endpoint, Audience: o.Audience,
			Purposes: append([]Purpose(nil), o.Purposes...), Created: now.Unix(), Expires: now.Add(o.TTL).Unix(), MaxApprovals: o.MaxApprovals, MaxPending: o.MaxPending, MaxPerMinute: o.MaxPerMinute,
			Interface: o.Interface, CIDRs: append([]string(nil), o.CIDRs...), AuditPrincipal: s.owner}
		sort.Slice(w.Purposes, func(i, j int) bool { return w.Purposes[i] < w.Purposes[j] })
		sort.Strings(w.CIDRs)
		if err := w.validate(); err != nil {
			return err
		}
		for _, purpose := range w.Purposes {
			if !roleAllowed(v, purpose.role()) {
				return fmt.Errorf("window role is not eligible")
			}
		}
		r.Windows = append(r.Windows, w)
		r.Sequence++
		if err := s.write(old, r); err != nil {
			return err
		}
		result = w
		return nil
	})
	return result, err
}

// CloseWindow is an administrator-local port, never a network approval route.
func (s *Service) CloseWindow(ctx context.Context, id string, v trust.VerifiedSnapshot, now time.Time) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.store.WithProcessLock(ctx, lockFile, func() error {
		old, r, err := s.loadForOperation(v, now)
		if err != nil {
			return err
		}
		for i := range r.Windows {
			if r.Windows[i].ID == id {
				if r.Windows[i].Closed {
					return nil
				}
				r.Windows[i].Closed = true
				for n := range r.Interactive {
					e := &r.Interactive[n]
					if e.Request.Request.Window == id {
						if e.Stage == "pending" || e.Stage == "approved" {
							e.Stage = "expired"
						}
						delete(s.live, e.ID)
					}
				}
				r.Sequence++
				return s.write(old, r)
			}
		}
		return fmt.Errorf("unknown enrollment window")
	})
}
