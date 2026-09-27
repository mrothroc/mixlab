package enrollment

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/json"
	"fmt"
	"net/netip"
	"time"

	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/internal/certificates"
)

// ChannelPort comes only from the live bootstrap transport, never a decoded
// request. Exporter material is consumed once and cleared before persistence.
type ChannelPort interface {
	EvidenceID() [32]byte
	Observe() (netip.Addr, string, error)
	Bind([32]byte) ([]byte, error)
}

type interactiveEntry struct {
	ID              string                   `json:"id"`
	Stage           string                   `json:"stage"`
	Submitted       int64                    `json:"submitted"`
	Request         SignedInteractiveRequest `json:"request"`
	Pairing         PairingContext           `json:"pairing"`
	PairingDigest   string                   `json:"pairing_digest"`
	Peer            string                   `json:"peer"`
	Interface       string                   `json:"interface"`
	ClientConfirmed bool                     `json:"client_confirmed"`
	AdminConfirmed  bool                     `json:"admin_confirmed"`
	Result          *InteractiveResult       `json:"result"`
}

type InteractiveResult struct {
	Evidence InteractiveEvidence  `json:"evidence"`
	Approval Approval             `json:"approval"`
	Proof    trust.SignedProof    `json:"proof"`
	Commit   trust.AcceptedProof  `json:"commit"`
	Chain    [][]byte             `json:"chain"`
	Snapshot trust.SignedSnapshot `json:"snapshot"`
	Delivery trust.SignedProof    `json:"delivery"`
}

type Progress struct {
	ID            string             `json:"id"`
	Stage         string             `json:"stage"`
	Pairing       PairingContext     `json:"pairing"`
	PairingDigest string             `json:"pairing_digest"`
	Result        *InteractiveResult `json:"result"`
}

func progress(e interactiveEntry) Progress {
	return Progress{ID: e.ID, Stage: e.Stage, Pairing: e.Pairing, PairingDigest: e.PairingDigest, Result: e.Result}
}

func (s *Service) Begin(ctx context.Context, request SignedInteractiveRequest, channel ChannelPort, v trust.VerifiedSnapshot, now time.Time) (Progress, error) {
	if channel == nil || channel.EvidenceID() == [32]byte{} {
		return Progress{}, fmt.Errorf("live enrollment channel required")
	}
	if err := request.verify(); err != nil {
		return Progress{}, err
	}
	b, err := json.Marshal(request)
	if err != nil || len(b) > 8192 {
		return Progress{}, fmt.Errorf("invalid request size")
	}
	var frozen SignedInteractiveRequest
	if err := json.Unmarshal(b, &frozen); err != nil {
		return Progress{}, err
	}
	request = frozen
	if err := request.verify(); err != nil {
		return Progress{}, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	var out Progress
	err = s.store.WithProcessLock(ctx, lockFile, func() error {
		for _, live := range s.live {
			if live.EvidenceID() == channel.EvidenceID() {
				return fmt.Errorf("connection already bound to an enrollment request")
			}
		}
		old, r, err := s.loadForOperation(v, now)
		if err != nil {
			return err
		}
		if len(r.Interactive) >= maxEntries {
			return fmt.Errorf("interactive enrollment journal capacity reached")
		}
		w, err := activeWindow(r, request.Request.Window, now)
		if err != nil {
			return err
		}
		q := request.Request
		if q.Cluster != w.Cluster || q.Fingerprint != w.Fingerprint || q.Policy != w.Policy || q.Endpoint != w.Endpoint || q.Audience != w.Audience || !w.allows(q.Purpose) {
			return fmt.Errorf("request/window binding mismatch")
		}
		if !roleAllowed(v, q.Role) {
			return fmt.Errorf("request role no longer eligible")
		}
		peer, iface, err := channel.Observe()
		if err != nil {
			return err
		}
		if !peer.IsValid() || !validAudience(iface) {
			return fmt.Errorf("observed network metadata required")
		}
		if w.Policy == TrustedLAN || len(w.CIDRs) > 0 {
			allowed := false
			for _, cidr := range w.CIDRs {
				p, err := privatePrefix(cidr)
				if err != nil {
					return err
				}
				if p.Contains(peer.Unmap()) {
					allowed = true
				}
			}
			if iface != w.Interface || !allowed {
				return fmt.Errorf("request outside trusted LAN window")
			}
		}
		pending, recent, approved := 0, 0, 0
		for i := range r.Interactive {
			e := &r.Interactive[i]
			if e.Stage == "pending" || e.Stage == "approved" {
				if err := s.liveChannel(*e, nil); err != nil {
					e.Stage = "expired"
				}
			}
			if e.Request.Request.Window == w.ID {
				if e.Stage == "pending" {
					pending++
				}
				if e.Submitted > now.Add(-time.Minute).Unix() {
					recent++
				}
				if e.Result != nil {
					approved++
				}
			}
		}
		if pending >= w.MaxPending || recent >= w.MaxPerMinute || approved >= w.MaxApprovals {
			return fmt.Errorf("enrollment window quota or rate reached")
		}
		if err := s.checkNewKey(r, q.PublicKey, v); err != nil {
			return err
		}
		requestID, err := certificates.NewID()
		if err != nil {
			return err
		}
		nonce := make([]byte, 32)
		if _, err := rand.Read(nonce); err != nil {
			return err
		}
		pairing := PairingContext{Version: PairingVersion, Fingerprint: q.Fingerprint, Audience: q.Audience, Purpose: q.Purpose, Role: q.Role, RequestID: requestID, RequestHash: digestText(interactiveVersion, request), ClientNonce: bytes.Clone(q.Nonce), ServerNonce: nonce}
		digest, err := pairing.Digest()
		if err != nil {
			return err
		}
		exporter, err := channel.Bind(digest)
		if err != nil {
			return err
		}
		defer clear(exporter)
		_, audit, err := PairingPresentation(pairing, exporter)
		if err != nil {
			return err
		}
		e := interactiveEntry{ID: requestID, Stage: "pending", Submitted: now.Unix(), Request: request, Pairing: pairing, PairingDigest: audit, Peer: peer.String(), Interface: iface}
		if w.Policy == TrustedLAN {
			e.ClientConfirmed, e.AdminConfirmed = true, true
		}
		r.Interactive = append(r.Interactive, e)
		r.Sequence++
		if err := s.write(old, r); err != nil {
			return err
		}
		if s.live == nil {
			s.live = make(map[string]ChannelPort)
		}
		s.live[e.ID] = channel
		out = progress(e)
		if w.Policy == TrustedLAN {
			out, err = s.finishInteractive(e.ID, v, now)
			return err
		}
		return nil
	})
	if err != nil {
		return Progress{}, err
	}
	return out, nil
}

func activeWindow(r record, id string, now time.Time) (Window, error) {
	for _, w := range r.Windows {
		if w.ID == id {
			if w.Closed || now.Unix() < w.Created || now.Unix() >= w.Expires {
				return Window{}, fmt.Errorf("enrollment window closed or expired")
			}
			return w, nil
		}
	}
	return Window{}, fmt.Errorf("unknown enrollment window")
}

func (s *Service) liveChannel(e interactiveEntry, expected ChannelPort) error {
	c := s.live[e.ID]
	if c == nil || (expected != nil && expected.EvidenceID() != c.EvidenceID()) {
		return fmt.Errorf("request does not belong to this live connection")
	}
	peer, iface, err := c.Observe()
	if err != nil || peer.String() != e.Peer || iface != e.Interface {
		return fmt.Errorf("enrollment connection lost or changed")
	}
	if expected != nil {
		if _, _, err := expected.Observe(); err != nil {
			return err
		}
	}
	return nil
}

func (s *Service) checkNewKey(r record, key []byte, v trust.VerifiedSnapshot) error {
	if bytes.Equal(key, s.a.PrincipalKey.Public().(ed25519.PublicKey)) {
		return fmt.Errorf("cannot reuse authority principal key")
	}
	snapshot, err := snapshotValue(v)
	if err != nil {
		return err
	}
	reserved := append([][]byte{s.a.Anchor.DER(), snapshot.SignerCertificate}, snapshot.Payload.Issuers...)
	for _, der := range reserved {
		c, _, err := certificates.Parse(der)
		if err != nil {
			return err
		}
		if bytes.Equal(key, c.PublicKey.(ed25519.PublicKey)) {
			return fmt.Errorf("cannot reuse CA-purpose key")
		}
	}
	for _, e := range r.Entries {
		if e.Request != nil && bytes.Equal(key, e.Request.Request.PublicKey) {
			return fmt.Errorf("principal key already enrolled")
		}
	}
	for _, e := range r.Interactive {
		if (e.Result != nil || (e.Stage != "expired" && e.Stage != "rejected")) && bytes.Equal(key, e.Request.Request.PublicKey) {
			return fmt.Errorf("principal key already pending or enrolled")
		}
	}
	return nil
}
