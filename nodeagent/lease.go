// Package nodeagent owns local node authorization, leases and NodeJob state.
// It does not issue identities, select cohorts, launch processes or initialize
// MLX. Those effects are driven through context-owned command/outcome ports.
package nodeagent

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/trust"
)

const LeaseVersion = "mixlab_node_lease_v1"
const (
	Reserved  = "RESERVED"
	Prepared  = "PREPARED"
	Running   = "RUNNING"
	Releasing = "RELEASING"
	Released  = "RELEASED"
)

type Lease struct {
	Format               string `json:"format"`
	ID                   string `json:"id"`
	Node                 string `json:"node"`
	Controller           string `json:"controller"`
	Run                  string `json:"run"`
	CapabilityGeneration uint64 `json:"capability_generation"`
	Version              uint64 `json:"version"`
	State                string `json:"state"`
	Created              int64  `json:"created"`
	Expires              int64  `json:"expires"`
	RenewBy              int64  `json:"renew_by"`
	Job                  string `json:"job"`
	CleanupReason        string `json:"cleanup_reason"`
}
type Reserve struct {
	IdempotencyKey       string `json:"idempotency_key"`
	ExpectedNodeVersion  uint64 `json:"expected_node_version"`
	CapabilityGeneration uint64 `json:"capability_generation"`
	Run                  string `json:"run"`
	TTLSeconds           int    `json:"ttl_seconds"`
}
type LeaseCommand struct {
	IdempotencyKey  string `json:"idempotency_key"`
	Lease           string `json:"lease"`
	ExpectedVersion uint64 `json:"expected_version"`
	TTLSeconds      int    `json:"ttl_seconds"`
}

func identifier(s string) bool {
	b, e := hex.DecodeString(s)
	return e == nil && len(b) == 16 && hex.EncodeToString(b) == s
}
func digest(v any) string {
	b, _ := json.Marshal(v)
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}
func newID() (string, error) {
	var b [16]byte
	_, e := rand.Read(b[:])
	return hex.EncodeToString(b[:]), e
}
func authorize(p trust.AuthenticatedPrincipal, cluster string, now time.Time) error {
	if p.Cluster != cluster || p.Role != trust.Controller || !identifier(p.Principal) || now.Before(p.NotBefore) || !now.Before(p.ExpiresAt) {
		return fmt.Errorf("current cluster controller required")
	}
	return nil
}
func ttl(seconds int) error {
	if seconds < 10 || seconds > 3600 {
		return fmt.Errorf("lease TTL must be in [10,3600] seconds")
	}
	return nil
}
func (l Lease) validate(node string) error {
	if l.Format != LeaseVersion || !identifier(l.ID) || l.Node != node || !identifier(l.Controller) || !identifier(l.Run) || l.Version == 0 || l.CapabilityGeneration == 0 || l.Created <= 0 || l.Expires <= l.Created || l.RenewBy > l.Expires || l.RenewBy < l.Created {
		return fmt.Errorf("invalid lease identity or lifetime")
	}
	switch l.State {
	case Reserved:
		if l.Job != "" {
			return fmt.Errorf("reserved lease has a job")
		}
	case Prepared, Running:
		if !identifier(l.Job) {
			return fmt.Errorf("active lease missing job")
		}
	case Releasing, Released:
		if l.Job != "" && !identifier(l.Job) {
			return fmt.Errorf("invalid released job")
		}
	default:
		return fmt.Errorf("invalid lease state")
	}
	switch l.CleanupReason {
	case "", "released", "expired", "failed", "canceled", "exited", "agent_restart", "agent_shutdown", "authority_unavailable":
	default:
		return fmt.Errorf("invalid cleanup reason")
	}
	return nil
}
