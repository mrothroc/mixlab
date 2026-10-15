package run

import (
	"fmt"
	"github.com/mrothroc/mixlab/diloco/internal/wire"
)

// Grant is a signed semantic body, never an authorization result. The API
// adapter must verify proof/principal/current assignment and reserve a durable
// ledger use before any storage I/O. Private artifacts stay local in R2.
type Grant struct {
	Format        string      `json:"format"`
	ID            string      `json:"grant_id"`
	Binding       Binding     `json:"binding"`
	Coordinator   Coordinator `json:"coordinator"`
	PrincipalID   string      `json:"principal_id"`
	ParticipantID string      `json:"participant_id"`
	WorkerID      string      `json:"worker_id"`
	AssignmentID  string      `json:"assignment_id"`
	Operation     string      `json:"operation"`
	Artifact      Blob        `json:"artifact"`
	UploadID      string      `json:"upload_id"`
	AccessClass   string      `json:"access_class"`
	IssuedAt      int64       `json:"issued_at"`
	ExpiresAt     int64       `json:"expires_at"`
	MaximumUses   uint64      `json:"maximum_uses"`
	Nonce         string      `json:"nonce"`
	SHA256        string      `json:"sha256"`
}

func (g Grant) Digest() (string, error) { g.SHA256 = ""; return Digest(g) }
func (g Grant) Validate() error {
	if g.Format != "mixlab_artifact_access_grant_v1" || !wire.Text(g.ID) || !wire.Text(g.PrincipalID) || !wire.Text(g.ParticipantID) || !wire.Text(g.WorkerID) || !wire.Text(g.AssignmentID) || !wire.Text(g.Nonce) || !wire.Window(g.IssuedAt, g.ExpiresAt) || g.MaximumUses == 0 || g.MaximumUses > 1024 {
		return fmt.Errorf("invalid artifact grant")
	}
	for _, e := range []error{g.Binding.Validate(), g.Coordinator.Validate(), g.Artifact.Validate()} {
		if e != nil {
			return e
		}
	}
	switch g.AccessClass {
	case "global_base":
		if g.Operation != "read" || g.UploadID != "" || (g.Artifact.Kind != "model_f32" && g.Artifact.Kind != "outer_state_f32") {
			return fmt.Errorf("global grant must read a model or outer state")
		}
	case "outer_update":
		if (g.Operation != "create" && g.Operation != "resumable_upload") || !wire.Text(g.UploadID) || g.Artifact.Kind != "delta_f32" {
			return fmt.Errorf("update grant must create a bounded delta")
		}
	default:
		return fmt.Errorf("unsupported R2 access class (private artifacts remain local)")
	}
	h, e := g.Digest()
	if e != nil {
		return e
	}
	if h != g.SHA256 {
		return fmt.Errorf("grant checksum mismatch")
	}
	return nil
}

// GrantUse identifies one ledger reservation; retransmitted chunks reuse it.
type GrantUse struct {
	Format        string `json:"format"`
	GrantHash     string `json:"grant_hash"`
	GrantID       string `json:"grant_id"`
	Operation     string `json:"operation"`
	UploadID      string `json:"upload_id"`
	IdempotencyID string `json:"idempotency_id"`
}

func (u GrantUse) Validate() error {
	if u.Format != "mixlab_artifact_use_v1" || !wire.Hash(u.GrantHash) || !wire.Text(u.GrantID) || !wire.Text(u.IdempotencyID) {
		return fmt.Errorf("invalid grant use")
	}
	if u.Operation == "read" && u.UploadID == "" {
		return nil
	}
	if (u.Operation == "create" || u.Operation == "resumable_upload") && wire.Text(u.UploadID) {
		return nil
	}
	return fmt.Errorf("invalid grant operation")
}
