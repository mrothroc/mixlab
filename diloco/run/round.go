package run

import (
	"fmt"
	"github.com/mrothroc/mixlab/diloco/internal/wire"
	"github.com/mrothroc/mixlab/distributed"
)

type AcceptedUpdate struct {
	WorkerID      string `json:"worker_id"`
	UpdateID      string `json:"update_id"`
	UpdateHash    string `json:"update_hash"`
	PrivateHash   string `json:"private_hash"`
	PrecisionHash string `json:"precision_hash"`
}
type Round struct {
	Format          string           `json:"format"`
	Binding         Binding          `json:"binding"`
	Round           uint64           `json:"round"`
	Contract        Contract         `json:"contract"`
	Model           Blob             `json:"model"`
	OuterState      Blob             `json:"outer_state"`
	Outer           Outer            `json:"outer"`
	Accepted        []AcceptedUpdate `json:"accepted_updates"`
	EffectiveTokens uint64           `json:"total_effective_tokens"`
	CreatedAt       int64            `json:"created_at"`
	SHA256          string           `json:"sha256"`
}

func (r Round) Digest() (string, error) { r.SHA256 = ""; return Digest(r) }
func (r Round) Validate() error {
	if r.Format != "mixlab_distributed_round_v1" || r.CreatedAt <= 0 || r.EffectiveTokens == 0 || len(r.Accepted) < 1 || len(r.Accepted) > distributed.MaxLearners {
		return fmt.Errorf("invalid global round")
	}
	for _, e := range []error{r.Binding.Validate(), r.Contract.Validate(), r.Model.Validate(), r.OuterState.Validate(), r.Outer.Validate()} {
		if e != nil {
			return e
		}
	}
	if r.Model.Kind != "model_f32" || r.OuterState.Kind != "outer_state_f32" {
		return fmt.Errorf("invalid round artifacts")
	}
	seen, ids := map[string]bool{}, map[string]bool{}
	for _, u := range r.Accepted {
		if !wire.Text(u.WorkerID) || !wire.Text(u.UpdateID) || !wire.Hash(u.UpdateHash) || !wire.Hash(u.PrivateHash) || !wire.Hash(u.PrecisionHash) || seen[u.WorkerID] || ids[u.UpdateID] {
			return fmt.Errorf("invalid/duplicate accepted update")
		}
		seen[u.WorkerID] = true
		ids[u.UpdateID] = true
	}
	h, e := r.Digest()
	if e != nil {
		return e
	}
	if h != r.SHA256 {
		return fmt.Errorf("round checksum mismatch")
	}
	return nil
}

// ValidateFor locks reduction order and the full accepted worker set. It does
// not load tensors or decide whether these receipts were durably accepted.
func (r Round) ValidateFor(p Plan, c Commit) error {
	if e := c.ValidateFor(p); e != nil {
		return e
	}
	if e := r.Validate(); e != nil {
		return e
	}
	if r.Binding != c.Binding || r.Contract != p.Contract || r.Outer != p.Outer || r.Round >= p.Outer.TotalSteps/p.Outer.InnerSteps || len(r.Accepted) != len(c.Registrations) || r.CreatedAt < c.CommittedAt {
		return fmt.Errorf("global round differs from committed contract")
	}
	for i, u := range r.Accepted {
		reg := c.Registrations[i]
		if u.WorkerID != reg.Slot.Member.WorkerID || u.PrecisionHash != reg.PrecisionHash {
			return fmt.Errorf("accepted updates must follow committed worker/policy order")
		}
	}
	return nil
}
