package run

import (
	"fmt"
	"net/url"

	"github.com/mrothroc/mixlab/diloco/internal/wire"
	"github.com/mrothroc/mixlab/distributed"
	"github.com/mrothroc/mixlab/precision"
)

const PlanVersion = "mixlab_distributed_run_v1"
const RegistrationVersion = "mixlab_learner_registration_v1"
const CommitVersion = "mixlab_distributed_run_commit_v1"

type Slot struct {
	Member         distributed.LearnerMember `json:"member"`
	CapabilityHash string                    `json:"capability_hash"`
	Allocation     Allocation                `json:"allocation"`
}

type Plan struct {
	Format          string                              `json:"format"`
	Cohort          distributed.LearnerCohortMembership `json:"cohort"`
	Coordinator     Coordinator                         `json:"coordinator"`
	Contract        Contract                            `json:"contract"`
	BaseModel       Blob                                `json:"base_model"`
	Slots           []Slot                              `json:"slots"`
	Outer           Outer                               `json:"outer"`
	ArtifactStoreID string                              `json:"artifact_store_id"`
	Endpoint        string                              `json:"endpoint"`
	CreatedAt       int64                               `json:"created_at"`
	ExpiresAt       int64                               `json:"expires_at"`
	SHA256          string                              `json:"sha256"`
}

func (p Plan) Digest() (string, error) { p.SHA256 = ""; return Digest(p) }
func (p Plan) Validate() error {
	if p.Format != PlanVersion || !wire.Window(p.CreatedAt, p.ExpiresAt) || !wire.Text(p.ArtifactStoreID) {
		return fmt.Errorf("invalid staged plan")
	}
	c, e := p.Cohort.Canonical()
	if e != nil {
		return e
	}
	if len(p.Slots) != len(c.OrderedLearners) {
		return fmt.Errorf("plan slots do not match cohort")
	}
	for i, s := range p.Slots {
		if s.Member != c.OrderedLearners[i] || s.Allocation.NodeID != s.Member.ParticipantID || !wire.Hash(s.CapabilityHash) {
			return fmt.Errorf("slot differs from cohort")
		}
		if e := s.Allocation.Validate(); e != nil {
			return e
		}
	}
	for _, e := range []error{p.Coordinator.Validate(), p.Contract.Validate(), p.Outer.Validate(), p.BaseModel.Validate()} {
		if e != nil {
			return e
		}
	}
	u, e := url.Parse(p.Endpoint)
	if e != nil || len(p.Endpoint) > 2048 || u.Scheme != "https" || u.Hostname() == "" || u.User != nil || u.RawQuery != "" || u.Fragment != "" || u.Path != "" {
		return fmt.Errorf("plan requires an HTTPS coordinator origin")
	}
	if p.BaseModel.Kind != "model_f32" {
		return fmt.Errorf("base must be float32 model")
	}
	h, e := p.Digest()
	if e != nil {
		return e
	}
	if h != p.SHA256 {
		return fmt.Errorf("plan checksum mismatch")
	}
	return nil
}

type Registration struct {
	Format             string           `json:"format"`
	RunID              string           `json:"run_id"`
	PlanHash           string           `json:"run_plan_sha256"`
	Generation         uint64           `json:"membership_generation"`
	CohortHash         string           `json:"cohort_hash"`
	Slot               Slot             `json:"slot"`
	Contract           Contract         `json:"contract"`
	BaseModelHash      string           `json:"base_model_hash"`
	PreparationHash    string           `json:"preparation_hash"`
	JobManifestHash    string           `json:"job_manifest_hash"`
	AttemptID          string           `json:"attempt_id"`
	WorkloadGeneration uint64           `json:"workload_generation"`
	Coordinator        Coordinator      `json:"coordinator"`
	Precision          precision.Policy `json:"precision"`
	PrecisionHash      string           `json:"precision_hash"`
}

func (r Registration) Validate() error {
	if r.Format != RegistrationVersion || !wire.Text(r.RunID) || !wire.Text(r.AttemptID) {
		return fmt.Errorf("invalid registration")
	}
	for _, h := range []string{r.PlanHash, r.CohortHash, r.BaseModelHash, r.PreparationHash, r.JobManifestHash, r.PrecisionHash, r.Slot.CapabilityHash} {
		if !wire.Hash(h) {
			return fmt.Errorf("invalid registration digest")
		}
	}
	if _, e := distributed.NewLearnerCohort(r.RunID, r.Generation, []distributed.LearnerMember{r.Slot.Member}); e != nil {
		return e
	}
	if r.Slot.Member.ParticipantID != r.Slot.Allocation.NodeID {
		return fmt.Errorf("allocation participant mismatch")
	}
	for _, e := range []error{r.Contract.Validate(), r.Coordinator.Validate(), r.Slot.Allocation.Validate(), r.Precision.Validate()} {
		if e != nil {
			return e
		}
	}
	h, e := r.Precision.Digest()
	if e != nil {
		return e
	}
	if h != r.PrecisionHash || r.Precision.Backend != "metal" {
		return fmt.Errorf("R2 registration requires matching Metal precision provenance")
	}
	return nil
}

func (r Registration) ValidateFor(p Plan) error {
	if e := p.Validate(); e != nil {
		return e
	}
	if e := r.Validate(); e != nil {
		return e
	}
	if r.RunID != p.Cohort.RunID || r.PlanHash != p.SHA256 || r.Generation != p.Cohort.Generation || r.CohortHash != p.Cohort.CohortHash || r.Contract != p.Contract || r.BaseModelHash != p.BaseModel.Ref.SHA256 || r.Coordinator != p.Coordinator {
		return fmt.Errorf("registration does not bind staged plan")
	}
	for _, s := range p.Slots {
		if s == r.Slot {
			return nil
		}
	}
	return fmt.Errorf("registration substitutes staged slot")
}

// Commit is the semantic body. The service signs it using trust.RunCommit; the
// trust envelope stays outside pure contracts and never enters the MLX child.
type Commit struct {
	Format        string         `json:"format"`
	Binding       Binding        `json:"binding"`
	Coordinator   Coordinator    `json:"coordinator"`
	Registrations []Registration `json:"registrations"`
	CommittedAt   int64          `json:"committed_at"`
}

func (c Commit) Digest() (string, error) { c.Binding.CommitHash = ""; return Digest(c) }
func (c Commit) Validate() error {
	if c.Format != CommitVersion || c.CommittedAt <= 0 || len(c.Registrations) < 1 || len(c.Registrations) > distributed.MaxLearners {
		return fmt.Errorf("invalid run commit")
	}
	if e := c.Binding.Validate(); e != nil {
		return e
	}
	if e := c.Coordinator.Validate(); e != nil {
		return e
	}
	members := make([]distributed.LearnerMember, len(c.Registrations))
	for i, r := range c.Registrations {
		if e := r.Validate(); e != nil {
			return e
		}
		if r.RunID != c.Binding.RunID || r.PlanHash != c.Binding.PlanHash || r.Generation != c.Binding.Generation || r.CohortHash != c.Binding.CohortHash || r.Coordinator != c.Coordinator {
			return fmt.Errorf("commit registration binding mismatch")
		}
		members[i] = r.Slot.Member
	}
	m, e := distributed.NewLearnerCohort(c.Binding.RunID, c.Binding.Generation, members)
	if e != nil {
		return e
	}
	if m.CohortHash != c.Binding.CohortHash {
		return fmt.Errorf("commit cohort mismatch")
	}
	h, e := c.Digest()
	if e != nil {
		return e
	}
	if h != c.Binding.CommitHash {
		return fmt.Errorf("commit checksum mismatch")
	}
	return nil
}
func (c Commit) ValidateFor(p Plan) error {
	if e := p.Validate(); e != nil {
		return e
	}
	if e := c.Validate(); e != nil {
		return e
	}
	if len(c.Registrations) != len(p.Slots) || c.CommittedAt < p.CreatedAt || c.CommittedAt >= p.ExpiresAt {
		return fmt.Errorf("commit outside staged slots/deadline")
	}
	for i, r := range c.Registrations {
		if e := r.ValidateFor(p); e != nil {
			return e
		}
		if r.Slot != p.Slots[i] {
			return fmt.Errorf("registration order differs from staged plan")
		}
	}
	return nil
}
