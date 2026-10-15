// Package run owns pure coordinator contracts. Validation is structural, not
// authentication, admission, signature verification, or repository policy.
package run

import (
	"fmt"
	"github.com/mrothroc/mixlab/artifact"
	"github.com/mrothroc/mixlab/diloco/internal/wire"
)

// Decode requires canonical versioned records; callers authenticate separately.
func Decode[T interface{ Validate() error }](b []byte, dst T) error { return wire.Decode(b, dst) }
func Digest(v any) (string, error)                                  { return wire.Digest(v) }

type Contract struct {
	ConfigHash         string `json:"config_hash"`
	ProgramHash        string `json:"program_hash"`
	WeightLayoutHash   string `json:"weight_layout_hash"`
	InnerOptimizerHash string `json:"inner_optimizer_hash"`
	DatasetHash        string `json:"dataset_hash"`
	BuildHash          string `json:"build_hash"`
}

func (c Contract) Validate() error {
	for _, s := range []string{c.ConfigHash, c.ProgramHash, c.WeightLayoutHash, c.InnerOptimizerHash, c.DatasetHash, c.BuildHash} {
		if !wire.Hash(s) {
			return fmt.Errorf("invalid training contract hash")
		}
	}
	return nil
}

// Binding identifies a committed run, never a staged plan alone.
type Binding struct {
	RunID      string `json:"run_id"`
	PlanHash   string `json:"run_plan_sha256"`
	CommitHash string `json:"run_commit_sha256"`
	Generation uint64 `json:"membership_generation"`
	CohortHash string `json:"cohort_hash"`
}

func (b Binding) Validate() error {
	if !wire.Text(b.RunID) || !wire.Hash(b.PlanHash) || !wire.Hash(b.CommitHash) || !wire.Hash(b.CohortHash) {
		return fmt.Errorf("committed run binding required")
	}
	return nil
}

type Coordinator struct {
	ClusterID   string `json:"cluster_id"`
	PrincipalID string `json:"principal_id"`
	Role        string `json:"role"`
	Audience    string `json:"audience"`
}

func (c Coordinator) Validate() error {
	if !wire.Text(c.ClusterID) || !wire.Text(c.PrincipalID) || !wire.Text(c.Audience) || c.Role != "coordinator" {
		return fmt.Errorf("coordinator binding required")
	}
	return nil
}

// Blob combines existing byte identity with an exact schema/media discriminator.
type Blob struct {
	Ref  artifact.Ref `json:"ref"`
	Kind string       `json:"kind"`
}

func (b Blob) Validate() error {
	switch b.Kind {
	case "model_f32", "delta_f32", "outer_state_f32", "private_state":
	default:
		return fmt.Errorf("unknown artifact kind")
	}
	return b.Ref.Validate()
}

type Outer struct {
	Kind         string  `json:"kind"`
	LR           float64 `json:"lr"`
	Momentum     float64 `json:"momentum"`
	Aggregation  string  `json:"aggregation"`
	InnerSteps   uint64  `json:"inner_steps"`
	TotalSteps   uint64  `json:"total_steps"`
	Accumulation uint64  `json:"gradient_accumulation_steps"`
}

func (o Outer) Validate() error {
	if o.Kind != "nesterov" || !wire.Finite(o.LR) || o.LR <= 0 || !wire.Finite(o.Momentum) || o.Momentum < 0 || o.Momentum >= 1 || o.InnerSteps == 0 || o.TotalSteps == 0 || o.TotalSteps%o.InnerSteps != 0 || o.Accumulation == 0 || (o.Aggregation != "uniform" && o.Aggregation != "effective_tokens") {
		return fmt.Errorf("invalid fixed-step Nesterov contract")
	}
	return nil
}

// Allocation is the R2-only discriminated provider proof. Adding external
// allocations requires a new supported variant, not arbitrary JSON payloads.
type Allocation struct {
	Provider string `json:"provider"`
	NodeID   string `json:"node_id"`
	LeaseID  string `json:"lease_id"`
	JobID    string `json:"job_id"`
	RefHash  string `json:"ref_hash"`
}

func (a Allocation) Validate() error {
	if a.Provider != "node_agent" || !wire.Text(a.NodeID) || !wire.Text(a.LeaseID) || !wire.Text(a.JobID) || !wire.Hash(a.RefHash) {
		return fmt.Errorf("R2 requires a node_agent allocation")
	}
	v := a
	v.RefHash = ""
	h, e := Digest(v)
	if e != nil {
		return e
	}
	if h != a.RefHash {
		return fmt.Errorf("allocation hash mismatch")
	}
	return nil
}
