package nodejob

import (
	"bytes"
	"encoding/json"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/workload"
)

const LearnerVersion = "mixlab_node_learner_job_v1"

// LearnerManifest authorizes one local learner, never a synthetic DDP rank.
// Local paths, credentials and executable selection are deliberately absent.
type LearnerManifest struct {
	Version            string                 `json:"version"`
	Controller         string                 `json:"controller"`
	Attempt            string                 `json:"attempt"`
	WorkloadGeneration uint64                 `json:"workload_generation"`
	Plan               run.Plan               `json:"plan"`
	Slot               run.Slot               `json:"slot"`
	Config             json.RawMessage        `json:"config"`
	DatasetSelector    string                 `json:"dataset_selector"`
	Limits             Limits                 `json:"limits"`
	Validation         *run.ValidationDataset `json:"validation,omitempty"`
}
type SignedLearner struct {
	Manifest LearnerManifest   `json:"manifest"`
	Proof    trust.SignedProof `json:"proof"`
}

func (m LearnerManifest) Validate() error {
	if m.Validation != nil {
		if err := m.Validation.Validate(); err != nil {
			return err
		}
	}
	if m.Version != LearnerVersion || !hexID(m.Controller, 16) || !hexID(m.Attempt, 16) || m.WorkloadGeneration == 0 {
		return fmt.Errorf("invalid learner job identity")
	}
	if err := m.Plan.Validate(); err != nil {
		return err
	}
	if m.Plan.ExpiresAt-m.Plan.CreatedAt > 3600 {
		return fmt.Errorf("learner admission exceeds one hour")
	}
	found := false
	for _, s := range m.Plan.Slots {
		found = found || s == m.Slot
	}
	if !found {
		return fmt.Errorf("learner slot differs from staged plan")
	}
	for _, id := range []string{m.Plan.Cohort.RunID, m.Plan.Coordinator.ClusterID, m.Plan.Coordinator.PrincipalID, m.Slot.Member.WorkerID, m.Slot.Allocation.NodeID, m.Slot.Allocation.JobID, m.Slot.Allocation.LeaseID} {
		if !hexID(id, 16) {
			return fmt.Errorf("managed learner requires canonical identifiers")
		}
	}
	b, err := CanonicalConfig(m.Config)
	if err != nil || !bytes.Equal(b, m.Config) || Hash(b) != m.Plan.Contract.ConfigHash {
		return fmt.Errorf("learner config differs from plan")
	}
	if m.DatasetSelector == "" || len(m.DatasetSelector) > 128 || m.DatasetSelector == "." || m.DatasetSelector == ".." {
		return fmt.Errorf("logical dataset selector required")
	}
	for _, c := range m.DatasetSelector {
		valid := c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '_' || c == '-' || c == '.'
		if !valid {
			return fmt.Errorf("invalid dataset selector")
		}
	}
	if err := m.Limits.Validate(); err != nil {
		return err
	}
	if _, err := m.WorkloadDeadline(); err != nil {
		return err
	}
	b, err = json.Marshal(m)
	if err != nil || len(b) > MaxManifestBytes {
		return fmt.Errorf("learner manifest too large")
	}
	return nil
}
func (m LearnerManifest) WorkloadDeadline() (time.Time, error) {
	return (Manifest{Created: m.Plan.CreatedAt, Expires: m.Plan.ExpiresAt, Limits: m.Limits}).WorkloadDeadline()
}
func (m LearnerManifest) SigningRequest() (trust.SignRequest, error) {
	if err := m.Validate(); err != nil {
		return trust.SignRequest{}, err
	}
	b, err := json.Marshal(m)
	if err != nil {
		return trust.SignRequest{}, err
	}
	return trust.SignRequest{Version: trust.ProofVersion, Purpose: trust.NodeJob, Digest: Hash(b), Context: m.Slot.Allocation.JobID + "/" + m.Attempt, Audience: m.Slot.Allocation.NodeID}, nil
}

type AcceptedLearner struct {
	signed SignedLearner
	proof  trust.AcceptedProof
}

func AcceptLearner(a trust.Anchor, v trust.VerifiedSnapshot, actor trust.AuthenticatedPrincipal, node string, j SignedLearner, now time.Time) (AcceptedLearner, error) {
	b, err := json.Marshal(j)
	if err != nil || len(b) > MaxManifestBytes+trust.MaxTrustBytes {
		return AcceptedLearner{}, fmt.Errorf("oversized learner job")
	}
	var frozen SignedLearner
	if err = json.Unmarshal(b, &frozen); err != nil {
		return AcceptedLearner{}, err
	}
	j = frozen
	m := j.Manifest
	q, err := m.SigningRequest()
	if err != nil {
		return AcceptedLearner{}, err
	}
	if actor.Role != trust.Controller || actor.Cluster != a.Cluster() || m.Plan.Coordinator.ClusterID != a.Cluster() || actor.Principal != m.Controller || j.Proof.Evidence.Principal != actor.Principal || m.Slot.Allocation.NodeID != node || now.Before(actor.NotBefore) || !now.Before(actor.ExpiresAt) || now.Unix() < m.Plan.CreatedAt || now.Unix() >= m.Plan.ExpiresAt {
		return AcceptedLearner{}, fmt.Errorf("learner actor, node or admission deadline mismatch")
	}
	p, err := trust.VerifyProof(a, v, j.Proof, q, now)
	if err != nil {
		return AcceptedLearner{}, err
	}
	return AcceptedLearner{j, p}, nil
}
func (a AcceptedLearner) Value() (SignedLearner, trust.AcceptedProof, error) {
	if a.proof.Generation == 0 {
		return SignedLearner{}, trust.AcceptedProof{}, fmt.Errorf("unverified learner job")
	}
	b, err := json.Marshal(a.signed)
	if err != nil {
		return SignedLearner{}, trust.AcceptedProof{}, err
	}
	var j SignedLearner
	err = json.Unmarshal(b, &j)
	return j, a.proof, err
}
func (m LearnerManifest) WorkloadScope(principal string) (workload.Scope, error) {
	q, err := m.SigningRequest()
	if err != nil {
		return workload.Scope{}, err
	}
	deadline, err := m.WorkloadDeadline()
	if err != nil {
		return workload.Scope{}, err
	}
	s := workload.Scope{Profile: workload.LearnerVersion, Cluster: m.Plan.Coordinator.ClusterID, Controller: m.Controller, Node: m.Slot.Allocation.NodeID, Lease: m.Slot.Allocation.LeaseID, Run: m.Plan.Cohort.RunID, Job: m.Slot.Allocation.JobID, Attempt: m.Attempt, Workload: principal, Generation: m.WorkloadGeneration, ManifestHash: q.Digest, Created: m.Plan.CreatedAt, AdmitUntil: m.Plan.ExpiresAt, Deadline: deadline.Unix(), Worker: m.Slot.Member.WorkerID, PlanHash: m.Plan.SHA256, Coordinator: m.Plan.Coordinator.PrincipalID}
	return s, s.Validate()
}
