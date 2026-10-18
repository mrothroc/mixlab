package nodeagent

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"time"

	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/workerprobe"
)

const learnerPreparationVersion = "mixlab_learner_preparation_v1"

type learnerPreparation struct {
	Format            string              `json:"format"`
	Job               string              `json:"job"`
	Lease             string              `json:"lease"`
	SignedHash        string              `json:"signed_hash"`
	ProofDigest       string              `json:"proof_digest"`
	ProofGeneration   uint64              `json:"proof_generation"`
	ProfileGeneration uint64              `json:"profile_generation"`
	ProfileHash       string              `json:"profile_hash"`
	ExpectedVersion   uint64              `json:"expected_version"`
	Probe             *workerprobe.Report `json:"probe,omitempty"`
	DatasetHash       string              `json:"dataset_hash,omitempty"`
	ValidationHash    string              `json:"validation_hash,omitempty"`
	Numerical         *workerprobe.Plan   `json:"numerical,omitempty"`
}

type learnerPreparationInput struct {
	signed nodejob.SignedLearner
	proof  trust.AcceptedProof
}

type learnerPreparationState struct {
	cache   learnerPreparation
	profile Profile
	lease   Lease
	job     *LearnerJob
}

func learnerPreparationFile(job string) (string, error) {
	if !identifier(job) {
		return "", fmt.Errorf("invalid learner preparation job")
	}
	return "learner-preparation-" + job + ".json", nil
}

func learnerPreparationContext(ctx context.Context, actor trust.AuthenticatedPrincipal, accepted nodejob.AcceptedLearner, clock func() time.Time) (context.Context, context.CancelFunc, learnerPreparationInput, error) {
	var in learnerPreparationInput
	if clock == nil {
		return nil, nil, in, fmt.Errorf("preparation clock required")
	}
	var err error
	in.signed, in.proof, err = accepted.Value()
	if err != nil {
		return nil, nil, in, err
	}
	deadline := time.Unix(in.signed.Manifest.Plan.ExpiresAt, 0)
	if actor.ExpiresAt.Before(deadline) {
		deadline = actor.ExpiresAt
	}
	remaining := deadline.Sub(clock())
	if remaining <= 0 {
		return nil, nil, in, fmt.Errorf("learner preparation authority or admission expired")
	}
	bounded, cancel := context.WithTimeout(ctx, remaining)
	return bounded, cancel, in, nil
}

// Caller holds nodeLock. The lock protects the short state transaction only,
// never a worker probe, dataset read or numerical-plan callback.
func (s *Store) guardLearnerPreparation(actor trust.AuthenticatedPrincipal, in learnerPreparationInput, now time.Time) (learnerPreparationState, error) {
	var state learnerPreparationState
	if err := authorize(actor, s.cluster, now); err != nil {
		return state, err
	}
	m := in.signed.Manifest
	if m.Controller != actor.Principal || m.Plan.Coordinator.ClusterID != s.cluster || m.Slot.Allocation.NodeID != s.node || now.Unix() < m.Plan.CreatedAt || now.Unix() >= m.Plan.ExpiresAt {
		return state, fmt.Errorf("learner admission expired or wrong owner")
	}
	_, profile, err := s.loadProfile()
	if err != nil {
		return state, err
	}
	_, leases, err := s.load()
	if err != nil {
		return state, err
	}
	l := findLease(&leases, m.Slot.Allocation.LeaseID)
	if l == nil || l.ID != leases.Active || l.Controller != actor.Principal || l.Run != m.Plan.Cohort.RunID || l.CapabilityGeneration != profile.Generation || leases.CapabilityGeneration != profile.Generation || now.Unix() >= l.Expires || l.CleanupReason != "" || l.Coordinator != "" || (l.State != Reserved && l.State != Prepared) {
		return state, fmt.Errorf("live original learner reservation/profile required")
	}
	if l.State == Prepared && (l.Job != m.Slot.Allocation.JobID || l.JobKind != "learner") {
		return state, fmt.Errorf("learner lease bound to another job")
	}
	if !m.Limits.Within(profile.Limits) {
		return state, fmt.Errorf("learner exceeds node policy")
	}
	_, j, err := s.loadLearner(m.Slot.Allocation.JobID)
	switch {
	case errors.Is(err, os.ErrNotExist):
		if l.State == Prepared {
			return state, fmt.Errorf("prepared learner history missing")
		}
	case err != nil:
		return state, err
	default:
		if digest(j.Signed) != digest(in.signed) || j.Accepted.Digest != in.proof.Digest || in.proof.Generation < j.Accepted.Generation || j.Stopped || j.Assignment != nil || j.Commit != nil {
			return state, fmt.Errorf("learner preparation changed or fenced")
		}
		state.job = &j
	}
	state.profile, state.lease = profile, *l
	return state, nil
}

func (s *Store) learnerPreparationState(ctx context.Context, actor trust.AuthenticatedPrincipal, in learnerPreparationInput, expected uint64, clock func() time.Time, update func(*learnerPreparationState) error) (out learnerPreparationState, err error) {
	err = s.path.WithProcessLock(ctx, nodeLock, func() error {
		if err := ctx.Err(); err != nil {
			return err
		}
		state, err := s.guardLearnerPreparation(actor, in, clock())
		if err != nil {
			return err
		}
		name, err := learnerPreparationFile(in.signed.Manifest.Slot.Allocation.JobID)
		if err != nil {
			return err
		}
		old, err := s.path.ReadFileLimit(name, 64<<10)
		switch {
		case errors.Is(err, os.ErrNotExist):
			if expected == 0 || (state.lease.State == Reserved && state.lease.Version != expected) {
				return fmt.Errorf("learner preparation requires current reservation version")
			}
			old = nil
			state.cache = learnerPreparation{Format: learnerPreparationVersion, Job: in.signed.Manifest.Slot.Allocation.JobID, Lease: state.lease.ID, SignedHash: digest(in.signed), ProofDigest: in.proof.Digest, ProofGeneration: in.proof.Generation, ProfileGeneration: state.profile.Generation, ProfileHash: digest(state.profile), ExpectedVersion: expected}
		case err != nil:
			return err
		default:
			state.cache, err = decodeLearnerPreparation(old)
			if err != nil {
				return err
			}
		}
		if err := state.validate(in); err != nil {
			return err
		}
		// Refresh clock/authorization after state I/O, before publishing a
		// stage or committing the lease. The node lock excludes abort/renew.
		if err := ctx.Err(); err != nil {
			return err
		}
		now := clock()
		if err := authorize(actor, s.cluster, now); err != nil {
			return err
		}
		if now.Unix() >= state.lease.Expires || now.Unix() >= in.signed.Manifest.Plan.ExpiresAt {
			return fmt.Errorf("learner preparation expired during state read")
		}
		if update != nil {
			if err := update(&state); err != nil {
				return err
			}
		}
		if err := state.validate(in); err != nil {
			return err
		}
		encoded, err := json.Marshal(state.cache)
		if err != nil || len(encoded) > 64<<10 {
			return fmt.Errorf("invalid learner preparation size")
		}
		if !bytes.Equal(old, encoded) {
			if err := s.path.CompareAndSwap(name, old, encoded); err != nil {
				return err
			}
		}
		out = state
		return nil
	})
	return out, err
}

func (state learnerPreparationState) validate(in learnerPreparationInput) error {
	c, m := state.cache, in.signed.Manifest
	if c.Format != learnerPreparationVersion || c.Job != m.Slot.Allocation.JobID || c.Lease != state.lease.ID || c.SignedHash != digest(in.signed) || c.ProofDigest != in.proof.Digest || c.ProofGeneration == 0 || c.ProofGeneration > in.proof.Generation || c.ProfileGeneration != state.profile.Generation || c.ProfileHash != digest(state.profile) || c.ExpectedVersion == 0 || c.ExpectedVersion > state.lease.Version {
		return fmt.Errorf("learner preparation identity/profile changed")
	}
	if c.Probe != nil {
		if err := validateLearnerProbe(*c.Probe, m, state.profile); err != nil {
			return err
		}
	}
	if c.DatasetHash != "" && (c.Probe == nil || c.DatasetHash != m.Plan.Contract.DatasetHash) {
		return fmt.Errorf("invalid learner dataset preparation")
	}
	if c.ValidationHash != "" && (c.DatasetHash == "" || m.Validation == nil || c.ValidationHash != m.Validation.Hash) {
		return fmt.Errorf("invalid learner validation preparation")
	}
	if c.Numerical != nil {
		if state.job == nil || state.lease.State != Prepared {
			return fmt.Errorf("numerical preparation lacks prepared learner")
		}
		return validateLearnerPlan(*c.Numerical, m)
	}
	return nil
}
