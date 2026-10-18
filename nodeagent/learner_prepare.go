package nodeagent

import (
	"context"
	"fmt"

	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/workerprobe"
)

// PrepareLearnerChecked checkpoints successful immutable preparation stages.
// The active lease freezes the local profile; all expensive ports run unlocked.
func (s *Store) PrepareLearnerChecked(ctx context.Context, actor trust.AuthenticatedPrincipal, accepted nodejob.AcceptedLearner, version uint64, p PreparationPorts) (LearnerJob, error) {
	if p.Clock == nil || p.Probe == nil || p.DatasetID == nil {
		return LearnerJob{}, fmt.Errorf("preparation ports required")
	}
	ctx, cancel, in, err := learnerPreparationContext(ctx, actor, accepted, p.Clock)
	if err != nil {
		return LearnerJob{}, err
	}
	defer cancel()
	state, err := s.learnerPreparationState(ctx, actor, in, version, p.Clock, nil)
	if err != nil {
		return LearnerJob{}, err
	}
	if state.job != nil && state.lease.State == Prepared {
		return *state.job, nil
	}
	m := in.signed.Manifest
	if state.cache.Probe == nil {
		probe, err := p.Probe(ctx)
		if err != nil {
			return LearnerJob{}, err
		}
		state, err = s.learnerPreparationState(ctx, actor, in, version, p.Clock, func(st *learnerPreparationState) error {
			if err := validateLearnerProbe(probe, m, st.profile); err != nil {
				return err
			}
			if st.cache.Probe == nil {
				st.cache.Probe = &probe
			}
			return nil
		})
		if err != nil {
			return LearnerJob{}, err
		}
	}
	if state.cache.DatasetHash == "" {
		path, err := learnerDatasetPath(state.profile, m.DatasetSelector, m.Plan.Contract.DatasetHash)
		if err != nil {
			return LearnerJob{}, err
		}
		hash, err := p.DatasetID(ctx, path)
		if err != nil {
			return LearnerJob{}, err
		}
		state, err = s.learnerPreparationState(ctx, actor, in, version, p.Clock, func(st *learnerPreparationState) error {
			if hash != m.Plan.Contract.DatasetHash {
				return fmt.Errorf("learner dataset content changed")
			}
			st.cache.DatasetHash = hash
			return nil
		})
		if err != nil {
			return LearnerJob{}, err
		}
	}
	if m.Validation != nil && state.cache.ValidationHash == "" {
		path, err := learnerDatasetPath(state.profile, m.Validation.Split, m.Validation.Hash)
		if err != nil {
			return LearnerJob{}, err
		}
		hash, err := p.DatasetID(ctx, path)
		if err != nil {
			return LearnerJob{}, err
		}
		state, err = s.learnerPreparationState(ctx, actor, in, version, p.Clock, func(st *learnerPreparationState) error {
			if hash != m.Validation.Hash {
				return fmt.Errorf("validation dataset content changed")
			}
			st.cache.ValidationHash = hash
			return nil
		})
		if err != nil {
			return LearnerJob{}, err
		}
	}
	state, err = s.learnerPreparationState(ctx, actor, in, version, p.Clock, func(st *learnerPreparationState) error {
		j, err := s.prepareLearnerLocked(actor, in.signed, in.proof, st.lease.Version, p.Clock())
		if err == nil {
			st.job = &j
		}
		return err
	})
	if err != nil {
		return LearnerJob{}, err
	}
	return *state.job, nil
}

func validateLearnerProbe(probe workerprobe.Report, m nodejob.LearnerManifest, profile Profile) error {
	if err := probe.Validate(); err != nil {
		return err
	}
	if !probe.Available || !probe.MLXSupported || probe.DeviceKind != "metal" || probe.BuildID != m.Plan.Contract.BuildHash || !sameProbeContract(probe, profile.Probe) {
		return fmt.Errorf("learner executable/device changed")
	}
	if probe.FreeMemoryBytes != nil && *probe.FreeMemoryBytes < m.Limits.MemoryBytes {
		return fmt.Errorf("insufficient observed learner memory")
	}
	return nil
}

func learnerDatasetPath(profile Profile, selector, hash string) (string, error) {
	for _, d := range profile.Datasets {
		if d.Selector == selector && d.ID == hash {
			return d.TrainPattern, nil
		}
	}
	return "", fmt.Errorf("learner dataset not registered")
}
