package nodeagent

import (
	"context"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/trust"
)

// PrepareLearnerChecked re-probes the installed worker and dataset under the
// original reservation. Unlike DDP preparation it has no ring requirement.
func (s *Store) PrepareLearnerChecked(ctx context.Context, actor trust.AuthenticatedPrincipal, accepted nodejob.AcceptedLearner, version uint64, p PreparationPorts) (LearnerJob, error) {
	if p.Clock == nil || p.Probe == nil || p.DatasetID == nil {
		return LearnerJob{}, fmt.Errorf("preparation ports required")
	}
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	if err := authorize(actor, s.cluster, p.Clock()); err != nil {
		return LearnerJob{}, err
	}
	signed, _, err := accepted.Value()
	if err != nil {
		return LearnerJob{}, err
	}
	m := signed.Manifest
	if m.Controller != actor.Principal || m.Slot.Allocation.NodeID != s.node {
		return LearnerJob{}, fmt.Errorf("wrong learner preparation owner")
	}
	_, profile, err := s.loadProfile()
	if err != nil {
		return LearnerJob{}, err
	}
	av, err := s.Availability(p.Clock())
	if err != nil {
		return LearnerJob{}, err
	}
	if av.Lease == nil || av.Lease.ID != m.Slot.Allocation.LeaseID || av.Lease.Controller != actor.Principal || av.Lease.CapabilityGeneration != profile.Generation || av.CapabilityGeneration != profile.Generation || p.Clock().Unix() >= av.Lease.Expires {
		return LearnerJob{}, fmt.Errorf("learner reservation/profile changed")
	}
	if !m.Limits.Within(profile.Limits) {
		return LearnerJob{}, fmt.Errorf("learner exceeds node policy")
	}
	probe, err := p.Probe(ctx)
	if err != nil {
		return LearnerJob{}, err
	}
	if err = probe.Validate(); err != nil {
		return LearnerJob{}, err
	}
	if !probe.Available || !probe.MLXSupported || probe.DeviceKind != "metal" || probe.BuildID != m.Plan.Contract.BuildHash || !sameProbeContract(probe, profile.Probe) {
		return LearnerJob{}, fmt.Errorf("learner executable/device changed")
	}
	if probe.FreeMemoryBytes != nil && *probe.FreeMemoryBytes < m.Limits.MemoryBytes {
		return LearnerJob{}, fmt.Errorf("insufficient observed learner memory")
	}
	path := ""
	for _, d := range profile.Datasets {
		if d.Selector == m.DatasetSelector && d.ID == m.Plan.Contract.DatasetHash {
			path = d.TrainPattern
		}
	}
	if path == "" {
		return LearnerJob{}, fmt.Errorf("learner dataset not registered")
	}
	hash, err := p.DatasetID(ctx, path)
	if err != nil {
		return LearnerJob{}, err
	}
	if hash != m.Plan.Contract.DatasetHash {
		return LearnerJob{}, fmt.Errorf("learner dataset content changed")
	}
	if err = ctx.Err(); err != nil {
		return LearnerJob{}, err
	}
	return s.PrepareLearner(ctx, actor, accepted, version, p.Clock())
}
