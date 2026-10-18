package nodeagent

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/workerprobe"
)

// PrepareLearnerNumerical returns a durable numerical plan for this exact
// signed admission/profile. The callback runs without the node process lock;
// its result is validated before publication and is never cached on failure.
func (s *Store) PrepareLearnerNumerical(ctx context.Context, actor trust.AuthenticatedPrincipal, accepted nodejob.AcceptedLearner, probe func(context.Context, []byte) (workerprobe.Plan, error), clock func() time.Time) (workerprobe.Plan, error) {
	ctx, cancel, in, err := learnerPreparationContext(ctx, actor, accepted, clock)
	if err != nil {
		return workerprobe.Plan{}, err
	}
	defer cancel()
	state, err := s.learnerPreparationState(ctx, actor, in, 0, clock, nil)
	if err != nil {
		return workerprobe.Plan{}, err
	}
	if state.job == nil || state.lease.State != Prepared {
		return workerprobe.Plan{}, fmt.Errorf("numerical probe requires prepared learner")
	}
	if state.cache.Numerical != nil {
		return *state.cache.Numerical, nil
	}
	if probe == nil {
		return workerprobe.Plan{}, fmt.Errorf("learner numerical probe unavailable")
	}
	plan, err := probe(ctx, append([]byte(nil), in.signed.Manifest.Config...))
	if err != nil {
		return workerprobe.Plan{}, err
	}
	// Freeze pointer/slice fields before they enter the canonical sidecar.
	encoded, err := json.Marshal(plan)
	if err != nil {
		return workerprobe.Plan{}, err
	}
	plan, err = workerprobe.DecodePlan(encoded)
	if err != nil {
		return workerprobe.Plan{}, err
	}
	if err := validateLearnerPlan(plan, in.signed.Manifest); err != nil {
		return workerprobe.Plan{}, err
	}
	state, err = s.learnerPreparationState(ctx, actor, in, 0, clock, func(st *learnerPreparationState) error {
		if st.job == nil || st.lease.State != Prepared {
			return fmt.Errorf("learner no longer prepared")
		}
		if st.cache.Numerical != nil && digest(*st.cache.Numerical) != digest(plan) {
			return fmt.Errorf("concurrent learner numerical probe changed")
		}
		st.cache.Numerical = &plan
		return nil
	})
	if err != nil {
		return workerprobe.Plan{}, err
	}
	return *state.cache.Numerical, nil
}

func validateLearnerPlan(p workerprobe.Plan, m nodejob.LearnerManifest) error {
	if err := p.Validate(); err != nil {
		return err
	}
	c := m.Plan.Contract
	if p.Learner == nil || p.Learner.Precision == nil || p.BuildID != c.BuildHash || p.ConfigHash != c.ConfigHash || p.ProgramHash != c.ProgramHash || p.WeightLayoutHash != c.WeightLayoutHash || p.OptimizerHash != c.InnerOptimizerHash || p.Learner.Outer != m.Plan.Outer {
		return fmt.Errorf("approved learner numerical contract changed")
	}
	return nil
}
