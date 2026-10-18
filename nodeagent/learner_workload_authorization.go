package nodeagent

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/workload"
)

// AuthorizeLearnerWorkload supplements an INITIAL request only. The credential
// owner verifies the fresh signature and binds its exact durable request; this
// store owns the live admission/intent guards, not credential keys or trust.
func (s *Store) AuthorizeLearnerWorkload(ctx context.Context, actor trust.AuthenticatedPrincipal, id string, original workload.SignedRequest, p WorkloadPorts) (workload.RequestAuthorization, error) {
	var out workload.RequestAuthorization
	if p.Clock == nil || p.AuthorizeRequest == nil {
		return out, fmt.Errorf("workload reauthorization ports required")
	}
	// Freeze the caller's slices before crossing the local credential port.
	b, err := json.Marshal(original)
	if err != nil || len(b) > trust.MaxTrustBytes+8192 {
		return out, fmt.Errorf("invalid workload request size")
	}
	var request workload.SignedRequest
	if err = json.Unmarshal(b, &request); err != nil {
		return out, err
	}
	want, err := request.ReauthorizationRequest()
	if err != nil {
		return out, err
	}
	now := p.Clock()
	budget := min(time.Unix(request.Request.Scope.AdmitUntil, 0).Sub(now), actor.ExpiresAt.Sub(now))
	ctx, cancel := context.WithTimeout(ctx, budget)
	defer cancel()
	err = s.withLearnerWorkloadLock(ctx, id, func() error {
		var scope workload.Scope
		var intent workloadIntent
		guard := func(first bool) error {
			return s.path.WithProcessLock(ctx, nodeLock, func() error {
				_, j, err := s.loadLearner(id)
				if err != nil {
					return err
				}
				if err := s.guardLearnerWorkload(ctx, actor, j, false, p.Clock()); err != nil {
					return err
				}
				if j.Workload == nil || !j.Workload.Initialized {
					return fmt.Errorf("initialized learner credential intent required")
				}
				current, err := learnerScope(j, j.Workload.Principal, false)
				if err != nil {
					return err
				}
				if current != request.Request.Scope {
					return fmt.Errorf("reauthorization request changed learner scope")
				}
				if first {
					scope, intent = current, *j.Workload
				} else if current != scope || *j.Workload != intent {
					return fmt.Errorf("learner credential intent changed during reauthorization")
				}
				return nil
			})
		}
		if err := guard(true); err != nil {
			return err
		}
		var err error
		out, err = p.AuthorizeRequest(ctx, scope, request)
		if err != nil {
			return err
		}
		if err := guard(false); err != nil {
			return err
		}
		if out.Version != workload.RequestAuthorizationVersion || out.Proof.Request != want || out.Proof.Evidence.Principal != scope.Node || out.Proof.Evidence.Role != trust.Node || out.Proof.Evidence.Cluster != scope.Cluster {
			return fmt.Errorf("credential port changed request authorization binding")
		}
		return ctx.Err()
	})
	if err != nil {
		return workload.RequestAuthorization{}, err
	}
	return out, nil
}
