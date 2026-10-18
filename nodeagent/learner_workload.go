package nodeagent

import (
	"context"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/workload"
)

// Always acquire this lock before nodeLock. Credential creation and destruction
// serialize per job, while reservation abort and lease maintenance remain free.
func (s *Store) withLearnerWorkloadLock(ctx context.Context, id string, fn func() error) error {
	if !identifier(id) {
		return fmt.Errorf("invalid learner job ID")
	}
	return s.path.WithProcessLock(ctx, "learner-workload-"+id+".lock", fn)
}

func (s *Store) PrepareLearnerWorkload(ctx context.Context, actor trust.AuthenticatedPrincipal, id string, p WorkloadPorts) (workload.SignedRequest, error) {
	return s.prepareLearnerWorkload(ctx, actor, id, p, false)
}

func (s *Store) PrepareLearnerRetryWorkload(ctx context.Context, actor trust.AuthenticatedPrincipal, id string, p WorkloadPorts) (workload.SignedRequest, error) {
	return s.prepareLearnerWorkload(ctx, actor, id, p, true)
}

func learnerWorkloadIntent(j *LearnerJob, retry bool) **workloadIntent {
	if retry {
		return &j.RetryWorkload
	}
	return &j.Workload
}

// Called only under nodeLock with a freshly loaded job. No captured lease or
// journal may authorize publication after an external credential operation.
func (s *Store) guardLearnerWorkload(ctx context.Context, actor trust.AuthenticatedPrincipal, j LearnerJob, retry bool, now time.Time) error {
	if err := authorizeLeaseActor(actor, s.cluster, now); err != nil {
		return err
	}
	m := j.Signed.Manifest
	_, ls, err := s.load()
	if err != nil {
		return err
	}
	l := findLease(&ls, m.Slot.Allocation.LeaseID)
	if l == nil || l.Job != m.Slot.Allocation.JobID || ls.Active != l.ID || now.Unix() >= l.Expires {
		return fmt.Errorf("learner workload admission no longer live")
	}
	if retry {
		if actor.Role != trust.Coordinator || !ownsLease(actor, *l) || l.State != Running || !j.Stopped || j.Recovery == nil || now.Unix() >= j.Recovery.Expires {
			return fmt.Errorf("retry workload is not authorized")
		}
	} else if actor.Role != trust.Controller || m.Controller != actor.Principal || j.Stopped || j.Assignment != nil || l.State != Prepared || now.Unix() >= m.Plan.ExpiresAt {
		return fmt.Errorf("initial learner workload is not authorized")
	}
	return ctx.Err()
}

func (s *Store) prepareLearnerWorkload(ctx context.Context, actor trust.AuthenticatedPrincipal, id string, p WorkloadPorts, retry bool) (out workload.SignedRequest, err error) {
	if p.Clock == nil || p.Initialize == nil || p.Request == nil {
		return out, fmt.Errorf("workload ports required")
	}
	if retry {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, 5*time.Second)
		defer cancel()
	}
	err = s.withLearnerWorkloadLock(ctx, id, func() error {
		var scope workload.Scope
		var intent workloadIntent
		var stop context.CancelFunc
		if e := s.path.WithProcessLock(ctx, nodeLock, func() error {
			old, j, e := s.loadLearner(id)
			if e != nil {
				return e
			}
			if e = s.guardLearnerWorkload(ctx, actor, j, retry, p.Clock()); e != nil {
				return e
			}
			i := learnerWorkloadIntent(&j, retry)
			if *i == nil {
				principal, e := newID()
				if e != nil {
					return e
				}
				*i = &workloadIntent{Principal: principal, PreparedVersion: j.Version}
				j.Version++
				if e = s.saveLearner(old, j); e != nil {
					return e
				}
			}
			intent = **i
			scope, e = learnerScope(j, intent.Principal, retry)
			if e != nil {
				return e
			}
			// Initial preparation has the original admission/actor lifetime,
			// not a fresh five-second window on every HTTP retry.
			budget := min(time.Unix(scope.AdmitUntil, 0).Sub(p.Clock()), actor.ExpiresAt.Sub(p.Clock()))
			ctx, stop = context.WithTimeout(ctx, budget)
			return ctx.Err()
		}); e != nil {
			if stop != nil {
				stop()
			}
			return e
		}
		defer stop()
		publish := func(initialized bool) error {
			return s.path.WithProcessLock(ctx, nodeLock, func() error {
				old, j, e := s.loadLearner(id)
				if e != nil {
					return e
				}
				if e = s.guardLearnerWorkload(ctx, actor, j, retry, p.Clock()); e != nil {
					return e
				}
				i := *learnerWorkloadIntent(&j, retry)
				if i == nil || i.Principal != intent.Principal || i.PreparedVersion != intent.PreparedVersion {
					return fmt.Errorf("learner workload intent changed")
				}
				current, e := learnerScope(j, i.Principal, retry)
				if e != nil || current != scope {
					return fmt.Errorf("learner workload scope changed")
				}
				if initialized && !i.Initialized {
					i.Initialized = true
					j.Version++
					return s.saveLearner(old, j)
				}
				return nil
			})
		}
		if !intent.Initialized {
			if e := p.Initialize(ctx, scope); e != nil {
				return e
			}
			if e := publish(true); e != nil {
				return e
			}
		}
		var e error
		out, e = p.Request(ctx, scope)
		if e != nil {
			return e
		}
		if e = publish(false); e != nil {
			return e
		}
		if out.Request.Scope != scope {
			return fmt.Errorf("workload port changed learner scope")
		}
		_, e = out.GrantRequest()
		return e
	})
	if err != nil {
		return workload.SignedRequest{}, err
	}
	return out, nil
}
