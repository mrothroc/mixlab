package clusterapp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/workload"
)

var errLearnerPreparationPending = errors.New("learner preparation is still running")

// Only one reservation can own this node. Keep at most one in-flight operation,
// independent of HTTP request lifetimes; the store owns durable stage receipts.
type learnerPreparation struct {
	job, digest string
	cancel      context.CancelCauseFunc
	done        chan struct{}
	prepared    *learnerPreparationResult
	learnerPreparationResult
	err error // published by closing done
}

// A replacement retains one detached prior result, never an operation chain.
// The receipt is populated only from successful fresh trust.VerifyProof.
type learnerPreparationResult struct {
	resultDigest  string
	authorization trust.AcceptedProof
	out           NodeLearnerPrepared
}

func (r learnerPreparationResult) AcceptedGeneration(digest string) (uint64, bool, error) {
	return r.authorization.Generation, r.authorization.Digest == digest, nil
}

func (r learnerPreparationResult) verifyDigest() error {
	digest, err := run.Digest(r.out)
	if err != nil || digest != r.resultDigest {
		return fmt.Errorf("cached learner preparation changed")
	}
	return nil
}

func (r learnerPreparationResult) verifyHistory(anchor trust.Anchor, view trust.VerifiedSnapshot, now time.Time) error {
	if err := r.verifyDigest(); err != nil {
		return err
	}
	want, err := r.out.Request.ReauthorizationRequest()
	if err != nil {
		return err
	}
	return trust.VerifyHistoricalProof(anchor, view, r.out.Authorization.Proof, want, r, now)
}

func (r learnerPreparationResult) snapshot() (*learnerPreparationResult, error) {
	b, err := json.Marshal(r.out)
	if err != nil {
		return nil, err
	}
	var frozen NodeLearnerPrepared
	if err := json.Unmarshal(b, &frozen); err != nil {
		return nil, err
	}
	frozenResult := &learnerPreparationResult{out: frozen, resultDigest: r.resultDigest, authorization: r.authorization}
	if err := frozenResult.verifyDigest(); err != nil {
		return nil, err
	}
	return frozenResult, nil
}

func (a *NodeJobs) PrepareLearner(ctx context.Context, actor trust.AuthenticatedPrincipal, q NodeLearnerPrepareRequest) (NodeLearnerPrepared, error) {
	var zero NodeLearnerPrepared
	if a.options.LearnerProbe == nil {
		return zero, fmt.Errorf("learner numerical probe unavailable")
	}
	// Authenticate every retry before joining or exposing a cached result.
	if err := a.checkLearnerPreparation(ctx, actor, q); err != nil {
		return zero, err
	}
	digest, err := run.Digest(q.Signed)
	if err != nil {
		return zero, err
	}
	job := q.Signed.Manifest.Slot.Allocation.JobID
	a.prepareMu.Lock()
	op := a.preparing
	var prepared *learnerPreparationResult
	if op != nil {
		select {
		case <-op.done:
			if op.job != job || op.digest != digest {
				op = nil
			} else if op.err != nil {
				prepared = op.prepared
				op = nil
			}
		default:
			if op.job != job || op.digest != digest {
				a.prepareMu.Unlock()
				return zero, errLearnerPreparationPending
			}
		}
	}
	if op == nil {
		op = a.startLearnerPreparation(ctx, actor, q, digest, prepared)
	}
	a.prepareMu.Unlock()
	for {
		select {
		case <-ctx.Done():
			return zero, ctx.Err()
		case <-op.done:
		}
		if op.err != nil {
			return zero, op.err
		}
		if err := op.verifyDigest(); err != nil {
			return zero, err
		}
		if err := a.checkLearnerPreparation(ctx, actor, q); err != nil {
			return zero, err
		}
		view, err := a.current(ctx)
		if err != nil {
			return zero, err
		}
		now := a.options.Clock()
		authorizationErr := workload.VerifyRequestAuthorization(a.options.Anchor, view, op.out.Request, op.out.Authorization, now)
		if authorizationErr == nil {
			if err := checkNodeCaller(ctx); err != nil {
				return zero, err
			}
			return op.out, nil
		}
		// This owner retains the actual receipt returned by fresh verification.
		// History permits signing a replacement, never returning an expired proof.
		// a.current authenticates the CURRENT node, while historical verification
		// rejects compromised old evidence without blocking ordinary renewal.
		if err := op.verifyHistory(a.options.Anchor, view, now); err != nil {
			return zero, err
		}
		prepared, err := op.snapshot()
		if err != nil {
			return zero, err
		}
		// A completed response's supplement can expire independently of its
		// durable request. Replace only this operation; concurrent callers join
		// the replacement without re-running preparation or touching its key.
		a.prepareMu.Lock()
		if a.preparing == op {
			op = a.startLearnerPreparation(ctx, actor, q, digest, prepared)
		} else {
			op = a.preparing
		}
		if op.job != job || op.digest != digest {
			a.prepareMu.Unlock()
			return zero, errLearnerPreparationPending
		}
		a.prepareMu.Unlock()
	}
}

// Caller holds prepareMu; trust and store I/O run only in the operation goroutine.
func (a *NodeJobs) startLearnerPreparation(ctx context.Context, actor trust.AuthenticatedPrincipal, q NodeLearnerPrepareRequest, digest string, prepared *learnerPreparationResult) *learnerPreparation {
	deadline := time.Unix(q.Signed.Manifest.Plan.ExpiresAt, 0)
	if actor.ExpiresAt.Before(deadline) {
		deadline = actor.ExpiresAt
	}
	// The injected clock determines admission time remaining; the timer must
	// still advance independently in tests and production.
	life, stop := context.WithTimeout(a.life, deadline.Sub(a.options.Clock()))
	life, cancel := context.WithCancelCause(life)
	if check, ok := ctx.Value(nodeCallerCheckKey{}).(func() error); ok {
		life = context.WithValue(life, nodeCallerCheckKey{}, check)
	}
	op := &learnerPreparation{job: q.Signed.Manifest.Slot.Allocation.JobID, digest: digest, prepared: prepared, cancel: func(cause error) {
		cancel(cause)
		stop()
	}, done: make(chan struct{})}
	a.preparing = op
	go a.runLearnerPreparation(life, actor, q, op)
	return op
}

func (a *NodeJobs) runLearnerPreparation(ctx context.Context, actor trust.AuthenticatedPrincipal, q NodeLearnerPrepareRequest, op *learnerPreparation) {
	defer close(op.done)
	defer op.cancel(nil)
	watchCtx, stopWatch := context.WithCancel(ctx)
	defer stopWatch()
	watchDone := make(chan struct{})
	go func() {
		defer close(watchDone)
		ticker := time.NewTicker(time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-watchCtx.Done():
				return
			case <-ticker.C:
				err := a.checkLearnerPreparation(watchCtx, actor, q)
				if watchCtx.Err() != nil {
					return
				}
				if err != nil {
					op.cancel(err)
					return
				}
			}
		}
	}()
	if op.prepared == nil {
		op.out, op.err = a.prepareLearner(ctx, actor, q)
	} else if op.err = a.checkLearnerPreparation(ctx, actor, q); op.err == nil {
		view, err := a.current(ctx)
		if err == nil {
			err = op.prepared.verifyHistory(a.options.Anchor, view, a.options.Clock())
		}
		if err == nil {
			err = checkNodeCaller(ctx)
		}
		if err == nil {
			op.out, err = a.authorizeLearnerPreparation(ctx, actor, op.job, op.prepared.out)
		}
		op.err = err
	}
	if op.err == nil {
		view, err := a.current(ctx)
		op.err = err
		if op.err == nil && op.prepared != nil {
			op.err = op.prepared.verifyHistory(a.options.Anchor, view, a.options.Clock())
		}
		if op.err == nil {
			want, err := op.out.Request.ReauthorizationRequest()
			op.err = err
			if op.err == nil {
				op.authorization, op.err = trust.VerifyProof(a.options.Anchor, view, op.out.Authorization.Proof, want, a.options.Clock())
			}
		}
	}
	if op.err == nil {
		op.err = a.checkLearnerPreparation(ctx, actor, q)
	}
	if op.err == nil {
		op.resultDigest, op.err = run.Digest(op.out)
	}
	stopWatch()
	<-watchDone
	op.err = errors.Join(op.err, context.Cause(ctx))
}

func (a *NodeJobs) checkLearnerPreparation(ctx context.Context, actor trust.AuthenticatedPrincipal, q NodeLearnerPrepareRequest) error {
	if err := a.learnerPreparationScope(ctx, actor, q); err != nil {
		return err
	}
	_, err := a.currentForEvidence(ctx, q.Signed.Proof.Evidence.Snapshot, func(v trust.VerifiedSnapshot) error {
		_, err := a.options.Store.AcceptLearnerPreparation(ctx, a.options.Anchor, v, actor, q.Signed, a.options.Clock())
		return err
	})
	return err
}

func (a *NodeJobs) learnerPreparationScope(ctx context.Context, actor trust.AuthenticatedPrincipal, q NodeLearnerPrepareRequest) error {
	m := q.Signed.Manifest
	l, err := a.options.Store.LeaseStatus(ctx, actor, m.Slot.Allocation.LeaseID, a.options.Clock())
	if err != nil {
		return err
	}
	x, err := a.options.Store.ActiveExecution(ctx)
	if err != nil {
		return err
	}
	if x == nil || x.Lease.ID != l.ID {
		return fmt.Errorf("live original learner reservation required")
	}
	l = x.Lease
	if q.ExpectedLeaseVersion == 0 || l.Controller != actor.Principal || l.Run != m.Plan.Cohort.RunID || a.options.Clock().Unix() >= l.Expires {
		return fmt.Errorf("live original learner reservation required")
	}
	if l.State == nodeagent.Reserved && l.Job == "" || l.State == nodeagent.Prepared && l.Job == m.Slot.Allocation.JobID && l.JobKind == "learner" {
		return nil
	}
	return fmt.Errorf("learner preparation reservation no longer active")
}

// Cleanup must join preparation before destroying a context it may be writing.
// It does not hold the node relay mutex or lease-store lock while waiting.
func (a *NodeJobs) StopLearnerPreparation(ctx context.Context, job string) error {
	a.prepareMu.Lock()
	op := a.preparing
	if op != nil && (job == "" || op.job == job) {
		op.cancel(context.Canceled)
	} else {
		op = nil
	}
	a.prepareMu.Unlock()
	if op != nil {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-op.done:
		}
	}
	return nil
}
