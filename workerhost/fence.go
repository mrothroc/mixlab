package workerhost

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"

	"github.com/mrothroc/mixlab/workerhost/contract"
	"github.com/mrothroc/mixlab/workerjob"
)

func (r Attempt) validateFence() error {
	if err := r.Fence.Validate(); err != nil {
		return err
	}
	hash, err := workerjob.Digest(*r.Fence)
	if err != nil || hash != r.Digest {
		return fmt.Errorf("invalid execution fence digest")
	}
	approval, _ := json.Marshal(r.Approval)
	empty, _ := json.Marshal(contract.Approved{})
	if string(approval) != string(empty) || r.Started != nil || r.Terminal == nil {
		return fmt.Errorf("fenced preparation contains execution state")
	}
	o := *r.Terminal
	if err := o.Validate(); err != nil {
		return err
	}
	if o.JobID != r.Fence.JobID || o.AttemptID != r.Fence.AttemptID || o.ManifestHash != r.Fence.ManifestHash || o.ApprovalHash != "" || o.Kind != contract.Canceled {
		return fmt.Errorf("cancellation fence outcome mismatch")
	}
	return nil
}

// FencePreparation is safe to retry and serializes against Run. It does not
// infer absence from a missing PID: only an entirely unclaimed attempt can be
// fenced without consulting the hosting reconciler.
func (s *AttemptStore) FencePreparation(ctx context.Context, f contract.Fence) (out contract.Outcome, err error) {
	if err := f.Validate(); err != nil {
		return out, err
	}
	hash, err := workerjob.Digest(f)
	if err != nil {
		return out, err
	}
	err = s.path.WithProcessLock(ctx, attemptLock, func() error {
		_, r, err := s.load()
		if err == nil {
			if r.Fence == nil || *r.Fence != f {
				return ErrReconciliationRequired
			}
			out = *r.Terminal
			return nil
		}
		if !errors.Is(err, os.ErrNotExist) {
			return err
		}
		out = contract.Outcome{JobID: f.JobID, AttemptID: f.AttemptID, ManifestHash: f.ManifestHash, Version: 1, Kind: contract.Canceled, NoChild: true}
		r = Attempt{Version: contract.Version, Fence: &f, Digest: hash, Terminal: &out}
		return s.publishUnstartedFence(r)
	})
	return out, err
}

// FenceApproved closes the node-STARTING/host-not-yet-claimed gap. An existing
// nonterminal claim requires physical reconciliation; it is never inferred
// dead from a missing PID or journal. This is a local hosting port only.
func (s *AttemptStore) FenceApproved(ctx context.Context, a contract.Approved) (out contract.Outcome, err error) {
	a, hash, err := freezeApproval(a)
	if err != nil {
		return out, err
	}
	err = s.path.WithProcessLock(ctx, attemptLock, func() error {
		_, r, err := s.load()
		if err == nil {
			if r.Digest != hash {
				return fmt.Errorf("attempt already bound to another approval")
			}
			if r.Terminal == nil {
				return ErrReconciliationRequired
			}
			out = *r.Terminal
			return nil
		}
		if !errors.Is(err, os.ErrNotExist) {
			return err
		}
		r = Attempt{Version: contract.Version, Approval: a, Digest: hash}
		out = terminalOutcome(r, contract.Canceled, nil)
		r.Terminal = &out
		return s.publishUnstartedFence(r)
	})
	return out, err
}

// Only an initializer without a ready marker can be completed here. A ready
// attempt with missing history must be reconciled physically, never recreated.
func (s *AttemptStore) publishUnstartedFence(r Attempt) error {
	if _, err := s.path.ReadFileLimit(attemptReady, 64); err == nil {
		return ErrReconciliationRequired
	} else if !errors.Is(err, os.ErrNotExist) {
		return err
	}
	claim, err := s.path.ReadFileLimit(attemptClaim, 64)
	switch {
	case errors.Is(err, os.ErrNotExist):
		if err := s.path.CompareAndSwap(attemptClaim, nil, []byte(r.Digest)); err != nil {
			return err
		}
	case err != nil:
		return err
	case string(claim) != r.Digest:
		return ErrReconciliationRequired
	}
	old, err := s.path.ReadFileLimit(attemptFile, 1<<20)
	if err == nil {
		var prior Attempt
		if json.Unmarshal(old, &prior) != nil || prior.validate() != nil || prior.Digest != r.Digest || prior.Started != nil || (prior.Terminal != nil && prior.Terminal.Kind != contract.Canceled) {
			return ErrReconciliationRequired
		}
		canonical, e := json.Marshal(prior)
		if e != nil || string(canonical) != string(old) {
			return fmt.Errorf("noncanonical pending execution")
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return err
	}
	if err := s.save(old, r); err != nil {
		return err
	}
	return s.path.CompareAndSwap(attemptReady, nil, []byte(r.Digest))
}
