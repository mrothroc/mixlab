package workerhost

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"

	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/workerhost/contract"
	"github.com/mrothroc/mixlab/workerjob"
)

const attemptFile = "execution-attempt.json"
const attemptLock = "execution-attempt.lock"
const attemptClaim = "execution-attempt.claim"
const attemptReady = "execution-attempt.ready"

var ErrReconciliationRequired = errors.New("execution attempt requires child reconciliation; restart is forbidden")

type Attempt struct {
	Version  string            `json:"version"`
	Approval contract.Approved `json:"approval"`
	Fence    *contract.Fence   `json:"fence"`
	Digest   string            `json:"digest"`
	Started  *contract.Outcome `json:"started"`
	Terminal *contract.Outcome `json:"terminal"`
}

// AttemptStore owns only execution state. It cannot write a node job or lease.
// A fresh approved attempt is recorded before invoking any launcher effect.
type AttemptStore struct{ path statehome.Path }

func NewAttemptStore(path statehome.Path) (*AttemptStore, error) {
	if path.Kind() != statehome.Worker {
		return nil, fmt.Errorf("execution journal requires worker-owned state")
	}
	if err := path.Validate(); err != nil {
		return nil, err
	}
	return &AttemptStore{path: path}, nil
}

func (r Attempt) validate() error {
	if r.Version != contract.Version {
		return fmt.Errorf("unsupported execution journal")
	}
	if r.Fence != nil {
		return r.validateFence()
	}
	if err := r.Approval.Validate(); err != nil {
		return err
	}
	digest, err := workerjob.Digest(r.Approval)
	if err != nil || digest != r.Digest {
		return fmt.Errorf("execution approval digest mismatch")
	}
	a := r.Approval.Assignment
	for _, o := range []*contract.Outcome{r.Started, r.Terminal} {
		if o == nil {
			continue
		}
		if err := o.Validate(); err != nil {
			return err
		}
		if o.JobID != a.JobID || o.AttemptID != a.AttemptID || o.ManifestHash != r.Approval.ManifestHash || o.ApprovalHash != digest {
			return fmt.Errorf("execution outcome binding mismatch")
		}
	}
	if r.Started != nil && r.Started.Kind != contract.Started {
		return fmt.Errorf("invalid started history")
	}
	if r.Terminal != nil {
		if !r.Terminal.Terminal() || (r.Started == nil && r.Terminal.Version != 1) || (r.Started != nil && (r.Terminal.Version != 2 || r.Terminal.PID != r.Started.PID)) {
			return fmt.Errorf("invalid terminal history")
		}
	}
	return nil
}

func (s *AttemptStore) load() ([]byte, Attempt, error) {
	ready, err := s.path.ReadFileLimit(attemptReady, 64)
	if err != nil {
		return nil, Attempt{}, err
	}
	b, err := s.path.ReadFileLimit(attemptFile, 1<<20)
	if err != nil {
		return nil, Attempt{}, err
	}
	var r Attempt
	if err := json.Unmarshal(b, &r); err != nil {
		return nil, r, err
	}
	canonical, err := json.Marshal(r)
	if err != nil || !bytes.Equal(canonical, b) {
		return nil, r, fmt.Errorf("noncanonical execution journal")
	}
	claim, err := s.path.ReadFileLimit(attemptClaim, 64)
	if err != nil || string(claim) != r.Digest || !bytes.Equal(ready, claim) {
		return nil, r, fmt.Errorf("execution claim missing or changed: %w", err)
	}
	return b, r, r.validate()
}

func (s *AttemptStore) save(old []byte, r Attempt) error {
	if err := r.validate(); err != nil {
		return err
	}
	b, err := json.Marshal(r)
	if err != nil {
		return err
	}
	return s.path.CompareAndSwap(attemptFile, old, b)
}

func (s *AttemptStore) Status() (Attempt, error) {
	_, r, err := s.load()
	return r, err
}

// Runner is a local hosting port. Run must not return until every spawned child
// has been reaped/fenced. The started callback must succeed before delivering
// the assignment. Reconcile must corroborate liveness independently of saved
// PIDs; false means uncertain/live and preserves the admission fence.
type Runner interface {
	Run(context.Context, contract.Approved, func(int) error) error
	Reconcile(context.Context, Attempt) (noChild bool, err error)
}

func freezeApproval(a contract.Approved) (contract.Approved, string, error) {
	b, err := json.Marshal(a)
	if err != nil {
		return a, "", err
	}
	var copy contract.Approved
	if err := json.Unmarshal(b, &copy); err != nil {
		return copy, "", err
	}
	if err := copy.Validate(); err != nil {
		return copy, "", err
	}
	hash, err := workerjob.Digest(copy)
	return copy, hash, err
}

// Run permits one launcher invocation for an exact durable attempt, even if
// the caller retries after losing the response. Existing nonterminal intent
// always goes to reconciliation, never back to the launcher.
func (s *AttemptStore) Run(ctx context.Context, a contract.Approved, runner Runner) (out contract.Outcome, err error) {
	a, hash, err := freezeApproval(a)
	if err != nil {
		return out, err
	}
	if runner == nil {
		return out, fmt.Errorf("missing execution runner")
	}
	err = s.path.WithProcessLock(ctx, attemptLock, func() error {
		old, r, err := s.load()
		if err == nil {
			if r.Digest != hash {
				return fmt.Errorf("attempt directory already bound to another approval")
			}
			if r.Terminal != nil {
				out = *r.Terminal
				return nil
			}
			return ErrReconciliationRequired
		}
		if !errors.Is(err, os.ErrNotExist) {
			return err
		}
		// A permanent claim prevents a missing published journal from being
		// interpreted as a new attempt. A crash in this initialization gap is
		// deliberately fail-closed, not permission to spawn again.
		if err := s.path.CompareAndSwap(attemptClaim, nil, []byte(hash)); err != nil {
			return fmt.Errorf("execution history missing or initialization interrupted: %w", err)
		}
		r = Attempt{Version: contract.Version, Approval: a, Digest: hash}
		if err := s.save(nil, r); err != nil {
			return err
		}
		// No launcher is called until initialization is durably marked ready.
		if err := s.path.CompareAndSwap(attemptReady, nil, []byte(hash)); err != nil {
			return err
		}
		old, r, err = s.load()
		if err != nil {
			return err
		}
		runErr := runner.Run(ctx, a, func(pid int) error {
			if r.Started != nil {
				return fmt.Errorf("duplicate started outcome")
			}
			o := contract.Outcome{JobID: a.Assignment.JobID, AttemptID: a.Assignment.AttemptID, ManifestHash: a.ManifestHash, ApprovalHash: hash, Kind: contract.Started, Version: 1, PID: pid}
			r.Started = &o
			if err := s.save(old, r); err != nil {
				return err
			}
			old, r, err = s.load()
			return err
		})
		if errors.Is(runErr, ErrReconciliationRequired) {
			// A hosting adapter must report uncertainty, not manufacture a
			// no-child terminal outcome after an incomplete physical teardown.
			return runErr
		}
		// Reload durable state: a failed started publication must not become a
		// fabricated successful start just because the callback mutated memory.
		old, r, err = s.load()
		if err != nil {
			return err
		}
		if runErr == nil && r.Started == nil {
			runErr = fmt.Errorf("worker returned success without a started outcome")
		}
		kind := contract.Exited
		if runErr != nil {
			kind = contract.Failed
			if errors.Is(ctx.Err(), context.Canceled) {
				kind = contract.Canceled
			}
		}
		out = terminalOutcome(r, kind, runErr)
		r.Terminal = &out
		return s.save(old, r)
	})
	return out, err
}

func terminalOutcome(r Attempt, kind string, failure error) contract.Outcome {
	a := r.Approval.Assignment
	o := contract.Outcome{JobID: a.JobID, AttemptID: a.AttemptID, ManifestHash: r.Approval.ManifestHash, ApprovalHash: r.Digest, Version: 1, Kind: kind, NoChild: true}
	if r.Started != nil {
		o.PID = r.Started.PID
		o.Version = 2
	}
	if failure != nil {
		o.Error = failure.Error()
		if len(o.Error) > 4096 {
			o.Error = "worker failed; see bounded attempt log"
		}
	}
	return o
}

func (s *AttemptStore) Reconcile(ctx context.Context, runner Runner) (out contract.Outcome, err error) {
	if runner == nil {
		return out, fmt.Errorf("missing execution reconciler")
	}
	err = s.path.WithProcessLock(ctx, attemptLock, func() error {
		old, r, err := s.load()
		if err != nil {
			return err
		}
		if r.Terminal != nil {
			out = *r.Terminal
			return nil
		}
		noChild, err := runner.Reconcile(ctx, r)
		if err != nil {
			return err
		}
		if !noChild {
			return ErrReconciliationRequired
		}
		out = terminalOutcome(r, contract.Failed, errors.New("execution supervisor interrupted; attempt will not restart"))
		r.Terminal = &out
		return s.save(old, r)
	})
	return out, err
}
