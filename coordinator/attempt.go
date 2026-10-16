package coordinator

import (
	"context"
	"fmt"

	"github.com/mrothroc/mixlab/diloco/learner"
	"github.com/mrothroc/mixlab/diloco/run"
)

// RecoveryAuthorization is implemented by an authenticated node adapter. It
// verifies current node proof, exact authorization hash and stopped old child;
// neither a hosting outcome alone nor a submitter can authorize replacement.
type RecoveryAuthorization interface {
	Check(context.Context, run.AttemptBinding) error
}

func (s State) CurrentAttempt(worker string) (string, uint64, error) {
	if s.Commit == nil {
		return "", 0, fmt.Errorf("uncommitted run")
	}
	for i := len(s.Attempts) - 1; i >= 0; i-- {
		a := s.Attempts[i]
		if a.WorkerID == worker {
			return a.AttemptID, a.WorkloadGeneration, nil
		}
	}
	for _, r := range s.Commit.Registrations {
		if r.Slot.Member.WorkerID == worker {
			return r.AttemptID, r.WorkloadGeneration, nil
		}
	}
	return "", 0, fmt.Errorf("unknown worker")
}
func (m Model) BindAttempt(ctx context.Context, a run.AttemptBinding, auth RecoveryAuthorization) error {
	if auth == nil {
		return fmt.Errorf("node recovery authorization required")
	}
	s, err := m.load(ctx, a.Binding.RunID)
	if err != nil {
		return err
	}
	if s.Commit == nil || (s.Phase != Ready && s.Phase != Open) {
		return fmt.Errorf("run cannot replace an attempt now")
	}
	if err = a.Validate(); err != nil {
		return err
	}
	if err = auth.Check(ctx, a); err != nil {
		return err
	}
	for _, prior := range s.Attempts {
		if prior == a {
			current, _, e := s.CurrentAttempt(a.WorkerID)
			if e != nil {
				return e
			}
			if current != a.AttemptID {
				return fmt.Errorf("retry already superseded")
			}
			return nil
		}
	}
	previous, generation, err := s.CurrentAttempt(a.WorkerID)
	if err != nil {
		return err
	}
	index := -1
	for i, slot := range s.Plan.Slots {
		if slot.Member.WorkerID == a.WorkerID {
			index = i
		}
	}
	if index < 0 {
		return fmt.Errorf("unknown worker")
	}
	expectedPrivate := s.PrivateHashes[index]
	for _, start := range s.Starts {
		if start.WorkerID == a.WorkerID {
			expectedPrivate = start.PrivateHash
		}
	}
	for _, u := range s.Updates {
		if u.WorkerID == a.WorkerID {
			// A lost receipt is recoverable without repeating local work. The
			// accepted update already binds the completed private checkpoint.
			expectedPrivate = u.PrivateHash
		}
	}
	if err = a.ValidateFor(*s.Commit, previous, generation, expectedPrivate); err != nil {
		return err
	}
	if err = m.Allocations.Ready(ctx, s.Plan.Slots[index]); err != nil {
		return err
	}
	if len(s.Attempts) >= 1024 {
		return fmt.Errorf("retry history capacity exceeded")
	}
	next := s
	next.Attempts = append(append([]run.AttemptBinding{}, s.Attempts...), a)
	return m.save(ctx, s, Change{Next: next})
}

// CheckSubmission checks this snapshot's operational fence. Update adapters use
// AcceptSubmission to bind this check atomically to publication. Assignment and
// RNG identity remain unchanged by recovery.
func (s State) CheckSubmission(q learner.Submission) error {
	if err := q.Validate(); err != nil {
		return err
	}
	if s.Commit == nil || q.Update.Binding != s.Commit.Binding {
		return fmt.Errorf("submission run mismatch")
	}
	id, generation, err := s.CurrentAttempt(q.Update.WorkerID)
	if err != nil {
		return err
	}
	if q.AttemptID != id || q.WorkloadGeneration != generation {
		return fmt.Errorf("stale learner attempt")
	}
	for _, slot := range s.Plan.Slots {
		if slot.Member.WorkerID == q.Update.WorkerID && slot.Allocation == q.Allocation {
			return nil
		}
	}
	return fmt.Errorf("submission allocation changed")
}

func (s State) validateAttempts() error {
	if len(s.Attempts) > 1024 {
		return fmt.Errorf("attempt journal capacity exceeded")
	}
	if len(s.Attempts) == 0 {
		return nil
	}
	if s.Commit == nil {
		return fmt.Errorf("uncommitted attempt history")
	}
	prior := s
	prior.Attempts = nil
	for _, a := range s.Attempts {
		id, g, e := prior.CurrentAttempt(a.WorkerID)
		if e != nil {
			return e
		}
		// Historical snapshots may precede the current private state. The exact
		// private hash was checked against the run head at admission time.
		if e = a.ValidateFor(*s.Commit, id, g, a.PrivateHash); e != nil {
			return e
		}
		prior.Attempts = append(prior.Attempts, a)
	}
	return nil
}
