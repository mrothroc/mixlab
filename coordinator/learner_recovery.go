package coordinator

import (
	"context"
	"fmt"

	"github.com/mrothroc/mixlab/diloco/run"
)

const LearnerRestartLimit = 3
const LearnerRecoverySeconds int64 = 60

// LearnerRecovery is persisted before remote effects. Restart never resets its
// deadline or chooses another attempt after an ambiguous response.
type LearnerRecovery struct {
	WorkerID    string `json:"worker_id"`
	Previous    string `json:"previous"`
	Attempt     string `json:"attempt"`
	PrivateHash string `json:"private_hash"`
	Started     int64  `json:"started"`
}

func (s State) ExpectedPrivate(worker string) (string, error) {
	for i, slot := range s.Plan.Slots {
		if slot.Member.WorkerID != worker {
			continue
		}
		h := s.PrivateHashes[i]
		if s.Phase == Open {
			for _, start := range s.Starts {
				if start.WorkerID == worker {
					h = start.PrivateHash
				}
			}
			for _, u := range s.Updates {
				if u.WorkerID == worker {
					h = u.PrivateHash
				}
			}
		}
		return h, nil
	}
	return "", fmt.Errorf("unknown recovery worker")
}

// Begin's CAS freezes this worker's private head until Bind installs the new
// attempt. Writes loaded before Begin lose their own publication CAS.
func (s State) checkRecoveryWrite(worker string) error {
	if s.Recovery == nil || s.Recovery.WorkerID != worker {
		return nil
	}
	current, _, err := s.CurrentAttempt(worker)
	if err != nil {
		return err
	}
	if current == s.Recovery.Previous {
		return fmt.Errorf("learner writes fenced by pending recovery")
	}
	return nil
}

func (m Model) BeginLearnerRecovery(ctx context.Context, id, worker string) error {
	s, err := m.load(ctx, id)
	if err != nil {
		return err
	}
	if s.Recovery != nil {
		return nil
	}
	if s.Phase != Ready && s.Phase != Open && s.Phase != Evaluating {
		return fmt.Errorf("run cannot recover now")
	}
	count := 0
	for _, a := range s.Attempts {
		if a.WorkerID == worker {
			count++
		}
	}
	if count >= LearnerRestartLimit {
		return m.Fail(ctx, id, "learner recovery exhausted same-allocation restart limit")
	}
	previous, _, err := s.CurrentAttempt(worker)
	if err != nil {
		return err
	}
	private, err := s.ExpectedPrivate(worker)
	if err != nil {
		return err
	}
	if !validHash(private) {
		return m.Fail(ctx, id, "learner recovery requires a committed private snapshot; fresh RNG initialization forbidden")
	}
	h, _ := run.Digest(struct{ Kind, Run, Worker, Previous string }{"learner-recovery-v1", id, worker, previous})
	next := s
	next.Recovery = &LearnerRecovery{WorkerID: worker, Previous: previous, Attempt: h[:32], PrivateHash: private, Started: m.Clock.Unix()}
	return m.save(ctx, s, Change{Next: next})
}

func (m Model) FinishLearnerRecovery(ctx context.Context, id, attempt string) error {
	s, err := m.load(ctx, id)
	if err != nil {
		return err
	}
	if s.Recovery == nil {
		return nil
	}
	if s.Recovery.Attempt != attempt {
		return ErrConflict
	}
	next := s
	next.Recovery = nil
	return m.save(ctx, s, Change{Next: next})
}

func (s State) validateRecovery() error {
	if s.LearnerUnavailableAt < 0 {
		return fmt.Errorf("invalid learner availability deadline")
	}
	if s.Recovery == nil {
		return nil
	}
	r := s.Recovery
	if s.Commit == nil || r.Started <= 0 || !validHash(r.PrivateHash) {
		return fmt.Errorf("invalid learner recovery intent")
	}
	h, _ := run.Digest(struct{ Kind, Run, Worker, Previous string }{"learner-recovery-v1", s.Plan.Cohort.RunID, r.WorkerID, r.Previous})
	current, _, err := s.CurrentAttempt(r.WorkerID)
	if err != nil {
		return err
	}
	if h[:32] != r.Attempt || (current != r.Previous && current != r.Attempt) {
		return fmt.Errorf("recovery attempt fence changed")
	}
	return nil
}

// ObserveLearnerAvailability bounds a management outage across service restarts.
func (m Model) ObserveLearnerAvailability(ctx context.Context, id string, unavailable bool) error {
	s, err := m.load(ctx, id)
	if err != nil {
		return err
	}
	if unavailable && s.LearnerUnavailableAt != 0 {
		if m.Clock.Unix()-s.LearnerUnavailableAt >= LearnerRecoverySeconds {
			return m.Fail(ctx, id, "learner recovery status unavailable beyond recovery budget")
		}
		return nil
	}
	if !unavailable && s.LearnerUnavailableAt == 0 {
		return nil
	}
	next := s
	next.LearnerUnavailableAt = 0
	if unavailable {
		next.LearnerUnavailableAt = m.Clock.Unix()
	}
	return m.save(ctx, s, Change{Next: next})
}
