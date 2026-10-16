package coordinator

import (
	"context"
	"fmt"

	"github.com/mrothroc/mixlab/diloco/learner"
	"github.com/mrothroc/mixlab/diloco/run"
)

// RecordStart commits a worker's durable replay point before it consumes data.
// Callers authenticate the current attempt before invoking this domain port.
func (m Model) RecordStart(ctx context.Context, start learner.Start) error {
	s, err := m.load(ctx, start.Binding.RunID)
	if err != nil {
		return err
	}
	return m.recordStart(ctx, s, start)
}

// RecordStartSubmission fences an authenticated attempt on the same revision
// as publication. A replaced process cannot install a different replay point.
func (m Model) RecordStartSubmission(ctx context.Context, q learner.StartSubmission) error {
	if err := q.Validate(); err != nil {
		return err
	}
	s, err := m.load(ctx, q.Start.Binding.RunID)
	if err != nil {
		return err
	}
	if s.Commit == nil || s.Commit.Binding != q.Start.Binding {
		return fmt.Errorf("start run mismatch")
	}
	id, g, err := s.CurrentAttempt(q.Start.WorkerID)
	if err != nil {
		return err
	}
	if id != q.AttemptID || g != q.WorkloadGeneration {
		return fmt.Errorf("stale learner start attempt")
	}
	found := false
	for _, slot := range s.Plan.Slots {
		if slot.Member.WorkerID == q.Start.WorkerID && slot.Allocation == q.Allocation {
			found = true
		}
	}
	if !found {
		return fmt.Errorf("start allocation mismatch")
	}
	return m.recordStart(ctx, s, q.Start)
}

func (m Model) recordStart(ctx context.Context, s State, start learner.Start) error {
	var err error
	if err = start.Validate(); err != nil {
		return err
	}
	if s.Phase != Open || s.Commit == nil || s.Commit.Binding != start.Binding {
		return fmt.Errorf("start commitment requires an open committed round")
	}
	for _, prior := range s.Starts {
		if prior.WorkerID == start.WorkerID {
			if prior == start {
				return nil
			}
			return ErrConflict
		}
	}
	index := -1
	for i, a := range s.Assignments {
		h, e := run.Digest(a)
		if e != nil {
			return e
		}
		if a.WorkerID == start.WorkerID && h == start.AssignmentHash {
			if m.Clock.Unix() >= a.ExpiresAt {
				return fmt.Errorf("round deadline expired")
			}
			index = i
		}
	}
	if index < 0 {
		return fmt.Errorf("start does not match assigned worker")
	}
	if err = m.Allocations.Ready(ctx, s.Plan.Slots[index]); err != nil {
		return err
	}
	next := s
	next.Starts = append(append([]learner.Start{}, s.Starts...), start)
	return m.save(ctx, s, Change{Next: next})
}

func (s State) validateStarts() error {
	seen := map[string]bool{}
	for _, start := range s.Starts {
		if err := start.Validate(); err != nil {
			return err
		}
		if s.Commit == nil || s.Commit.Binding != start.Binding || seen[start.WorkerID] {
			return fmt.Errorf("invalid/duplicate start commitment")
		}
		seen[start.WorkerID] = true
		found := false
		for _, a := range s.Assignments {
			h, err := run.Digest(a)
			if err != nil {
				return err
			}
			if a.WorkerID == start.WorkerID && h == start.AssignmentHash {
				found = true
			}
		}
		if !found {
			return fmt.Errorf("start commitment without assignment")
		}
	}
	return nil
}
