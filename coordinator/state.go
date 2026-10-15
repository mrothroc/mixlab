package coordinator

import (
	"encoding/hex"
	"fmt"

	"github.com/mrothroc/mixlab/diloco/learner"
	"github.com/mrothroc/mixlab/diloco/run"
)

// Validate checks restored aggregate structure, not external trust or blob
// integrity. Persistence adapters must not recover an unchecked partial state.
func (s State) Validate() error {
	if s.Revision == 0 {
		return fmt.Errorf("missing repository revision")
	}
	if e := s.Plan.Validate(); e != nil {
		return e
	}
	if e := validatePolicy(s.Policy, s.Plan); e != nil {
		return e
	}
	if s.Phase == Staged {
		if s.Commit != nil || s.Round != 0 || len(s.Assignments) != 0 || len(s.Updates) != 0 || s.Candidate != nil || s.Evaluation != nil {
			return fmt.Errorf("staged plan contains round authority")
		}
		return nil
	}
	if s.Commit == nil {
		return fmt.Errorf("committed state requires run commit")
	}
	if e := s.Commit.ValidateFor(s.Plan); e != nil {
		return e
	}
	if e := checkBlob(s.Base, "model_f32"); e != nil {
		return e
	}
	if e := checkBlob(s.Momentum, "outer_state_f32"); e != nil {
		return e
	}
	if s.Round > s.Plan.Outer.TotalSteps/s.Plan.Outer.InnerSteps || len(s.PrivateHashes) != len(s.Plan.Slots) {
		return fmt.Errorf("invalid round/private state count")
	}
	for _, h := range s.PrivateHashes {
		if (s.Round > 0 || s.Phase == Evaluating || h != "") && !validHash(h) {
			return fmt.Errorf("missing previous private state")
		}
	}
	switch s.Phase {
	case Ready:
		if s.Round >= s.Plan.Outer.TotalSteps/s.Plan.Outer.InnerSteps {
			return fmt.Errorf("completed run is not ready")
		}
		return nil
	case Complete:
		if s.Round == 0 {
			return fmt.Errorf("completed without evaluation")
		}
		return nil
	case Failed:
		if s.Failure == "" || len(s.Failure) > 4096 {
			return fmt.Errorf("missing failure reason")
		}
		return nil
	case Open, Aggregating, Publishing, Evaluating:
	default:
		return fmt.Errorf("unknown coordinator phase")
	}
	if s.Round >= s.Plan.Outer.TotalSteps/s.Plan.Outer.InnerSteps || len(s.Assignments) != len(s.Plan.Slots) {
		return fmt.Errorf("incomplete active assignments")
	}
	for i, a := range s.Assignments {
		if e := a.ValidateFor(s.Plan, *s.Commit); e != nil {
			return e
		}
		if a.WorkerID != s.Plan.Slots[i].Member.WorkerID || a.Round != s.Round {
			return fmt.Errorf("assignment order/round mismatch")
		}
		if s.Phase != Evaluating && (a.BaseModel != s.Base || a.OuterState != s.Momentum || a.PreviousPrivateHash != s.PrivateHashes[i]) {
			return fmt.Errorf("assignment differs from current global/private state")
		}
	}
	byID := map[string]learner.Update{}
	updateIDs := map[string]bool{}
	var totalTokens uint64
	for _, u := range s.Updates {
		if _, ok := byID[u.WorkerID]; ok || updateIDs[u.ID] {
			return fmt.Errorf("duplicate accepted update")
		}
		index := -1
		for i, a := range s.Assignments {
			if a.WorkerID == u.WorkerID {
				index = i
			}
		}
		if index < 0 {
			return fmt.Errorf("accepted worker outside cohort")
		}
		if e := u.ValidateFor(s.Assignments[index]); e != nil {
			return e
		}
		member := s.Plan.Slots[index].Member
		if u.ParticipantID != member.ParticipantID || u.IslandID != member.IslandID {
			return fmt.Errorf("accepted update identity mismatch")
		}
		if u.EffectiveTokens > ^uint64(0)-totalTokens {
			return fmt.Errorf("accepted token count overflow")
		}
		totalTokens += u.EffectiveTokens
		byID[u.WorkerID] = u
		updateIDs[u.ID] = true
	}
	if s.Phase != Open && len(byID) != len(s.Plan.Slots) {
		return fmt.Errorf("aggregation without complete cohort")
	}
	if s.Phase == Publishing || s.Phase == Evaluating {
		if s.Candidate == nil {
			return fmt.Errorf("missing candidate")
		}
		r := s.Candidate
		if e := r.ValidateFor(s.Plan, *s.Commit); e != nil {
			return e
		}
		if r.Round != s.Round {
			return fmt.Errorf("candidate round mismatch")
		}
		var tokens uint64
		for _, receipt := range r.Accepted {
			u := byID[receipt.WorkerID]
			h, e := run.Digest(u)
			if e != nil {
				return e
			}
			if receipt.UpdateHash != h || receipt.UpdateID != u.ID || receipt.PrivateHash != u.PrivateHash || u.EffectiveTokens > ^uint64(0)-tokens {
				return fmt.Errorf("candidate accepted set mismatch")
			}
			tokens += u.EffectiveTokens
		}
		if tokens != r.EffectiveTokens {
			return fmt.Errorf("candidate token count mismatch")
		}
	}
	if s.Phase == Evaluating {
		if s.Evaluation == nil {
			return fmt.Errorf("missing evaluation")
		}
		a := s.Evaluation
		if e := a.Validate(); e != nil {
			return e
		}
		if a.Model != s.Base || s.Base != s.Candidate.Model || s.Momentum != s.Candidate.OuterState || a.Round != s.Round || a.Binding != s.Commit.Binding || a.WorkerID != s.Policy.WorkerID {
			return fmt.Errorf("evaluation differs from published global state")
		}
		p := s.Policy
		if a.Coordinator != s.Plan.Coordinator || a.ValidationHash != p.ValidationHash || a.Split != p.Split || a.BatchTokens != p.BatchTokens || a.SequenceLength != p.SequenceLength || a.Batches != p.Batches || a.ExpiresAt-a.IssuedAt != p.TimeoutSeconds {
			return fmt.Errorf("evaluation differs from immutable policy")
		}
		for _, r := range s.Commit.Registrations {
			if r.Slot.Member.WorkerID == a.WorkerID && (a.Precision != r.Precision || a.PrecisionHash != r.PrecisionHash) {
				return fmt.Errorf("evaluation precision differs from committed worker")
			}
		}
		for i, u := range s.Candidate.Accepted {
			if u.PrivateHash != s.PrivateHashes[i] {
				return fmt.Errorf("published private hash mismatch")
			}
		}
	}
	return nil
}

func validHash(h string) bool {
	b, e := hex.DecodeString(h)
	return e == nil && len(b) == 32 && hex.EncodeToString(b) == h
}
