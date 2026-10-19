package nodeagent

import "time"

// LearnerDrainTimeout bounds successful shutdown, not work or lease renewal.
// It allows one semantic transport retry budget plus process teardown. Terminal
// reads still require current trust and an exact hosting binding.
const LearnerDrainTimeout = 90 * time.Second

func (x LocalExecution) LearnerDraining(now time.Time) bool {
	return x.Lease.State == Releasing && x.Lease.CleanupReason == "completed" &&
		x.Lease.DrainUntil > now.Unix() && x.Learner != nil && !x.Learner.Stopped &&
		x.Learner.Assignment != nil && x.Learner.Assignment.AttemptID == x.Lease.CompletedAttempt
}
