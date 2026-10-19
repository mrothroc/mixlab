package clusterapp

import (
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/admission"
	"github.com/mrothroc/mixlab/nodejob"
)

func allocationFailureReason(q admission.Submission, now time.Time) string {
	const reason = "allocation lease maintenance permanently rejected"
	m := nodejob.Manifest{Created: q.Specification.CreatedAt, Expires: q.Deadline, Limits: q.Launch.Selection.Requirements.Limits}
	deadline, err := m.WorkloadDeadline()
	if err != nil {
		return reason
	}
	status := "not yet reached"
	if !now.Before(deadline) {
		status = "reached"
	}
	// A 409 before this deadline can be a non-extending renewal or another
	// permanent rejection. Do not infer expiry or weaken authentication from it.
	return fmt.Sprintf("%s; original workload deadline %s (%s; runtime limit %ds; recovery does not extend deadline)", reason, deadline.UTC().Format(time.RFC3339), status, m.Limits.RuntimeSeconds)
}
