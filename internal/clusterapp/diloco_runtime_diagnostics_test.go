package clusterapp

import (
	"strings"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/admission"
)

func TestAllocationFailureDeadlineDiagnostic(t *testing.T) {
	_, f := admissionLeaseSetup(t, 2)
	// Immutable admission deadline 2800 + runtime 3600 + cleanup 300.
	deadline := time.Unix(6700, 0)
	for _, tc := range []struct {
		name   string
		now    time.Time
		status string
	}{
		{"capped-renewal-before-expiry", deadline.Add(-5 * time.Minute), "not yet reached"},
		{"exact-expiry", deadline, "reached"},
		{"after-recovery", deadline.Add(time.Second), "reached"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			reason := allocationFailureReason(f.q, tc.now)
			for _, want := range []string{"allocation lease maintenance permanently rejected", deadline.UTC().Format(time.RFC3339), "(" + tc.status + ";", "runtime limit 3600s", "recovery does not extend deadline"} {
				if !strings.Contains(reason, want) {
					t.Fatalf("missing %q: %s", want, reason)
				}
			}
		})
	}
	if got := allocationFailureReason(admission.Submission{}, time.Now()); got != "allocation lease maintenance permanently rejected" {
		t.Fatalf("invented deadline for invalid submission: %s", got)
	}
}
