package clusterapp

import (
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"

	"github.com/mrothroc/mixlab/admission"
	"github.com/mrothroc/mixlab/internal/clusterdiagnostic"
)

func admissionRetryPhase(v admission.Snapshot) string {
	switch {
	case v.Committed:
		return "committed"
	case v.Done:
		return "aborted"
	case v.Aborting:
		return "aborting"
	default:
		return "admitting"
	}
}

// Two seconds initially, exponential growth to 48s plus deterministic jitter
// (up to 25%), at most 60s. This fits the existing lease-renewal lead budget.
// Per-run jitter avoids synchronized retries without consuming training RNG.
func admissionRetryDelay(id string, failures uint32, at int64) int64 {
	base := min(int64(2)<<min(failures-1, 5), 48)
	h := sha256.Sum256([]byte(fmt.Sprintf("%s/%d/%d", id, failures, at)))
	return base + int64(binary.LittleEndian.Uint64(h[:8])%uint64(base/4+1))
}

func serviceObservation(id string, at int64, previous admission.Observation, before, after admission.Snapshot, retryAllowed bool, effect error) (admission.Observation, bool) {
	v := admission.Observation{At: at, EventAt: previous.EventAt}
	if effect == nil {
		return v, false
	}
	v.Error = effect.Error()
	if len(v.Error) > 4096 {
		v.Error = "service effect failed; diagnostic exceeded limit"
	}
	v.FailureClass = clusterdiagnostic.Classify(effect).Reason
	var scoped interface {
		error
		NodeID() string
	}
	if errors.As(effect, &scoped) && nodeRouteID(scoped.NodeID()) {
		v.FailureNode = scoped.NodeID()
		// Classify this node's cause, not a higher-priority error belonging to
		// another node in a joined lease-maintenance result.
		v.FailureClass = clusterdiagnostic.Classify(scoped).Reason
	}
	// Actual saga progress must not delay handoff or cleanup of reachable nodes.
	// Abort scan rotation is intentionally absent from Snapshot: rotating an
	// unreachable target is not progress. Never infer ABORTED from elapsed time,
	// even after lease expiry: ambiguous effects still need node acknowledgement.
	// Live coordinator work keeps its normal tick cadence even when recovery
	// errors persist alongside round progress. Only durable terminal state can
	// enable committed-run backoff; admission snapshots cannot establish it.
	if retryAllowed && !admissionMadeProgress(before, after) {
		v.RetryFailures = min(previous.RetryFailures+1, 16)
		v.RetryAt = at + admissionRetryDelay(id, v.RetryFailures, at)
		v.RetryPhase = admissionRetryPhase(after)
	}
	emit := previous.EventAt == 0 || at < previous.EventAt || at-previous.EventAt >= 60
	if emit {
		v.EventAt = at
	}
	return v, emit
}

func admissionMadeProgress(a, b admission.Snapshot) bool {
	if a.Committed != b.Committed || a.Done != b.Done || a.Aborting != b.Aborting || (a.Plan == nil) != (b.Plan == nil) || (a.Commit == nil) != (b.Commit == nil) {
		return true
	}
	for i, n := range a.Nodes {
		if i >= len(b.Nodes) {
			return true
		}
		m := b.Nodes[i]
		if (n.Lease == nil) != (m.Lease == nil) || n.Finalized != m.Finalized || n.Released != m.Released {
			return true
		}
	}
	return false
}
