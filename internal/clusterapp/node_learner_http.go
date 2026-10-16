package clusterapp

import (
	"context"
	"net/http"

	"github.com/mrothroc/mixlab/diloco/authenticated"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/workload"
)

type NodeLearnerPrepareRequest struct {
	ExpectedLeaseVersion uint64                `json:"expected_lease_version"`
	Signed               nodejob.SignedLearner `json:"signed"`
}
type NodeLearnerPrepared struct {
	Registration run.Registration       `json:"registration"`
	Request      workload.SignedRequest `json:"request"`
}
type NodeLearnerCredentialRequest struct {
	Grant      workload.Grant  `json:"grant"`
	Credential workload.Result `json:"credential"`
}
type NodeLearnerStartRequest struct {
	Key string `json:"key"`
}
type NodeLearnerAck struct {
	Job string `json:"job"`
}

// Remote commands cannot supply assignments, local dataset/private paths or
// executable arguments. Those are resolved exclusively by the node owner.
type NodeLearnerOperations interface {
	PrepareLearner(context.Context, trust.AuthenticatedPrincipal, NodeLearnerPrepareRequest) (NodeLearnerPrepared, error)
	CredentialLearner(context.Context, trust.AuthenticatedPrincipal, string, NodeLearnerCredentialRequest) error
	StartLearner(context.Context, trust.AuthenticatedPrincipal, string, NodeLearnerStartRequest) error
	FinalizeLearner(context.Context, trust.AuthenticatedPrincipal, string, authenticated.Commit) error
}

func (h *nodeHTTP) learner(w http.ResponseWriter, r *http.Request, actor trust.AuthenticatedPrincipal, parts []string) {
	owner, ok := h.jobs.(NodeLearnerOperations)
	if !ok || len(parts) != 3 || !nodeRouteID(parts[1]) {
		http.NotFound(w, r)
		return
	}
	id := parts[1]
	if actor.Role != trust.Controller && (actor.Role != trust.Coordinator || parts[2] != "finalize") {
		http.Error(w, "learner operation is not authorized for this role", http.StatusForbidden)
		return
	}
	switch {
	case parts[2] == "prepare" && r.Method == http.MethodPut:
		var q NodeLearnerPrepareRequest
		if !h.decode(w, r, &q, 2<<20, &actor) {
			return
		}
		if q.ExpectedLeaseVersion == 0 || q.Signed.Manifest.Slot.Allocation.JobID != id {
			http.Error(w, "learner preparation binding mismatch", http.StatusBadRequest)
			return
		}
		out, err := owner.PrepareLearner(r.Context(), actor, q)
		nodeResponse(w, out, err)
	case parts[2] == "credential" && r.Method == http.MethodPut:
		var q NodeLearnerCredentialRequest
		if !h.decode(w, r, &q, 1<<20, &actor) {
			return
		}
		nodeResponse(w, NodeLearnerAck{id}, owner.CredentialLearner(r.Context(), actor, id, q))
	case parts[2] == "start" && r.Method == http.MethodPost:
		var q NodeLearnerStartRequest
		if !h.decode(w, r, &q, 4096, &actor) {
			return
		}
		if !nodeRouteID(q.Key) {
			http.Error(w, "start key required", http.StatusBadRequest)
			return
		}
		nodeResponse(w, NodeLearnerAck{id}, owner.StartLearner(r.Context(), actor, id, q))
	case parts[2] == "finalize" && r.Method == http.MethodPut:
		var q authenticated.Commit
		if !h.decode(w, r, &q, 2<<20, &actor) {
			return
		}
		nodeResponse(w, NodeLearnerAck{id}, owner.FinalizeLearner(r.Context(), actor, id, q))
	default:
		http.NotFound(w, r)
	}
}
