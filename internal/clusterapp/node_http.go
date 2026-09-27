package clusterapp

import (
	"context"
	"encoding/hex"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/mrothroc/mixlab/grouptransport"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/transport/managedtls"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/workload"
)

type NodePrepareRequest struct {
	ExpectedLeaseVersion uint64         `json:"expected_lease_version"`
	Signed               nodejob.Signed `json:"signed"`
}

type NodeTransportRequest struct {
	ExpectedVersion uint64                `json:"expected_version"`
	Signed          grouptransport.Signed `json:"signed"`
	Grant           workload.Grant        `json:"grant"`
	Credential      workload.Result       `json:"credential"`
}

type NodeCancelRequest struct {
	ExpectedVersion uint64 `json:"expected_version"`
}

// NodeJobOperations is the local application boundary. Implementations verify
// signed admission, resolve local paths, and own relay activation; HTTP never
// accepts worker assignments, filesystem paths, or credential-use operations.
type NodeJobOperations interface {
	Prepare(context.Context, trust.AuthenticatedPrincipal, NodePrepareRequest) (nodeagent.PreparedWorkload, error)
	Transport(context.Context, trust.AuthenticatedPrincipal, string, NodeTransportRequest) (nodeagent.Job, error)
	Start(context.Context, trust.AuthenticatedPrincipal, nodeagent.StartCommand) (nodeagent.Job, error)
}

// NodeManagementHandler must only be served after the execution owner's Ready
// callback. Listener deadlines/rate limits are the server's responsibility.
// Closed canonical schemas deliberately reject extra credential envelopes in
// R1.1, including envelope fields nested within signed manifests.
func NodeManagementHandler(store *nodeagent.Store, policy *managedtls.Policy, clock func() time.Time, jobs NodeJobOperations) (http.Handler, error) {
	if store == nil || policy == nil || clock == nil || jobs == nil {
		return nil, fmt.Errorf("node store, TLS policy, clock and job operations required")
	}
	capabilities, err := NodeCapabilitiesHandler(store, policy, clock)
	if err != nil {
		return nil, err
	}
	h := &nodeHTTP{store, policy, clock, jobs}
	mutating := policy.AuthenticateHTTP(http.HandlerFunc(h.serve))
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == nodeCapabilityRoute {
			capabilities.ServeHTTP(w, r)
			return
		}
		mutating.ServeHTTP(w, r)
	}), nil
}

type nodeHTTP struct {
	store  *nodeagent.Store
	policy *managedtls.Policy
	clock  func() time.Time
	jobs   NodeJobOperations
}

func nodeRouteID(id string) bool {
	b, err := hex.DecodeString(id)
	return err == nil && len(b) == 16 && hex.EncodeToString(b) == id
}

func (h *nodeHTTP) serve(w http.ResponseWriter, r *http.Request) {
	if r.URL.RawQuery != "" || r.URL.RawPath != "" {
		http.NotFound(w, r)
		return
	}
	actor, ok := managedtls.Principal(r.Context())
	if !ok || actor.Role != trust.Controller {
		http.Error(w, "controller required", http.StatusForbidden)
		return
	}
	timeout := nodeOperationTimeout(r.Method, r.URL.Path)
	if timeout > 10*time.Second {
		if err := http.NewResponseController(w).SetWriteDeadline(time.Now().Add(timeout + 5*time.Second)); err != nil && !errors.Is(err, http.ErrNotSupported) {
			http.Error(w, "operation deadline unavailable", http.StatusServiceUnavailable)
			return
		}
	}
	ctx, cancel := context.WithTimeout(r.Context(), timeout)
	defer cancel()
	r = r.WithContext(ctx)
	parts := strings.Split(strings.TrimPrefix(r.URL.Path, "/v1/agent/"), "/")
	if !strings.HasPrefix(r.URL.Path, "/v1/agent/") {
		http.NotFound(w, r)
		return
	}
	if r.Method == http.MethodPost && len(parts) == 1 && parts[0] == "leases" {
		var q nodeagent.Reserve
		if !h.decode(w, r, &q, 4096, &actor) {
			return
		}
		out, err := h.store.Reserve(ctx, actor, q, h.clock())
		nodeResponse(w, out, err)
		return
	}
	if len(parts) < 2 || !nodeRouteID(parts[1]) {
		http.NotFound(w, r)
		return
	}
	id := parts[1]
	if parts[0] == "reservations" && len(parts) == 3 && parts[2] == "abort" && r.Method == http.MethodPost {
		var q nodeagent.Reserve
		if !h.decode(w, r, &q, 4096, &actor) {
			return
		}
		if q.IdempotencyKey != id {
			http.Error(w, "reservation route/body mismatch", http.StatusBadRequest)
			return
		}
		out, err := h.store.AbortReservation(ctx, actor, q, h.clock())
		nodeResponse(w, out, err)
		return
	}
	if parts[0] == "leases" && len(parts) == 2 && r.Method == http.MethodGet {
		if r.ContentLength != 0 || len(r.TransferEncoding) != 0 {
			http.Error(w, "body not allowed", http.StatusBadRequest)
			return
		}
		out, err := h.store.LeaseStatus(ctx, actor, id, h.clock())
		nodeResponse(w, out, err)
		return
	}
	if parts[0] == "leases" && ((len(parts) == 2 && r.Method == http.MethodDelete) || (len(parts) == 3 && parts[2] == "renew" && r.Method == http.MethodPut)) {
		var q nodeagent.LeaseCommand
		if !h.decode(w, r, &q, 4096, &actor) {
			return
		}
		if q.Lease != id {
			http.Error(w, "lease route/body mismatch", http.StatusBadRequest)
			return
		}
		var out nodeagent.Lease
		var err error
		if r.Method == http.MethodDelete {
			out, err = h.store.ReleaseLease(ctx, actor, q, h.clock())
		} else {
			out, err = h.store.RenewLease(ctx, actor, q, h.clock())
		}
		nodeResponse(w, out, err)
		return
	}
	if parts[0] != "jobs" {
		http.NotFound(w, r)
		return
	}
	if len(parts) == 2 && r.Method == http.MethodGet {
		if r.ContentLength != 0 || len(r.TransferEncoding) != 0 {
			http.Error(w, "body not allowed", http.StatusBadRequest)
			return
		}
		out, err := h.store.JobStatus(actor, id, h.clock())
		nodeResponse(w, out, err)
		return
	}
	if len(parts) != 3 {
		http.NotFound(w, r)
		return
	}
	if parts[2] == "output" {
		h.output(w, r, actor, id)
		return
	}
	if parts[2] == "input" && r.Method == http.MethodPut {
		owner, ok := h.jobs.(interface {
			Input(context.Context, trust.AuthenticatedPrincipal, NodeInputRequest) (NodeInputReceipt, error)
		})
		if !ok {
			http.NotFound(w, r)
			return
		}
		var q NodeInputRequest
		if !h.decode(w, r, &q, 2<<20, &actor) {
			return
		}
		if q.Signed.Manifest.Job != id {
			http.Error(w, "job binding mismatch", http.StatusBadRequest)
			return
		}
		out, err := owner.Input(ctx, actor, q)
		nodeResponse(w, out, err)
		return
	}
	switch {
	case parts[2] == "prepare" && r.Method == http.MethodPut:
		var q NodePrepareRequest
		if !h.decode(w, r, &q, nodejob.MaxManifestBytes+trust.MaxTrustBytes+4096, &actor) {
			return
		}
		if q.Signed.Manifest.Job != id || q.ExpectedLeaseVersion == 0 {
			http.Error(w, "invalid job preparation binding", http.StatusBadRequest)
			return
		}
		out, err := h.jobs.Prepare(ctx, actor, q)
		nodeResponse(w, out, err)
	case parts[2] == "transport" && r.Method == http.MethodPut:
		var q NodeTransportRequest
		if !h.decode(w, r, &q, 1<<20, &actor) {
			return
		}
		if q.ExpectedVersion == 0 {
			http.Error(w, "job version required", http.StatusBadRequest)
			return
		}
		if _, err := h.store.JobStatus(actor, id, h.clock()); err != nil {
			nodeResponse(w, nil, err)
			return
		}
		out, err := h.jobs.Transport(ctx, actor, id, q)
		nodeResponse(w, out, err)
	case parts[2] == "start" && r.Method == http.MethodPost:
		var q nodeagent.StartCommand
		if !h.decode(w, r, &q, 4096, &actor) {
			return
		}
		if q.Job != id || !nodeRouteID(q.IdempotencyKey) || q.ExpectedVersion == 0 {
			http.Error(w, "invalid start binding", http.StatusBadRequest)
			return
		}
		if _, err := h.store.JobStatus(actor, id, h.clock()); err != nil {
			nodeResponse(w, nil, err)
			return
		}
		out, err := h.jobs.Start(ctx, actor, q)
		nodeResponse(w, out, err)
	case parts[2] == "cancel" && r.Method == http.MethodPost:
		var q NodeCancelRequest
		if !h.decode(w, r, &q, 4096, &actor) {
			return
		}
		if q.ExpectedVersion == 0 {
			http.Error(w, "job version required", http.StatusBadRequest)
			return
		}
		out, err := h.store.CancelJob(ctx, actor, id, q.ExpectedVersion, h.clock())
		nodeResponse(w, out, err)
	default:
		http.NotFound(w, r)
	}
}

func (h *nodeHTTP) decode(w http.ResponseWriter, r *http.Request, dst any, limit int64, actor *trust.AuthenticatedPrincipal) bool {
	if err := readJSON(w, r, dst, limit); err != nil {
		http.Error(w, "invalid bounded canonical request; extra fields and credential envelopes are unsupported", http.StatusBadRequest)
		return false
	}
	// Recheck current trust after consuming the body, not only at TLS handshake.
	if r.TLS == nil {
		http.Error(w, "TLS required", http.StatusUnauthorized)
		return false
	}
	current, err := h.policy.Authenticate(*r.TLS)
	if err != nil || current.Role != trust.Controller || current.Principal != actor.Principal || r.Context().Err() != nil {
		http.Error(w, "current controller required", http.StatusForbidden)
		return false
	}
	*actor = current
	return true
}

func nodeResponse(w http.ResponseWriter, value any, err error) {
	if err != nil {
		// Local paths and key-store details must not enter remote error bodies.
		http.Error(w, "node operation rejected; inspect local agent diagnostics", http.StatusConflict)
		return
	}
	writeJSON(w, value)
}
