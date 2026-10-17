package clusterapp

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/mrothroc/mixlab/diloco/authenticated"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/nodecredentials"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/principal"
	"github.com/mrothroc/mixlab/trust/workload"
)

type NodeLearnerRetryRequest struct {
	Attempt     string `json:"attempt"`
	PrivateHash string `json:"private_hash"`
}
type NodeLearnerRetryPrepared struct {
	Evidence nodejob.SignedLearnerRecovery `json:"evidence"`
	Request  workload.SignedRequest        `json:"request"`
}
type NodeLearnerRetryStart struct {
	Acknowledgement authenticated.Recovery `json:"acknowledgement"`
	Grant           workload.Grant         `json:"grant"`
	Credential      workload.Result        `json:"credential"`
}

func learnerRecoveryRoute(parts []string, method string) bool {
	return len(parts) == 3 && parts[0] == "learners" && nodeRouteID(parts[1]) &&
		(parts[2] == "status" && method == http.MethodGet || (parts[2] == "retry-authorize" || parts[2] == "retry-start") && method == http.MethodPut)
}

func (h *nodeHTTP) learnerRecovery(w http.ResponseWriter, r *http.Request, actor trust.AuthenticatedPrincipal, parts []string) {
	if actor.Role != trust.Coordinator {
		http.Error(w, "coordinator required", http.StatusForbidden)
		return
	}
	id := parts[1]
	if parts[2] == "status" {
		if r.ContentLength != 0 || len(r.TransferEncoding) != 0 {
			http.Error(w, "body not allowed", http.StatusBadRequest)
			return
		}
		out, err := h.store.LearnerStatus(r.Context(), actor, id, h.clock())
		h.recoveryResponse(w, id, parts[2], out, err)
		return
	}
	owner, ok := h.jobs.(interface {
		AuthorizeLearnerRetry(context.Context, trust.AuthenticatedPrincipal, string, NodeLearnerRetryRequest) (NodeLearnerRetryPrepared, error)
		StartLearnerRetry(context.Context, trust.AuthenticatedPrincipal, string, NodeLearnerRetryStart) error
	})
	if !ok {
		http.NotFound(w, r)
		return
	}
	if parts[2] == "retry-authorize" {
		var q NodeLearnerRetryRequest
		if !h.decode(w, r, &q, 4096, &actor) {
			return
		}
		out, err := owner.AuthorizeLearnerRetry(r.Context(), actor, id, q)
		h.recoveryResponse(w, id, parts[2], out, err)
	} else {
		var q NodeLearnerRetryStart
		if !h.decode(w, r, &q, 2<<20, &actor) {
			return
		}
		h.recoveryResponse(w, id, parts[2], NodeLearnerAck{id}, owner.StartLearnerRetry(r.Context(), actor, id, q))
	}
}

func (h *nodeHTTP) recoveryResponse(w http.ResponseWriter, job, operation string, out any, err error) {
	if err != nil && h.event != nil {
		diagnostic := err.Error()
		if len(diagnostic) > 4096 {
			diagnostic = diagnostic[:4096]
		}
		// Keep private diagnostics on the node. The coordinator sees stage and
		// HTTP status, never credential payloads or local paths from the body.
		h.event(fmt.Errorf("learner recovery job=%s operation=%s error=%q", job, operation, diagnostic))
	}
	nodeResponse(w, out, err)
}

func (a *NodeJobs) AuthorizeLearnerRetry(ctx context.Context, actor trust.AuthenticatedPrincipal, id string, q NodeLearnerRetryRequest) (out NodeLearnerRetryPrepared, err error) {
	if err = a.lockRecovery(ctx); err != nil {
		return out, err
	}
	defer a.mu.Unlock()
	if a.options.SignRecovery == nil {
		return out, fmt.Errorf("node recovery signer unavailable")
	}
	if _, err = a.current(ctx); err != nil {
		return out, err
	}
	body, err := a.options.Store.AuthorizeLearnerRetry(ctx, actor, id, q.Attempt, q.PrivateHash, a.options.Clock())
	if err != nil {
		return out, err
	}
	request, err := body.SigningRequest()
	if err != nil {
		return out, err
	}
	out.Evidence.Body = body
	out.Evidence.Proof, err = a.options.SignRecovery(ctx, request)
	if err != nil {
		return out, err
	}
	out.Request, err = a.options.Store.PrepareLearnerRetryWorkload(ctx, actor, id, a.options.Workload)
	return
}

func (a *NodeJobs) StartLearnerRetry(ctx context.Context, actor trust.AuthenticatedPrincipal, id string, q NodeLearnerRetryStart) error {
	if err := a.lockRecovery(ctx); err != nil {
		return err
	}
	defer a.mu.Unlock()
	if err := a.options.Store.CheckLearnerRetryStart(ctx, actor, id, q.Acknowledgement, q.Grant, q.Credential, a.options.Clock()); err != nil {
		return err
	}
	v, err := a.currentForEvidence(ctx, nodeEvidenceSnapshot(q.Acknowledgement.Proof.Evidence.Snapshot, q.Credential.Snapshot), func(v trust.VerifiedSnapshot) error {
		if err := q.Acknowledgement.Validate(); err != nil {
			return err
		}
		if _, err := trust.VerifyProof(a.options.Anchor, v, q.Acknowledgement.Proof, q.Acknowledgement.Proof.Request, a.options.Clock()); err != nil {
			return err
		}
		return a.verifyCredential(v, q.Grant, q.Credential)
	})
	if err != nil {
		return err
	}
	// The store verifies caller, acknowledgement, exact pending scope and live
	// lease before invoking this local effect. No launch approval exists yet.
	_, err = a.options.Store.StartLearnerRetry(ctx, actor, id, a.options.Anchor, v, q.Acknowledgement, q.Grant, q.Credential, func(ctx context.Context, old workload.Scope) error {
		scope := q.Grant.Request.Request.Scope
		dir, err := workloadDirectory(a.options.CredentialRoot, scope)
		if err != nil {
			return err
		}
		c, err := nodecredentials.OpenTransport(dir, a.options.Anchor, scope)
		if err != nil {
			return err
		}
		defer func() { _ = c.Close() }()
		if err = c.Install(ctx, q.Grant, q.Credential, v, a.options.Clock()); err != nil {
			return err
		}
		return DestroyNodeWorkload(ctx, a.options.CredentialRoot, a.options.Anchor, nodeagent.CleanupRequest{Job: id, Attempt: old.Attempt, Workload: old, WorkloadInitialized: true})
	}, a.options.Clock())
	return err
}

func NodeRecoverySigner(p *principal.Store, clock func() time.Time) func(context.Context, trust.SignRequest) (trust.SignedProof, error) {
	return func(ctx context.Context, q trust.SignRequest) (trust.SignedProof, error) {
		if err := ctx.Err(); err != nil {
			return trust.SignedProof{}, err
		}
		s, key, err := p.Active(clock())
		if err != nil {
			return trust.SignedProof{}, err
		}
		if s.Role != trust.Node || q.Purpose != trust.LearnerRecovery {
			return trust.SignedProof{}, fmt.Errorf("node recovery proof required")
		}
		a, err := trust.PinRoot(s.Root, s.Fingerprint, clock())
		if err != nil {
			return trust.SignedProof{}, err
		}
		v, err := trust.VerifySnapshot(a, s.Snapshot, clock())
		if err != nil {
			return trust.SignedProof{}, err
		}
		return trust.SignPrincipalProof(a, s.Chain, key, v, q, clock())
	}
}

func (c *NodeClient) LearnerStatus(ctx context.Context, id string) (out nodeagent.LearnerStatus, err error) {
	if !nodeRouteID(id) {
		return out, fmt.Errorf("canonical learner job required")
	}
	err = c.request(ctx, http.MethodGet, "/v1/agent/learners/"+id+"/status", nil, &out)
	if err == nil && (out.Allocation.JobID != id || out.Allocation.NodeID != c.node || !nodeRouteID(out.Attempt)) {
		err = fmt.Errorf("learner status identity mismatch")
	}
	return
}
func (c *NodeClient) AuthorizeLearnerRetry(ctx context.Context, id string, q NodeLearnerRetryRequest) (out NodeLearnerRetryPrepared, err error) {
	if !nodeRouteID(id) {
		return out, fmt.Errorf("canonical learner job required")
	}
	err = c.request(ctx, http.MethodPut, "/v1/agent/learners/"+id+"/retry-authorize", q, &out)
	if err == nil {
		b := out.Evidence.Body.Attempt
		if b.Allocation.NodeID != c.node || b.Allocation.JobID != id || b.AttemptID != q.Attempt || b.PrivateHash != q.PrivateHash {
			return out, fmt.Errorf("retry response identity mismatch")
		}
		err = out.Evidence.Body.Validate()
	}
	return
}
func (c *NodeClient) StartLearnerRetry(ctx context.Context, id string, q NodeLearnerRetryStart) error {
	return c.learnerCommand(ctx, http.MethodPut, id, "retry-start", q)
}
