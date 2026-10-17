// Package learnerapi authenticates agent-owned learner RPCs. It never launches
// children or interprets private tensors; artifact bytes use artifactapi.
package learnerapi

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/mrothroc/mixlab/coordinator"
	"github.com/mrothroc/mixlab/coordinator/artifactapi"
	"github.com/mrothroc/mixlab/diloco/authenticated"
	private "github.com/mrothroc/mixlab/diloco/learner"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/transport/managedtls"
	"github.com/mrothroc/mixlab/trust"
)

const Route = "/v1/diloco/learner"
const MaxMessage = 1 << 20

type Request struct {
	Workload trust.WorkloadBinding `json:"workload"`
	Call     private.RelayRequest  `json:"call"`
}

func (q Request) Validate() error { return q.Call.Validate() }

type Response struct {
	Reply private.RelayResponse `json:"reply"`
	Grant *authenticated.Grant  `json:"grant,omitempty"`
}

func (r Response) Validate() error {
	if err := r.Reply.Validate(); err != nil {
		return err
	}
	if r.Grant != nil {
		return r.Grant.Validate()
	}
	return nil
}

type Gateway struct {
	Policy       *managedtls.Policy
	Anchor       trust.Anchor
	Trust        artifactapi.CurrentTrust
	RefreshTrust func(context.Context) error
	Coordinator  run.Coordinator
	Model        coordinator.Model
	// ModelForRun selects service-owned allocation ports for an accepted run.
	// Repository and trust remain shared; request bodies cannot supply ports.
	ModelForRun func(context.Context, string) (coordinator.Model, error)
	Clock       func() time.Time
	// Sign uses the coordinator's protected principal signer and current trust.
	Sign    func(context.Context, trust.SignRequest) (trust.SignedProof, error)
	Grants  *Grants
	Timeout time.Duration
}

// NewPolicy admits worker and node TLS profiles so the same listener can mount
// learner RPCs and P3 artifactapi. Each handler independently restricts its role.
func NewPolicy(local managedtls.Identity, anchor trust.Anchor, current artifactapi.CurrentTrust, clock func() time.Time) (*managedtls.Policy, error) {
	if current == nil || clock == nil {
		return nil, fmt.Errorf("current pinned trust required")
	}
	return managedtls.New(local, func(chain [][]byte, now time.Time) (trust.AuthenticatedPrincipal, error) {
		v, err := current.Current(context.Background(), now)
		if err != nil {
			return trust.AuthenticatedPrincipal{}, err
		}
		p, err := trust.AuthenticatePrincipal(anchor, v, chain, now)
		if err == nil && p.Role != trust.Worker && p.Role != trust.Node {
			err = fmt.Errorf("learner or node principal required")
		}
		return p, err
	}, clock)
}
func (g Gateway) Handler() (http.Handler, error) {
	if g.Policy == nil || g.Trust == nil || g.Clock == nil || g.Sign == nil || g.Grants == nil || g.Model.Repository == nil || g.Model.Allocations == nil || g.Model.Artifacts == nil || g.Model.Clock == nil || g.Model.Evaluator == nil || g.Timeout <= 0 || g.Timeout > time.Minute {
		return nil, fmt.Errorf("complete bounded learner gateway required")
	}
	if _, ok := g.Model.Repository.(coordinator.EvaluationReceipts); !ok {
		return nil, fmt.Errorf("durable evaluation receipt lookup required")
	}
	if err := g.Coordinator.Validate(); err != nil {
		return nil, err
	}
	if g.Anchor.Cluster() != g.Coordinator.ClusterID {
		return nil, fmt.Errorf("coordinator anchor mismatch")
	}
	return g.Policy.AuthenticateHTTP(http.HandlerFunc(g.serve)), nil
}
func (g Gateway) serve(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost || r.URL.Path != Route || r.URL.RawQuery != "" {
		http.NotFound(w, r)
		return
	}
	control := http.NewResponseController(w)
	deadline := time.Now().Add(g.Timeout)
	if control.SetReadDeadline(deadline) != nil || control.SetWriteDeadline(deadline) != nil {
		http.Error(w, "bounded transport required", http.StatusServiceUnavailable)
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), g.Timeout)
	defer cancel()
	if ctx.Err() != nil {
		learnerError(w, ctx, "learner service unavailable", http.StatusServiceUnavailable)
		return
	}
	b, err := io.ReadAll(io.LimitReader(r.Body, MaxMessage+1))
	var q Request
	if ctx.Err() != nil || err != nil || authenticated.Decode(b, &q) != nil {
		learnerError(w, ctx, "invalid learner request", http.StatusBadRequest)
		return
	}
	if g.ModelForRun != nil {
		model, err := g.ModelForRun(ctx, q.Call.Binding.RunID)
		if ctx.Err() != nil || err != nil || model.Repository == nil || model.Allocations == nil || model.Artifacts == nil || model.Clock == nil || model.Evaluator == nil {
			learnerError(w, ctx, "run is not available", http.StatusForbidden)
			return
		}
		g.Model = model
	}
	guard := guardedRepository{Repository: g.Model.Repository, check: func(ctx context.Context, s coordinator.State) error { return g.authorize(ctx, r, q, s) }}
	s, err := guard.Load(ctx, q.Call.Binding.RunID)
	if ctx.Err() != nil || err != nil {
		learnerError(w, ctx, "learner access rejected", http.StatusForbidden)
		return
	}
	m := g.Model
	m.Repository = guard
	out, err := g.dispatch(ctx, m, s, q)
	if errors.Is(err, coordinator.ErrConcurrent) || errors.Is(err, trust.ErrTrustBehind) {
		learnerError(w, ctx, "learner service unavailable", http.StatusServiceUnavailable)
		return
	}
	if ctx.Err() != nil || err != nil {
		learnerError(w, ctx, "learner operation rejected", http.StatusConflict)
		return
	}
	// Never release a result after trust or the attempt changed during the call.
	if _, err = guard.Load(ctx, q.Call.Binding.RunID); ctx.Err() != nil || err != nil {
		learnerError(w, ctx, "learner access changed", http.StatusForbidden)
		return
	}
	b, err = json.Marshal(out)
	if ctx.Err() != nil || err != nil || len(b) > MaxMessage || out.Validate() != nil {
		learnerError(w, ctx, "invalid learner result", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write(b)
}

func learnerError(w http.ResponseWriter, ctx context.Context, message string, status int) {
	// Shutdown and request deadlines invalidate the operation, not its credentials.
	// A dependency's context-shaped error alone is not evidence of unavailability.
	if ctx.Err() != nil {
		message, status = "learner service unavailable", http.StatusServiceUnavailable
	}
	http.Error(w, message, status)
}

func (g Gateway) authorize(ctx context.Context, r *http.Request, q Request, s coordinator.State) error {
	if err := s.Validate(); err != nil {
		return err
	}
	now := g.Clock()
	v, err := g.Trust.Current(ctx, now)
	if err != nil {
		return err
	}
	chain := make([][]byte, len(r.TLS.PeerCertificates))
	for i, c := range r.TLS.PeerCertificates {
		chain[i] = c.Raw
	}
	b := q.Workload
	// All admission-owned fields are constrained below before exact certificate
	// verification. Principal and validity are authenticated certificate fields.
	if b.Version != trust.LearnerBindingVersion || b.Role != trust.Worker || b.Audience != "diloco/learner" || b.Cluster != g.Anchor.Cluster() || b.Coordinator != g.Coordinator.PrincipalID || b.Run != q.Call.Binding.RunID || b.PlanHash != q.Call.Binding.PlanHash || b.Worker != q.Call.WorkerID || b.Group != "" || b.Member != "" || b.MembershipHash != "" || b.Rank != 0 {
		return fmt.Errorf("learner binding mismatch")
	}
	if err = trust.VerifyWorkload(g.Anchor, v, chain, b, now); err != nil {
		return err
	}
	if s.Commit == nil || s.Commit.Binding != q.Call.Binding || s.Plan.Coordinator != g.Coordinator {
		return fmt.Errorf("uncommitted/foreign run")
	}
	attempt, generation, err := s.CurrentAttempt(b.Worker)
	if err != nil || attempt != b.Attempt || generation != b.Generation {
		return fmt.Errorf("stale attempt")
	}
	if s.Recovery != nil && s.Recovery.WorkerID == b.Worker && s.Recovery.Previous == b.Attempt {
		return fmt.Errorf("attempt fenced by pending recovery")
	}
	for _, reg := range s.Commit.Registrations {
		slot := reg.Slot
		if slot.Member.WorkerID == b.Worker && slot.Member.ParticipantID == b.Participant && slot.Allocation.NodeID == b.Participant && slot.Allocation.JobID == b.Job && slot.Allocation.LeaseID == b.Lease && reg.JobManifestHash == b.ManifestHash {
			// A completed run releases its allocations. Exact final replies and
			// receipt retries remain readable under current trust and attempt
			// binding; they cannot authorize more work or artifact grants.
			if s.Phase == coordinator.Complete || s.Phase == coordinator.Failed {
				switch q.Call.Operation {
				case "poll", "receipt", "evaluate":
					return nil
				default:
					return fmt.Errorf("terminal run permits only status and receipt reads")
				}
			}
			return g.Model.Allocations.Ready(ctx, slot)
		}
	}
	return fmt.Errorf("workload allocation/manifest mismatch")
}

type guardedRepository struct {
	coordinator.Repository
	check func(context.Context, coordinator.State) error
}

func (r guardedRepository) Load(ctx context.Context, id string) (coordinator.State, error) {
	s, err := r.Repository.Load(ctx, id)
	if err == nil {
		err = r.check(ctx, s)
	}
	return s, err
}
func (r guardedRepository) Save(ctx context.Context, id string, revision uint64, c coordinator.Change) error {
	if err := r.check(ctx, c.Next); err != nil {
		return err
	}
	return r.Repository.Save(ctx, id, revision, c)
}
func (g Gateway) dispatch(ctx context.Context, m coordinator.Model, s coordinator.State, q Request) (Response, error) {
	out := Response{Reply: private.RelayResponse{Format: "mixlab_learner_reply_v1"}}
	c, b := q.Call, q.Workload
	var allocation run.Allocation
	for _, slot := range s.Plan.Slots {
		if slot.Member.WorkerID == b.Worker {
			allocation = slot.Allocation
		}
	}
	switch c.Operation {
	case "poll":
		if s.Phase == coordinator.Failed {
			return out, fmt.Errorf("run failed")
		}
		out.Reply.Done = s.Phase == coordinator.Complete
		if s.Phase == coordinator.Open {
			for _, a := range s.Assignments {
				if a.WorkerID == b.Worker {
					if g.Clock().Unix() < a.IssuedAt || g.Clock().Unix() >= a.ExpiresAt {
						return out, fmt.Errorf("assignment expired")
					}
					out.Reply.Assignment = &a
				}
			}
		}
		if s.Phase == coordinator.Evaluating && s.Evaluation != nil && s.Evaluation.WorkerID == b.Worker {
			a := s.Evaluation
			if g.Clock().Unix() < a.IssuedAt || g.Clock().Unix() >= a.ExpiresAt {
				return out, fmt.Errorf("evaluation expired")
			}
			out.Reply.Evaluation = a
		}
	case "start":
		live := false
		for _, a := range s.Assignments {
			h, err := run.Digest(a)
			if err == nil && a.WorkerID == b.Worker && h == c.Start.AssignmentHash && g.Clock().Unix() >= a.IssuedAt && g.Clock().Unix() < a.ExpiresAt {
				live = true
			}
		}
		if !live {
			return out, fmt.Errorf("start assignment not current")
		}
		return out, m.RecordStartSubmission(ctx, private.StartSubmission{Format: "mixlab_start_submission_v1", Start: *c.Start, PrincipalID: b.Principal, AttemptID: b.Attempt, WorkloadGeneration: b.Generation, Allocation: allocation})
	case "submit", "receipt":
		u := *c.Update
		if u.ParticipantID != b.Participant {
			return out, fmt.Errorf("update participant mismatch")
		}
		for _, slot := range s.Plan.Slots {
			if slot.Member.WorkerID == b.Worker && u.IslandID != slot.Member.IslandID {
				return out, fmt.Errorf("update island mismatch")
			}
		}
		var receipt private.Receipt
		var err error
		if c.Operation == "submit" {
			receipt, err = m.AcceptSubmission(ctx, private.Submission{Format: "mixlab_update_submission_v1", Update: u, PrincipalID: b.Principal, AttemptID: b.Attempt, WorkloadGeneration: b.Generation, Allocation: allocation})
		} else {
			receipt, err = m.Repository.Receipt(ctx, b.Run, u.ID)
		}
		if err != nil {
			return out, err
		}
		h, err := run.Digest(u)
		if err != nil || receipt.UpdateHash != h || receipt.UpdateID != u.ID || receipt.Binding != u.Binding || receipt.PrivateHash != u.PrivateHash {
			return out, fmt.Errorf("receipt differs from update")
		}
		out.Reply.Receipt = &receipt
	case "evaluate":
		// The decision may already have advanced the run before a reply was
		// lost. Only the identical committed result is an idempotent retry.
		if receipts, ok := g.Model.Repository.(coordinator.EvaluationReceipts); ok {
			prior, err := receipts.EvaluationResult(ctx, b.Run, c.Evaluation.Assignment.ID)
			if err == nil {
				if prior != *c.Evaluation {
					return out, fmt.Errorf("evaluation conflicts with committed result")
				}
				return out, nil
			}
			if !errors.Is(err, coordinator.ErrNotFound) {
				return out, err
			}
		}
		if s.Phase != coordinator.Evaluating || s.Evaluation == nil || *s.Evaluation != c.Evaluation.Assignment {
			return out, fmt.Errorf("evaluation not current")
		}
		m.Evaluator = fixedEvaluation{*c.Evaluation}
		return out, m.Advance(ctx, b.Run)
	case "artifact":
		grant, err := g.grant(ctx, s, q)
		if err != nil {
			return out, err
		}
		out.Grant = &grant
	}
	return out, nil
}

type fixedEvaluation struct{ result run.EvaluationResult }

func (f fixedEvaluation) Evaluate(_ context.Context, a run.EvaluationAssignment) (run.EvaluationResult, error) {
	if a != f.result.Assignment {
		return run.EvaluationResult{}, fmt.Errorf("evaluation changed")
	}
	return f.result, nil
}
func (g Gateway) grant(ctx context.Context, s coordinator.State, q Request) (authenticated.Grant, error) {
	var out authenticated.Grant
	r := q.Call.Artifact
	a := r.Assignment
	found := false
	if e := r.Evaluation; e != nil {
		found = s.Phase == coordinator.Evaluating && s.Evaluation != nil && *s.Evaluation == *e && r.Operation == "read" && r.Blob == e.Model
		a = run.Assignment{ID: e.ID, Binding: e.Binding, WorkerID: e.WorkerID, BaseModel: e.Model, IssuedAt: e.IssuedAt, ExpiresAt: e.ExpiresAt}
	} else if s.Phase == coordinator.Open {
		for _, current := range s.Assignments {
			if current == a {
				found = true
			}
		}
	}
	if !found || g.Clock().Unix() < a.IssuedAt || g.Clock().Unix() >= a.ExpiresAt {
		return out, fmt.Errorf("artifact assignment not current")
	}
	if (r.Operation == "read" && r.Blob != a.BaseModel && r.Blob != a.OuterState) || (r.Operation == "create" && r.Blob.Kind != "delta_f32") {
		return out, fmt.Errorf("unassigned artifact")
	}
	id, err := run.Digest(q)
	if err != nil {
		return out, err
	}
	v, err := g.Trust.Current(ctx, g.Clock())
	if err != nil {
		return out, err
	}
	// One grant per artifact kind/assignment/attempt/trust generation. A worker
	// cannot grow the grant journal by asking for arbitrary delta digests.
	keyRequest := q
	artifactScope := *r
	artifactScope.Blob = run.Blob{Kind: r.Blob.Kind}
	keyRequest.Call.Artifact = &artifactScope
	grantID, err := run.Digest(struct {
		Request    Request
		Generation uint64
	}{keyRequest, v.Generation()})
	if err != nil {
		return out, err
	}
	b := run.Grant{Format: "mixlab_artifact_access_grant_v1", ID: grantID, Binding: a.Binding, Coordinator: g.Coordinator, PrincipalID: q.Workload.Participant, ParticipantID: q.Workload.Participant, WorkerID: a.WorkerID, AssignmentID: a.ID, Operation: r.Operation, Artifact: r.Blob, AccessClass: "global_base", IssuedAt: a.IssuedAt, ExpiresAt: min(a.ExpiresAt, q.Workload.ExpiresAt), MaximumUses: 8, Nonce: id}
	if r.Operation == "create" {
		b.AccessClass = "outer_update"
		b.UploadID = id
	}
	b.SHA256, err = b.Digest()
	if err != nil {
		return out, err
	}
	if err = b.Validate(); err != nil {
		return out, err
	}
	out, err = g.Grants.issue(ctx, b, func() (authenticated.Grant, error) {
		proof, err := g.Sign(ctx, trust.SignRequest{Version: trust.ProofVersion, Purpose: trust.ArtifactGrant, Digest: b.SHA256, Context: b.Binding.RunID, Audience: b.Coordinator.Audience})
		return authenticated.Grant{Body: b, Proof: proof}, err
	})
	if err != nil {
		return out, err
	}
	_, err = trust.CurrentForEvidence(ctx, g.Anchor, out.Proof.Evidence.Snapshot,
		func(ctx context.Context) (trust.VerifiedSnapshot, error) { return g.Trust.Current(ctx, g.Clock()) },
		g.RefreshTrust, g.Clock, func(v trust.VerifiedSnapshot) error {
			_, err := trust.VerifyProof(g.Anchor, v, out.Proof, out.Proof.Request, g.Clock())
			return err
		})
	return out, err
}
