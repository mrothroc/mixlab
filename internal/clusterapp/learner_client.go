package clusterapp

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"slices"
	"time"

	"github.com/mrothroc/mixlab/coordinator/artifactapi"
	"github.com/mrothroc/mixlab/coordinator/learnerapi"
	"github.com/mrothroc/mixlab/diloco/authenticated"
	private "github.com/mrothroc/mixlab/diloco/learner"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/nodecredentials"
	"github.com/mrothroc/mixlab/transport/managedtls"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/principal"
)

// LearnerClient belongs to the agent. Only semantic values and byte streams
// cross its local IPC adapter; neither this object nor credentials enter MLX.
type LearnerClient struct{ options LearnerClientOptions }
type LearnerClientOptions struct {
	Plan       run.Plan
	Commit     run.Commit
	Workload   trust.WorkloadBinding
	Endpoint   string
	Anchor     trust.Anchor
	Trust      artifactapi.CurrentTrust
	Clock      func() time.Time
	Policy     func() (*managedtls.Policy, error)
	NodePolicy func() (*managedtls.Policy, error)
}
type learnerTrust func(context.Context, time.Time) (trust.VerifiedSnapshot, error)

func (f learnerTrust) Current(ctx context.Context, now time.Time) (trust.VerifiedSnapshot, error) {
	return f(ctx, now)
}

func NewLearnerClient(p *principal.Store, credential *nodecredentials.Transport, plan run.Plan, commit run.Commit, binding trust.WorkloadBinding, clock func() time.Time) (*LearnerClient, error) {
	if p == nil || credential == nil || clock == nil {
		return nil, fmt.Errorf("agent credentials required")
	}
	s, _, err := p.Active(clock())
	if err != nil {
		return nil, err
	}
	if s.Role != trust.Node || s.Principal != binding.Participant {
		return nil, fmt.Errorf("owning node required")
	}
	anchor, err := trust.PinRoot(s.Root, s.Fingerprint, clock())
	if err != nil {
		return nil, err
	}
	current := learnerTrust(func(_ context.Context, now time.Time) (trust.VerifiedSnapshot, error) {
		v, err := p.View(now)
		if err != nil {
			return trust.VerifiedSnapshot{}, err
		}
		return trust.VerifySnapshot(anchor, v.Snapshot, now)
	})
	policy := func() (*managedtls.Policy, error) {
		v, err := current.Current(context.Background(), clock())
		if err != nil {
			return nil, err
		}
		chain, key, err := credential.Identity(v, clock())
		if err != nil {
			return nil, err
		}
		if err = trust.VerifyWorkload(anchor, v, chain, binding, clock()); err != nil {
			return nil, err
		}
		return managedtls.New(managedtls.Identity{Chain: chain, Key: key}, func(chain [][]byte, now time.Time) (trust.AuthenticatedPrincipal, error) {
			v, err := current.Current(context.Background(), now)
			if err != nil {
				return trust.AuthenticatedPrincipal{}, err
			}
			return trust.AuthenticatePrincipal(anchor, v, chain, now)
		}, clock)
	}
	return NewLearnerClientWithPolicy(LearnerClientOptions{Plan: plan, Commit: commit, Workload: binding, Endpoint: plan.Endpoint, Anchor: anchor, Trust: current, Clock: clock, Policy: policy, NodePolicy: func() (*managedtls.Policy, error) { return managedPrincipalPolicy(p, trust.Coordinator, clock) }})
}

// WithPolicy is the composition/test port, equivalent to NodeClient's policy
// port. Policies must use current protected trust; exact server pin is enforced
// here before any body bytes, even when endpoint discovery is compromised.
func NewLearnerClientWithPolicy(o LearnerClientOptions) (*LearnerClient, error) {
	if err := o.Commit.ValidateFor(o.Plan); err != nil {
		return nil, err
	}
	b := o.Workload
	if o.Policy == nil || o.Trust == nil || o.Clock == nil || o.Anchor.Cluster() != o.Plan.Coordinator.ClusterID || b.Version != trust.LearnerBindingVersion || b.Role != trust.Worker || b.Audience != "diloco/learner" || b.Cluster != o.Anchor.Cluster() || b.Run != o.Commit.Binding.RunID || b.PlanHash != o.Plan.SHA256 || b.Coordinator != o.Plan.Coordinator.PrincipalID || b.Generation == 0 {
		return nil, fmt.Errorf("committed learner scope and current trust required")
	}
	found := false
	for _, r := range o.Commit.Registrations {
		if r.Slot.Member.WorkerID == b.Worker && r.Slot.Member.ParticipantID == b.Participant && r.Slot.Allocation.JobID == b.Job && r.Slot.Allocation.LeaseID == b.Lease && r.JobManifestHash == b.ManifestHash {
			found = true
		}
	}
	if !found {
		return nil, fmt.Errorf("workload not in committed cohort")
	}
	u, err := url.Parse(o.Endpoint)
	if err != nil || u.Scheme != "https" || u.Host == "" || u.User != nil || u.Path != "" || u.RawQuery != "" || u.Fragment != "" || u.Opaque != "" {
		return nil, fmt.Errorf("HTTPS coordinator origin required")
	}
	// Freeze slice-backed admission state against caller mutation.
	o.Plan.Slots = append([]run.Slot(nil), o.Plan.Slots...)
	o.Plan.Cohort.OrderedLearners = slices.Clone(o.Plan.Cohort.OrderedLearners)
	o.Commit.Registrations = append([]run.Registration(nil), o.Commit.Registrations...)
	return &LearnerClient{options: o}, nil
}
func (c *LearnerClient) request(operation string) private.RelayRequest {
	return private.RelayRequest{Format: "mixlab_learner_relay_v1", Binding: c.options.Commit.Binding, WorkerID: c.options.Workload.Worker, Operation: operation}
}
func (c *LearnerClient) httpClient(node bool) (*http.Client, *managedtls.Policy, error) {
	f := c.options.Policy
	if node {
		f = c.options.NodePolicy
	}
	if f == nil {
		return nil, nil, fmt.Errorf("node artifact policy required")
	}
	p, err := f()
	if err != nil {
		return nil, nil, err
	}
	p, err = p.RequirePeer(trust.Coordinator, c.options.Plan.Coordinator.PrincipalID)
	if err != nil {
		return nil, nil, err
	}
	h, err := p.HTTPClient(time.Minute)
	return h, p, err
}
func (c *LearnerClient) call(ctx context.Context, q private.RelayRequest) (learnerapi.Response, error) {
	var out learnerapi.Response
	if err := q.Validate(); err != nil {
		return out, err
	}
	if q.Binding != c.options.Commit.Binding || q.WorkerID != c.options.Workload.Worker {
		return out, fmt.Errorf("local IPC scope mismatch")
	}
	b, err := json.Marshal(learnerapi.Request{Workload: c.options.Workload, Call: q})
	if err != nil {
		return out, err
	}
	if len(b) > learnerapi.MaxMessage {
		return out, fmt.Errorf("learner request too large")
	}
	return retryLearnerSemantic(ctx, func(attempt context.Context) (learnerapi.Response, bool, error) {
		return c.callAttempt(attempt, q, b)
	})
}

// Only the already-encoded semantic request is replayable. Stream transfers
// have caller-owned cursors and deliberately do not use this retry path.
func (c *LearnerClient) callAttempt(ctx context.Context, q private.RelayRequest, body []byte) (learnerapi.Response, bool, error) {
	var out learnerapi.Response
	h, p, err := c.httpClient(false)
	if err != nil {
		return out, false, err
	}
	defer h.CloseIdleConnections()
	r, err := http.NewRequestWithContext(ctx, http.MethodPost, c.options.Endpoint+learnerapi.Route, bytes.NewReader(body))
	if err != nil {
		return out, false, err
	}
	r.Header.Set("Content-Type", "application/json")
	resp, err := h.Do(r)
	if err != nil {
		return out, transientLearnerTransport(err), err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK || resp.TLS == nil {
		return out, false, fmt.Errorf("learner operation rejected (HTTP %d)", resp.StatusCode)
	}
	if _, err = p.Authenticate(*resp.TLS); err != nil {
		return out, false, err
	}
	b, err := io.ReadAll(io.LimitReader(resp.Body, learnerapi.MaxMessage+1))
	if err != nil {
		return out, transientLearnerTransport(err), err
	}
	if _, err = p.Authenticate(*resp.TLS); err != nil {
		return out, false, err
	}
	if err = authenticated.Decode(b, &out); err != nil {
		return out, false, err
	}
	if err = c.validateReply(q, out); err != nil {
		return learnerapi.Response{}, false, err
	}
	return out, false, ctx.Err()
}
func (c *LearnerClient) validateReply(q private.RelayRequest, r learnerapi.Response) error {
	o := r.Reply
	if (q.Operation == "artifact") != (r.Grant != nil) || ((q.Operation == "submit" || q.Operation == "receipt") != (o.Receipt != nil)) || (q.Operation != "poll" && (o.Assignment != nil || o.Evaluation != nil || o.Done)) {
		return fmt.Errorf("unexpected learner reply")
	}
	if a := o.Assignment; a != nil {
		if err := a.ValidateFor(c.options.Plan, c.options.Commit); err != nil {
			return err
		}
		if a.WorkerID != q.WorkerID || c.options.Clock().Unix() < a.IssuedAt || c.options.Clock().Unix() >= a.ExpiresAt {
			return fmt.Errorf("foreign/expired assignment")
		}
	}
	if a := o.Evaluation; a != nil {
		if a.Binding != q.Binding || a.WorkerID != q.WorkerID || a.Coordinator != c.options.Plan.Coordinator || c.options.Clock().Unix() < a.IssuedAt || c.options.Clock().Unix() >= a.ExpiresAt {
			return fmt.Errorf("foreign/expired evaluation")
		}
		for _, reg := range c.options.Commit.Registrations {
			if reg.Slot.Member.WorkerID == q.WorkerID && a.PrecisionHash != reg.PrecisionHash {
				return fmt.Errorf("evaluation precision changed")
			}
		}
	}
	if receipt := o.Receipt; receipt != nil {
		h, err := run.Digest(*q.Update)
		if err != nil {
			return err
		}
		if receipt.Binding != q.Binding || receipt.UpdateID != q.Update.ID || receipt.UpdateHash != h || receipt.PrivateHash != q.Update.PrivateHash {
			return fmt.Errorf("receipt mismatch")
		}
	}
	if r.Grant != nil {
		g := r.Grant.Body
		a := q.Artifact.Assignment
		if e := q.Artifact.Evaluation; e != nil {
			a.ID = e.ID
			a.ExpiresAt = e.ExpiresAt
		}
		if g.Binding != q.Binding || g.WorkerID != q.WorkerID || g.PrincipalID != c.options.Workload.Participant || g.ParticipantID != c.options.Workload.Participant || g.Coordinator != c.options.Plan.Coordinator || g.AssignmentID != a.ID || g.Artifact != q.Artifact.Blob || g.Operation != q.Artifact.Operation || g.ExpiresAt > a.ExpiresAt || c.options.Clock().Unix() < g.IssuedAt || c.options.Clock().Unix() >= g.ExpiresAt {
			return fmt.Errorf("artifact grant mismatch")
		}
		v, err := c.options.Trust.Current(context.Background(), c.options.Clock())
		if err != nil {
			return err
		}
		_, err = trust.VerifyProof(c.options.Anchor, v, r.Grant.Proof, r.Grant.Proof.Request, c.options.Clock())
		return err
	}
	return nil
}

// Call is the bounded semantic local-IPC adapter port. Artifact bytes must use
// Read/Write, not JSON messages; the signed grant is never returned to the child.
func (c *LearnerClient) Call(ctx context.Context, q private.RelayRequest) (private.RelayResponse, error) {
	r, err := c.call(ctx, q)
	return r.Reply, err
}
func (c *LearnerClient) Poll(ctx context.Context) (private.RelayResponse, error) {
	return c.Call(ctx, c.request("poll"))
}
func (c *LearnerClient) Start(ctx context.Context, s private.Start) error {
	q := c.request("start")
	q.Start = &s
	_, err := c.Call(ctx, q)
	return err
}
func (c *LearnerClient) Submit(ctx context.Context, u private.Update) (private.Receipt, error) {
	return c.receipt(ctx, "submit", u)
}
func (c *LearnerClient) Receipt(ctx context.Context, u private.Update) (private.Receipt, error) {
	return c.receipt(ctx, "receipt", u)
}
func (c *LearnerClient) receipt(ctx context.Context, op string, u private.Update) (private.Receipt, error) {
	q := c.request(op)
	q.Update = &u
	r, err := c.Call(ctx, q)
	if err != nil {
		return private.Receipt{}, err
	}
	return *r.Receipt, nil
}
func (c *LearnerClient) Evaluate(ctx context.Context, result run.EvaluationResult) error {
	q := c.request("evaluate")
	q.Evaluation = &result
	_, err := c.Call(ctx, q)
	return err
}
