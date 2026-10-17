package clusterapp

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"net"
	"net/http"
	"time"

	"github.com/mrothroc/mixlab/coordinator"
	"github.com/mrothroc/mixlab/coordinator/recoveryauth"
	"github.com/mrothroc/mixlab/diloco/authenticated"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/transport/managedtls"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/workload"
)

type LearnerRecoveryNode interface {
	LearnerStatus(context.Context, string) (nodeagent.LearnerStatus, error)
	AuthorizeLearnerRetry(context.Context, string, NodeLearnerRetryRequest) (NodeLearnerRetryPrepared, error)
	StartLearnerRetry(context.Context, string, NodeLearnerRetryStart) error
}

// DiLoCoRecovery composes existing owners; the repository, node and authority
// each retain their own idempotency records. No mutable state lives here.
type DiLoCoRecovery struct {
	Model   coordinator.Model
	Node    func(run.Allocation) (LearnerRecoveryNode, error)
	Anchor  trust.Anchor
	Current func(context.Context) (trust.VerifiedSnapshot, error)
	// Refresh fetches from locally pinned authority endpoints, never from proof
	// evidence. It must persist trust monotonically before Current is re-read.
	Refresh func(context.Context) error
	Clock   func() time.Time
	Sign    func(context.Context, trust.SignRequest) (trust.SignedProof, error)
	Grant   func(context.Context, workload.SignedRequest) (workload.Grant, error)
	Issue   func(context.Context, workload.Grant) (workload.Result, error)
}

func (r DiLoCoRecovery) currentForRecoveryEvidence(ctx context.Context, proof trust.SignedProof, verify func(trust.VerifiedSnapshot) error) (trust.VerifiedSnapshot, error) {
	return trust.CurrentForEvidence(ctx, r.Anchor, proof.Evidence.Snapshot, r.Current, r.Refresh, r.Clock, verify)
}

func recoveryTransient(err error) bool {
	// Inspect each wrapper before descending: certificate/remote rejection
	// remains terminal even when its cause joins otherwise transient failures.
	switch e := err.(type) {
	case nil, tls.AlertError, *tls.CertificateVerificationError,
		x509.CertificateInvalidError, x509.UnknownAuthorityError, x509.HostnameError:
		return false
	case *net.OpError:
		if e.Op == "remote error" {
			return false
		}
	}
	if err == context.Canceled || err == managedtls.ErrPeerRejected {
		return false
	}
	// Lease maintenance can report several independent peers. One retryable
	// timeout must never hide another peer's permanent authorization rejection.
	if joined, ok := err.(interface{ Unwrap() []error }); ok {
		causes := joined.Unwrap()
		if len(causes) == 0 {
			return false
		}
		for _, cause := range causes {
			if !recoveryTransient(cause) {
				return false
			}
		}
		return true
	}
	if wrapped, ok := err.(interface{ Unwrap() error }); ok {
		return recoveryTransient(wrapped.Unwrap())
	}
	if errors.Is(err, coordinator.ErrConflict) || errors.Is(err, trust.ErrTrustBehind) {
		return true
	}
	var h *NodeHTTPError
	if errors.As(err, &h) {
		return recoveryHTTPTransient(h.Status)
	}
	var authority *enrollmentHTTPError
	if errors.As(err, &authority) {
		return recoveryHTTPTransient(authority.Status)
	}
	return transientLearnerTransport(err)
}

func recoveryHTTPTransient(status int) bool {
	return status == http.StatusServiceUnavailable || status == http.StatusBadGateway || status == http.StatusGatewayTimeout || status == http.StatusTooManyRequests
}

func (r DiLoCoRecovery) failure(ctx context.Context, id string, pending bool, err error) error {
	return r.failureWithReason(ctx, id, pending, err, "learner same-allocation recovery permanently rejected")
}

func (r DiLoCoRecovery) failureWithReason(ctx context.Context, id string, pending bool, err error, permanentReason string) error {
	if errors.Is(ctx.Err(), context.Canceled) {
		return ctx.Err()
	}
	if errors.Is(ctx.Err(), context.DeadlineExceeded) {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
		defer cancel()
	}
	if recoveryTransient(err) {
		if !pending {
			return errors.Join(err, r.Model.ObserveLearnerAvailability(ctx, id, true))
		}
		s, loadErr := r.Model.Repository.Load(ctx, id)
		if loadErr != nil {
			return errors.Join(err, loadErr)
		}
		if s.Recovery != nil && r.Model.Clock.Unix()-s.Recovery.Started >= coordinator.LearnerRecoverySeconds {
			return errors.Join(err, r.Model.Fail(ctx, id, "learner same-allocation recovery budget exhausted"))
		}
		return err
	}
	// Remote diagnostics never enter public run state.
	return errors.Join(err, r.Model.Fail(ctx, id, permanentReason))
}

func checkLearnerStatus(s coordinator.State, slot run.Slot, status nodeagent.LearnerStatus) error {
	if s.Commit == nil || status.Binding != s.Commit.Binding || status.Allocation != slot.Allocation || status.WorkerID != slot.Member.WorkerID {
		return fmt.Errorf("learner status changed committed identity")
	}
	return nil
}

// Tick returns busy while recovery owns progression. Lease maintenance and the
// model's original deadlines still run on every service tick.
func (r DiLoCoRecovery) Tick(ctx context.Context, id string) (busy bool, err error) {
	stage := "recovery ports"
	defer func() { err = diLoCoStageError(stage, err) }()
	if r.Node == nil || r.Current == nil || r.Clock == nil || r.Sign == nil || r.Grant == nil || r.Issue == nil {
		return false, fmt.Errorf("complete learner recovery ports required")
	}
	stage = "recovery state"
	s, err := r.Model.Repository.Load(ctx, id)
	if err != nil {
		return false, err
	}
	if s.Phase != coordinator.Ready && s.Phase != coordinator.Open && s.Phase != coordinator.Evaluating {
		return false, nil
	}
	if s.Recovery != nil && r.Clock().Unix()-s.Recovery.Started >= coordinator.LearnerRecoverySeconds {
		stage = "recovery deadline"
		return true, r.Model.Fail(ctx, id, "learner same-allocation recovery budget exhausted")
	}
	for _, slot := range s.Plan.Slots {
		if s.Recovery != nil && s.Recovery.WorkerID != slot.Member.WorkerID {
			continue
		}
		stage = "recovery node client"
		node, e := r.Node(slot.Allocation)
		if e != nil {
			return true, r.failure(ctx, id, s.Recovery != nil, e)
		}
		stage = "recovery learner status"
		status, e := node.LearnerStatus(ctx, slot.Allocation.JobID)
		if e == nil {
			e = checkLearnerStatus(s, slot, status)
		}
		if e != nil {
			return true, r.failure(ctx, id, s.Recovery != nil, e)
		}
		stage = "recovery current attempt"
		current, _, e := s.CurrentAttempt(slot.Member.WorkerID)
		if e != nil {
			return true, r.failure(ctx, id, s.Recovery != nil, e)
		}
		if s.Recovery == nil {
			stage = "recovery fence"
			if status.Attempt != current || status.Recovery != nil {
				return true, r.failure(ctx, id, false, fmt.Errorf("unowned node recovery fence"))
			}
			if !status.Stopped {
				continue
			}
			stage = "recovery begin"
			return true, r.Model.BeginLearnerRecovery(ctx, id, slot.Member.WorkerID)
		}
		intent := *s.Recovery
		if status.Attempt == intent.Attempt && current == intent.Attempt && status.Recovery == nil {
			// Includes a new child that already exited: the next tick counts another
			// retry, never replays Start against that terminal attempt.
			stage = "recovery finish"
			return true, r.Model.FinishLearnerRecovery(ctx, id, intent.Attempt)
		}
		stage = "recovery fence"
		if status.Attempt != intent.Previous || !status.Stopped {
			return true, r.failure(ctx, id, true, fmt.Errorf("old learner is not fenced"))
		}
		stage = "recovery resume"
		e = r.resume(ctx, s, slot, node)
		if e != nil {
			return true, r.failure(ctx, id, true, e)
		}
		return true, nil
	}
	stage = "recovery availability"
	return false, r.Model.ObserveLearnerAvailability(ctx, id, false)
}

func (r DiLoCoRecovery) resume(ctx context.Context, s coordinator.State, slot run.Slot, node LearnerRecoveryNode) (err error) {
	stage := "authorize retry"
	// Log only stage and returned errors, never request/proof/credential objects.
	defer func() { err = diLoCoStageError(stage, err) }()
	intent := *s.Recovery
	id := s.Plan.Cohort.RunID
	remaining := time.Duration(coordinator.LearnerRecoverySeconds-(r.Clock().Unix()-intent.Started)) * time.Second
	ctx, cancel := context.WithTimeout(ctx, remaining)
	defer cancel()
	prepared, err := node.AuthorizeLearnerRetry(ctx, slot.Allocation.JobID, NodeLearnerRetryRequest{Attempt: intent.Attempt, PrivateHash: intent.PrivateHash})
	if err != nil {
		return err
	}
	stage = "validate retry intent"
	b := prepared.Evidence.Body
	if b.Attempt.PreviousAttemptID != intent.Previous || b.Attempt.AttemptID != intent.Attempt || b.Attempt.PrivateHash != intent.PrivateHash || b.Attempt.Allocation != slot.Allocation || b.Attempt.WorkerID != intent.WorkerID || b.Attempt.Binding != s.Commit.Binding {
		return fmt.Errorf("node recovery changed intent")
	}
	auth := recoveryauth.Authorization{Anchor: r.Anchor, Current: func(ctx context.Context) (trust.VerifiedSnapshot, error) {
		return r.currentForRecoveryEvidence(ctx, prepared.Evidence.Proof, func(v trust.VerifiedSnapshot) error {
			return prepared.Evidence.Verify(r.Anchor, v, r.Clock())
		})
	}, Clock: r.Clock, Coordinator: s.Plan.Coordinator, Evidence: prepared.Evidence, Confirm: func(ctx context.Context, want nodejob.LearnerRecovery) error {
		status, err := node.LearnerStatus(ctx, slot.Allocation.JobID)
		if err != nil {
			return err
		}
		if err = checkLearnerStatus(s, slot, status); err != nil {
			return err
		}
		if !status.Stopped || status.Attempt != want.Attempt.PreviousAttemptID || status.Recovery == nil || *status.Recovery != want {
			return fmt.Errorf("node recovery fence changed")
		}
		return nil
	}}
	stage = "bind attempt"
	if err = r.Model.BindAttempt(ctx, b.Attempt, auth); err != nil {
		return err
	}
	stage = "validate workload scope"
	scope := prepared.Request.Request.Scope
	// Reconstruct the entire allowed scope rather than signing fields chosen by
	// the remote node. Only its fresh public workload identity is node-selected.
	var registration run.Registration
	for _, reg := range s.Commit.Registrations {
		if reg.Slot == slot {
			registration = reg
		}
	}
	expected := workload.Scope{Cluster: s.Plan.Coordinator.ClusterID, Controller: s.Plan.Coordinator.PrincipalID, Node: slot.Allocation.NodeID, Lease: slot.Allocation.LeaseID, Run: id, Job: slot.Allocation.JobID, Attempt: intent.Attempt, Workload: scope.Workload, Generation: b.Attempt.WorkloadGeneration, ManifestHash: registration.JobManifestHash, Created: b.Created, AdmitUntil: b.Expires, Deadline: scope.Deadline, Profile: workload.LearnerVersion, Worker: intent.WorkerID, PlanHash: s.Plan.SHA256, Coordinator: s.Plan.Coordinator.PrincipalID}
	if scope != expected {
		return fmt.Errorf("recovery workload scope changed")
	}
	// The original manifest's runtime deadline is checked by the production
	// grant adapter, which owns the accepted submission and manifest evidence.
	stage = "verify workload request"
	request, err := prepared.Request.Request.SigningRequest()
	if err != nil {
		return err
	}
	verifyRequest := func(v trust.VerifiedSnapshot) error {
		_, err := trust.VerifyProof(r.Anchor, v, prepared.Request.Proof, request, r.Clock())
		return err
	}
	view, err := r.currentForRecoveryEvidence(ctx, prepared.Request.Proof, verifyRequest)
	if err != nil {
		return err
	}
	if err = verifyRequest(view); err != nil {
		return err
	}
	stage = "grant workload"
	grant, err := r.Grant(ctx, prepared.Request)
	if err != nil {
		return err
	}
	stage = "issue workload credential"
	credential, err := r.Issue(ctx, grant)
	if err != nil {
		return err
	}
	stage = "sign recovery acknowledgement"
	h, err := run.Digest(b.Attempt)
	if err != nil {
		return err
	}
	proof, err := r.Sign(ctx, trust.SignRequest{Version: trust.ProofVersion, Purpose: trust.RecoveryGrant, Digest: h, Context: id, Audience: s.Plan.Coordinator.Audience})
	if err != nil {
		return err
	}
	stage = "start retry"
	return node.StartLearnerRetry(ctx, slot.Allocation.JobID, NodeLearnerRetryStart{Acknowledgement: authenticated.Recovery{Body: b.Attempt, Coordinator: s.Plan.Coordinator, Proof: proof}, Grant: grant, Credential: credential})
}
