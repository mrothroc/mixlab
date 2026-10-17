package clusterapp

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/mrothroc/mixlab/admission"
	"github.com/mrothroc/mixlab/coordinator"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/recruitment"
	"github.com/mrothroc/mixlab/trust"
)

// ErrAdmissionLeaseLost requires compensation before commit, or a failed run
// and continued handoff/cleanup reconciliation after commit. It never permits
// replacement, resurrection, or opening rounds before admission is Done.
var ErrAdmissionLeaseLost = errors.New("admission allocation lost")

const admissionLeaseCallBudget = learnerAttemptBudget
const admissionLeaseBudget = diLoCoServiceBudget - admission.StepTimeout
const admissionLeaseRenewLead = diLoCoServiceBudget + diLoCoServiceMaxInterval + admissionLeaseBudget
const admissionLeaseConcurrency = 8

type AdmissionLeaseClient interface {
	Reserve(context.Context, nodeagent.Reserve) (nodeagent.Lease, error)
	LeaseStatus(context.Context, string) (nodeagent.Lease, error)
	Renew(context.Context, nodeagent.LeaseCommand) (nodeagent.Lease, error)
}

// DiLoCoAdmissionLeases maintains original allocations during long admission
// and partial handoff. It holds no keys, changes no saga/plan/proof, and never
// opens rounds. Call before forward saga effects, not as a gate on compensation
// or committed handoff reconciliation. All ports must honor context deadlines.
type DiLoCoAdmissionLeases struct {
	Client     func(trust.Role, recruitment.Candidate) (AdmissionLeaseClient, error)
	Repository coordinator.Repository
	Clock      func() time.Time
	// Only scan order is local, never lease authority or admission progress.
	// The owning inbox bounds retained runs; restart may safely restart a scan.
	scans sync.Map
}

func (k *DiLoCoAdmissionLeases) Maintain(ctx context.Context, q admission.Submission, v admission.Snapshot) error {
	if k.Client == nil || k.Clock == nil {
		return fmt.Errorf("admission lease clients and clock required")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := q.Validate(); err != nil {
		return err
	}
	want, err := run.Digest(q)
	if err != nil {
		return err
	}
	got, err := run.Digest(admission.Submission{Owner: q.Owner, Launch: v.Request, Specification: v.Specification, Deadline: v.Deadline})
	if err != nil || got != want || len(v.Nodes) != len(q.Launch.Selection.Selected) || len(v.Nodes) > 64 {
		return fmt.Errorf("admission lease snapshot differs from accepted submission")
	}
	if v.Done || v.Aborting && !v.Committed {
		k.scans.Delete(q.Launch.Run)
		return nil
	}
	if !v.Committed && k.Clock().Unix() >= v.Deadline {
		return fmt.Errorf("%w: admission deadline expired", ErrAdmissionLeaseLost)
	}
	budget := admissionLeaseBudget
	if !v.Committed {
		budget = min(budget, time.Unix(v.Deadline, 0).Sub(k.Clock()))
	}
	ctx, cancel := context.WithTimeout(ctx, budget)
	defer cancel()
	if v.Committed {
		if k.Repository == nil || v.Commit == nil || v.Plan == nil {
			return fmt.Errorf("committed admission repository and evidence required")
		}
		s, err := k.Repository.Load(ctx, q.Launch.Run)
		if err != nil {
			return err
		}
		if err = s.Validate(); err != nil {
			return err
		}
		if s.Commit == nil || s.Commit.Binding != v.Commit.Binding || s.Plan.SHA256 != v.Plan.SHA256 {
			return fmt.Errorf("%w: coordinator commit changed", ErrAdmissionLeaseLost)
		}
		if s.Phase == coordinator.Failed || s.Phase == coordinator.Complete || s.Phase == coordinator.Aborted {
			return fmt.Errorf("%w: coordinator run is terminal", ErrAdmissionLeaseLost)
		}
		if s.Phase != coordinator.Ready {
			return fmt.Errorf("%w: rounds started before handoff completed", ErrAdmissionLeaseLost)
		}
	}
	// Independent nodes cannot starve each other's renewals behind an unavailable
	// handoff target. The fixed worker count bounds concurrent sockets and calls.
	start := 0
	if saved, ok := k.scans.Load(q.Launch.Run); ok {
		start = saved.(int) % len(v.Nodes)
	}
	jobs := make(chan int, len(v.Nodes))
	for offset := range v.Nodes {
		jobs <- (start + offset) % len(v.Nodes)
	}
	close(jobs)
	failures := make([]error, len(v.Nodes))
	attempted := make([]bool, len(v.Nodes))
	interrupted := make([]bool, len(v.Nodes))
	var wg sync.WaitGroup
	for range min(admissionLeaseConcurrency, len(v.Nodes)) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for n := range jobs {
				if err := ctx.Err(); err != nil {
					failures[n] = err
					continue
				}
				attempted[n] = true
				failures[n] = k.maintainNode(ctx, q, v, n)
				if err := ctx.Err(); err != nil {
					interrupted[n] = errors.Is(failures[n], err)
				}
			}
		}()
	}
	wg.Wait()
	// A 64-node cohort cannot all consume full 50s calls in one turn with
	// eight workers. Give scan-interrupted work a full turn before unvisited
	// slots; a started status read is not a completed renewal. If the starting
	// slot still cannot finish with a full turn, rotate a worker cohort rather
	// than repeatedly prioritizing that prefix. All effects re-read live state.
	for offset := range v.Nodes {
		n := (start + offset) % len(v.Nodes)
		if interrupted[n] || !attempted[n] {
			if n == start && interrupted[n] {
				n = (start + min(admissionLeaseConcurrency, len(v.Nodes))) % len(v.Nodes)
			}
			k.scans.Store(q.Launch.Run, n)
			break
		}
	}
	if !v.Committed && k.Clock().Unix() >= v.Deadline {
		failures = append(failures, fmt.Errorf("%w: admission deadline expired", ErrAdmissionLeaseLost))
	}
	return errors.Join(append(failures, ctx.Err())...)
}

func admissionLeaseCall(ctx context.Context, f func(context.Context) (nodeagent.Lease, error)) (nodeagent.Lease, error) {
	call, cancel := context.WithTimeout(ctx, admissionLeaseCallBudget)
	defer cancel()
	l, err := f(call)
	return l, errors.Join(err, call.Err())
}

func (k *DiLoCoAdmissionLeases) callUntil(ctx context.Context, until int64, f func(context.Context) (nodeagent.Lease, error)) (nodeagent.Lease, error) {
	remaining := time.Unix(until, 0).Sub(k.Clock())
	if remaining <= 0 {
		return nodeagent.Lease{}, fmt.Errorf("%w: allocation authority deadline expired", ErrAdmissionLeaseLost)
	}
	call, cancel := context.WithTimeout(ctx, remaining)
	defer cancel()
	l, err := admissionLeaseCall(call, f)
	if k.Clock().Unix() >= until {
		err = errors.Join(err, fmt.Errorf("%w: allocation authority deadline expired", ErrAdmissionLeaseLost))
	}
	return l, err
}

func (k *DiLoCoAdmissionLeases) maintainNode(ctx context.Context, q admission.Submission, v admission.Snapshot, index int) error {
	n := v.Nodes[index]
	if !n.Touched {
		if n.Lease != nil || n.Finalized || n.Released {
			return fmt.Errorf("%w: untouched allocation has effects", ErrAdmissionLeaseLost)
		}
		return nil
	}
	if n.Released {
		return fmt.Errorf("%w: allocation already released", ErrAdmissionLeaseLost)
	}
	candidate := q.Launch.Selection.Selected[index]
	var controller AdmissionLeaseClient
	getController := func() error {
		if controller != nil {
			return nil
		}
		var err error
		controller, err = k.Client(trust.Controller, candidate)
		if err != nil {
			return err
		}
		if controller == nil {
			return fmt.Errorf("controller lease client required")
		}
		return nil
	}
	var lease nodeagent.Lease
	var err error
	if n.Lease == nil {
		if v.Committed || n.Finalized {
			return fmt.Errorf("%w: committed reservation history missing", ErrAdmissionLeaseLost)
		}
		// The durable touched intent permits only this exact reservation replay.
		// Reserve's response is immutable, so always follow it with live status.
		h, _ := run.Digest(struct{ Run, Attempt, Node string }{q.Launch.Run, q.Launch.Attempt, candidate.Capabilities.Node})
		want := nodeagent.Reserve{IdempotencyKey: h[:32], ExpectedNodeVersion: candidate.Capabilities.Availability.NodeVersion, CapabilityGeneration: candidate.Capabilities.Generation, Run: q.Launch.Run, TTLSeconds: q.Launch.TTLSeconds}
		if n.Reserve != want {
			return fmt.Errorf("%w: reservation intent changed", ErrAdmissionLeaseLost)
		}
		if err := getController(); err != nil {
			return err
		}
		lease, err = k.callUntil(ctx, v.Deadline, func(ctx context.Context) (nodeagent.Lease, error) { return controller.Reserve(ctx, n.Reserve) })
		if err != nil {
			return err
		}
		if !nodeRouteID(lease.ID) {
			return fmt.Errorf("%w: invalid recovered lease ID", ErrAdmissionLeaseLost)
		}
	} else {
		lease = *n.Lease
	}
	client, role := controller, trust.Controller
	var handoffErr error
	if v.Committed {
		c, e := k.Client(trust.Coordinator, candidate)
		if e != nil {
			return e
		}
		if c == nil {
			return fmt.Errorf("coordinator lease client required")
		}
		live, e := admissionLeaseCall(ctx, func(ctx context.Context) (nodeagent.Lease, error) { return c.LeaseStatus(ctx, lease.ID) })
		switch {
		case e == nil:
			lease, client, role = live, c, trust.Coordinator
		case n.Finalized:
			return e
		default:
			handoffErr = e
		}
	}
	if role == trust.Controller {
		if err := getController(); err != nil {
			return errors.Join(handoffErr, err)
		}
		client = controller
		// This is observation, not renewed controller authority. After a lost
		// handoff ACK the coordinator may own a live lease beyond admission;
		// expiry of the controller window alone cannot prove allocation loss.
		// Precommit observation remains capped by Maintain's admission context.
		lease, err = admissionLeaseCall(ctx, func(ctx context.Context) (nodeagent.Lease, error) { return controller.LeaseStatus(ctx, lease.ID) })
		if err != nil {
			return errors.Join(handoffErr, err)
		}
	}
	if err := k.match(q, v, index, lease, role); err != nil {
		return err
	}
	now := k.Clock().Unix()
	cap := v.Deadline
	if role == trust.Coordinator {
		cap = lease.WorkloadDeadline
	}
	if cap <= now {
		return fmt.Errorf("%w: allocation authority deadline expired", ErrAdmissionLeaseLost)
	}
	// Renew before another service turn, polling interval and maintenance scan
	// can consume the remaining lease. Short legacy TTLs cannot create headroom
	// beyond their accepted TTL or authority cap; never extend either here.
	if now < lease.RenewBy && time.Duration(lease.Expires-now)*time.Second > admissionLeaseRenewLead {
		return nil
	}
	if lease.Expires >= cap {
		return nil
	}
	ttl := min(q.Launch.TTLSeconds, int(cap-now))
	if ttl < 10 || now+int64(ttl) <= lease.Expires {
		return nil
	}
	h, err := run.Digest(struct {
		Run, Lease string
		Role       trust.Role
		Version    uint64
		TTL        int
	}{q.Launch.Run, lease.ID, role, lease.Version, ttl})
	if err != nil {
		return err
	}
	command := nodeagent.LeaseCommand{IdempotencyKey: h[:32], Lease: lease.ID, ExpectedVersion: lease.Version, TTLSeconds: ttl}
	// Expiry of the observed lease bounds this attempt, but is not proof of
	// loss: an ambiguous renewal may already have advanced its live expiry.
	renewCtx, stopRenew := context.WithTimeout(ctx, time.Unix(lease.Expires, 0).Sub(k.Clock()))
	renewed, err := k.callUntil(renewCtx, cap, func(ctx context.Context) (nodeagent.Lease, error) { return client.Renew(ctx, command) })
	stopRenew()
	if err != nil {
		// A lost ACK must not replay an old version forever or assume success.
		var readErr error
		// The old lease may already have been extended by the ambiguous call.
		// Re-read under the authority cap, not its obsolete observed expiry.
		renewed, readErr = k.callUntil(ctx, cap, func(ctx context.Context) (nodeagent.Lease, error) { return client.LeaseStatus(ctx, lease.ID) })
		if readErr != nil {
			return errors.Join(err, readErr)
		}
		if matchErr := k.match(q, v, index, renewed, role); matchErr != nil {
			return matchErr
		}
		if renewed.Version <= lease.Version || renewed.Expires <= lease.Expires {
			return err
		}
	}
	if err := k.match(q, v, index, renewed, role); err != nil {
		return err
	}
	if renewed.ID != lease.ID || renewed.Version <= lease.Version || renewed.Expires <= lease.Expires || renewed.Expires > cap {
		return fmt.Errorf("%w: renewal changed allocation or failed to extend it", ErrAdmissionLeaseLost)
	}
	return nil
}

func (k *DiLoCoAdmissionLeases) match(q admission.Submission, v admission.Snapshot, index int, l nodeagent.Lease, role trust.Role) error {
	n, c := v.Nodes[index], q.Launch.Selection.Selected[index]
	bad := l.Format != nodeagent.LeaseVersion || !nodeRouteID(l.ID) || l.Node != c.Capabilities.Node || l.Controller != q.Launch.Controller || l.Run != q.Launch.Run || l.CapabilityGeneration != c.Capabilities.Generation || l.Version == 0 || l.Created <= 0 || l.RenewBy < l.Created || l.RenewBy > l.Expires || l.CleanupReason != ""
	bad = bad || n.Lease != nil && (l.ID != n.Lease.ID || l.Version < n.Lease.Version)
	if bad {
		return fmt.Errorf("%w: lease scope changed", ErrAdmissionLeaseLost)
	}
	if k.Clock().Unix() >= l.Expires || l.State == nodeagent.Releasing || l.State == nodeagent.Released {
		return fmt.Errorf("%w: lease expired or terminal", ErrAdmissionLeaseLost)
	}
	if role == trust.Coordinator {
		if !v.Committed || v.Commit == nil || l.Coordinator != q.Specification.Coordinator.PrincipalID || l.RunCommitHash != v.Commit.Binding.CommitHash || l.WorkloadDeadline <= k.Clock().Unix() || l.State != nodeagent.Running {
			return fmt.Errorf("%w: coordinator handoff changed", ErrAdmissionLeaseLost)
		}
	} else if n.Finalized || l.Coordinator != "" || l.RunCommitHash != "" || l.WorkloadDeadline != 0 || (l.State != nodeagent.Reserved && l.State != nodeagent.Prepared) {
		return fmt.Errorf("%w: controller authority changed", ErrAdmissionLeaseLost)
	}
	if l.State == nodeagent.Reserved {
		if l.Job != "" || l.JobKind != "" {
			return fmt.Errorf("%w: reservation unexpectedly bound", ErrAdmissionLeaseLost)
		}
	} else if v.Plan == nil || index >= len(v.Plan.Slots) || l.JobKind != "learner" || l.Job != v.Plan.Slots[index].Allocation.JobID || l.ID != v.Plan.Slots[index].Allocation.LeaseID || l.Node != v.Plan.Slots[index].Allocation.NodeID {
		return fmt.Errorf("%w: prepared allocation changed", ErrAdmissionLeaseLost)
	}
	return nil
}
