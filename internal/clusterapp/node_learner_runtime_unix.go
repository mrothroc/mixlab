//go:build darwin || linux

package clusterapp

import (
	"context"
	"fmt"
	"io"
	"time"

	private "github.com/mrothroc/mixlab/diloco/learner"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/learneripc"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/nodecredentials"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/principal"
	wc "github.com/mrothroc/mixlab/workercontrol"
	"github.com/mrothroc/mixlab/workerhost"
	"github.com/mrothroc/mixlab/workerhost/contract"
)

// NodeLearnerRuntime composes protected node credentials with opaque hosting.
// Every RPC rechecks the live attempt; no cached client can outlive fencing.
type NodeLearnerRuntime struct {
	Store       *nodeagent.Store
	Principal   *principal.Store
	Runtime     *workerhost.GuardianRuntime
	Credentials statehome.Path
	Anchor      trust.Anchor
	Clock       func() time.Time
	// Event is a protected agent-local diagnostic sink, never a child reply.
	Event func(error)
}

func (r NodeLearnerRuntime) Open(ctx context.Context, x nodeagent.LocalExecution) (*workerhost.AttemptStore, workerhost.Runner, error) {
	if r.Store == nil || r.Principal == nil || r.Runtime == nil || r.Clock == nil || x.Learner == nil || x.Learner.Assignment == nil || x.Approval == nil {
		return nil, nil, fmt.Errorf("approved learner and local runtime required")
	}
	a := x.Learner.Assignment
	binding, err := a.Binding()
	if err != nil {
		return nil, nil, err
	}
	dir, err := r.Runtime.LearnerPrivate(ctx, a.RunID, a.Member.WorkerID)
	if err != nil {
		return nil, nil, err
	}
	server := r.relayServer(binding, min(a.PrivateMaxBytes, 1<<30))
	server.SessionLifetime = time.Duration(a.RuntimeSeconds) * time.Second
	relay := func(ctx context.Context, e wc.Envelope) (wc.Envelope, error) {
		if e.JobID != binding.JobID || e.AttemptID != binding.AttemptID {
			return wc.Envelope{}, fmt.Errorf("learner IPC attempt changed")
		}
		return server.Handle(ctx, e)
	}
	host, runner, err := r.Runtime.OpenLearner(ctx, a.JobID, a.AttemptID, relay, dir)
	if err != nil {
		server.Close()
		return nil, nil, err
	}
	return host, learnerClosingRunner{Runner: runner, server: server}, nil
}

func (r NodeLearnerRuntime) relayServer(binding wc.Binding, maxBytes uint64) *learneripc.Server {
	server := &learneripc.Server{Remote: &nodeLearnerRemote{runtime: r, binding: binding}, MaxBytes: maxBytes}
	if r.Event != nil {
		server.OnError = func(operation string, err error) {
			// Context comes from the admitted attempt, not a request body. Quote
			// the diagnostic so joined errors remain a single local log record.
			message := err.Error()
			const limit, truncated = 2048, " [truncated]"
			if len(message) > limit {
				message = message[:limit-len(truncated)] + truncated
			}
			r.Event(fmt.Errorf("learner relay error: time=%s operation=%q job=%q attempt=%q error=%q", r.Clock().UTC().Format(time.RFC3339Nano), operation, binding.JobID, binding.AttemptID, message))
		}
	}
	return server
}

type learnerClosingRunner struct {
	workerhost.Runner
	server *learneripc.Server
}

func (r learnerClosingRunner) Run(ctx context.Context, a contract.Approved, started func(int) error) error {
	defer r.server.Close()
	return r.Runner.Run(ctx, a, started)
}

type nodeLearnerRemote struct {
	runtime NodeLearnerRuntime
	binding wc.Binding
}

func (r *nodeLearnerRemote) execution(ctx context.Context, terminalRead bool) (*nodeagent.LocalExecution, error) {
	o := r.runtime
	anchor, view, err := NodeCurrentTrust(o.Principal)(ctx, o.Clock())
	if err != nil {
		return nil, err
	}
	if anchor.Fingerprint() != o.Anchor.Fingerprint() {
		return nil, fmt.Errorf("node trust pin changed")
	}
	if err := o.Store.CheckExecutionTrust(ctx, anchor, view, o.Clock()); err != nil {
		return nil, err
	}
	x, err := o.Store.ActiveExecution(ctx)
	if err != nil {
		return nil, err
	}
	if x == nil || x.Learner == nil || x.Learner.Assignment == nil || x.Learner.Stopped {
		return nil, fmt.Errorf("learner attempt no longer live")
	}
	if x.Lease.State == nodeagent.Releasing && (!terminalRead || !x.LearnerDraining(o.Clock())) {
		return nil, fmt.Errorf("learner attempt no longer live")
	}
	b, err := x.Learner.Assignment.Binding()
	if err != nil || b != r.binding {
		return nil, fmt.Errorf("learner IPC belongs to another attempt")
	}
	return x, nil
}

func (r *nodeLearnerRemote) client(ctx context.Context, terminalRead bool) (*LearnerClient, func(), error) {
	x, err := r.execution(ctx, terminalRead)
	if err != nil {
		return nil, nil, err
	}
	o := r.runtime
	if x.Learner.Commit == nil {
		return nil, func() {}, nil
	}
	if x.Learner.Credential == nil {
		return nil, nil, fmt.Errorf("learner credential missing")
	}
	dir, err := workloadDirectory(o.Credentials, x.Workload)
	if err != nil {
		return nil, nil, err
	}
	credential, err := nodecredentials.OpenTransport(dir, o.Anchor, x.Workload)
	if err != nil {
		return nil, nil, err
	}
	close := func() { _ = credential.Close() }
	client, err := NewLearnerClient(o.Principal, credential, x.Learner.Signed.Manifest.Plan, *x.Learner.Commit, x.Learner.Credential.Binding, o.Clock)
	if err != nil {
		close()
		return nil, nil, err
	}
	return client, close, nil
}
func (r *nodeLearnerRemote) Poll(ctx context.Context) (private.RelayResponse, error) {
	c, close, err := r.client(ctx, true)
	if err != nil {
		return private.RelayResponse{}, err
	}
	defer close()
	// The admitted child may become ready before the atomic run commit.
	// Return promptly so preparation cannot deadlock behind a pending poll.
	if c == nil {
		return private.RelayResponse{Format: "mixlab_learner_reply_v1"}, nil
	}
	out, err := c.Poll(ctx)
	if err != nil {
		return out, err
	}
	x, err := r.execution(ctx, true)
	if err != nil {
		return private.RelayResponse{}, err
	}
	if x.LearnerDraining(r.runtime.Clock()) && !out.Done {
		// A release may win while the reply is in flight. Drain only a real
		// terminal reply; never launch stale work or invent successful completion.
		return private.RelayResponse{Format: "mixlab_learner_reply_v1"}, nil
	}
	return out, nil
}
func (r *nodeLearnerRemote) committed(ctx context.Context, terminalRead bool) (*LearnerClient, func(), error) {
	c, close, err := r.client(ctx, terminalRead)
	if err == nil && c == nil {
		close()
		return nil, nil, fmt.Errorf("learner is not committed")
	}
	return c, close, err
}
func (r *nodeLearnerRemote) Call(ctx context.Context, q private.RelayRequest) (private.RelayResponse, error) {
	// Poll must use Poll's final drain recheck, not the generic Call path.
	if q.Operation == "poll" {
		return private.RelayResponse{}, fmt.Errorf("poll requires the polling path")
	}
	c, close, err := r.committed(ctx, q.TerminalRead())
	if err != nil {
		return private.RelayResponse{}, err
	}
	defer close()
	return c.Call(ctx, q)
}
func (r *nodeLearnerRemote) Read(ctx context.Context, a run.Assignment, b run.Blob, w io.Writer) error {
	c, close, err := r.committed(ctx, false)
	if err != nil {
		return err
	}
	defer close()
	return c.Read(ctx, a, b, w)
}
func (r *nodeLearnerRemote) Write(ctx context.Context, a run.Assignment, b run.Blob, rd io.Reader) error {
	c, close, err := r.committed(ctx, false)
	if err != nil {
		return err
	}
	defer close()
	return c.Write(ctx, a, b, rd)
}
func (r *nodeLearnerRemote) ReadEvaluation(ctx context.Context, a run.EvaluationAssignment, w io.Writer) error {
	c, close, err := r.committed(ctx, false)
	if err != nil {
		return err
	}
	defer close()
	return c.ReadEvaluation(ctx, a, w)
}
