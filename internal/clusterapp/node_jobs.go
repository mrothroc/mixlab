package clusterapp

import (
	"context"
	"errors"
	"fmt"
	"github.com/mrothroc/mixlab/artifact"
	"net"
	"sync"
	"time"

	"github.com/mrothroc/mixlab/grouptransport"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/nodecredentials"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/transport/ringproxy"
	"github.com/mrothroc/mixlab/transport/ringtls"
	"github.com/mrothroc/mixlab/trust"
)

// NodeJobOptions contains administrator-local ports, never HTTP input.
// CurrentTrust must validate this node's live principal as well as its snapshot.
type NodeJobOptions struct {
	Store          *nodeagent.Store
	Node           string
	Anchor         trust.Anchor
	CredentialRoot statehome.Path
	Workload       nodeagent.WorkloadPorts
	Preparation    nodeagent.PreparationPorts
	CurrentTrust   func(context.Context, time.Time) (trust.Anchor, trust.VerifiedSnapshot, error)
	RelayAddress   string
	Clock          func() time.Time
	Output         func(context.Context, string, string) (statehome.Path, artifact.Ref, error)
	InputRoot      statehome.Path
}

// NodeJobs owns in-process relay lifetime. The durable node store owns launch
// permission; the credential context owns keys. Restart never resumes a relay.
type NodeJobs struct {
	options NodeJobOptions
	life    context.Context
	mu      sync.Mutex
	relay   *nodeRelay
}

func NewNodeJobs(life context.Context, o NodeJobOptions) (*NodeJobs, error) {
	if life == nil || life.Err() != nil || o.Store == nil || !nodeRouteID(o.Node) || o.Anchor.Cluster() == "" || o.CurrentTrust == nil || o.Clock == nil || o.CredentialRoot.Kind() != statehome.Agent || o.Workload.Initialize == nil || o.Workload.Request == nil || o.Workload.Clock == nil || o.Preparation.Probe == nil || o.Preparation.DatasetID == nil || o.Preparation.ArtifactPresent == nil || o.Preparation.Clock == nil {
		return nil, fmt.Errorf("live node context, protected credentials and local job ports required")
	}
	if err := o.CredentialRoot.Validate(); err != nil {
		return nil, err
	}
	if err := nodeagent.ValidateTransportEndpoint(o.RelayAddress); err != nil {
		return nil, err
	}
	return &NodeJobs{options: o, life: life}, nil
}

func (a *NodeJobs) current(ctx context.Context) (trust.VerifiedSnapshot, error) {
	if err := ctx.Err(); err != nil {
		return trust.VerifiedSnapshot{}, err
	}
	if err := a.life.Err(); err != nil {
		return trust.VerifiedSnapshot{}, err
	}
	anchor, view, err := a.options.CurrentTrust(ctx, a.options.Clock())
	if err != nil {
		return trust.VerifiedSnapshot{}, err
	}
	if anchor.Fingerprint() != a.options.Anchor.Fingerprint() {
		return trust.VerifiedSnapshot{}, fmt.Errorf("node trust pin changed")
	}
	return view, ctx.Err()
}

func (a *NodeJobs) Prepare(ctx context.Context, actor trust.AuthenticatedPrincipal, q NodePrepareRequest) (nodeagent.PreparedWorkload, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	v, err := a.current(ctx)
	if err != nil {
		return nodeagent.PreparedWorkload{}, err
	}
	accepted, err := nodejob.Accept(a.options.Anchor, v, actor, a.options.Node, q.Signed, a.options.Clock())
	if err != nil {
		return nodeagent.PreparedWorkload{}, err
	}
	preparation := a.options.Preparation
	if len(q.Signed.Manifest.Artifacts) > 0 {
		if _, err := a.preparedInput(ctx, q.Signed.Manifest); err != nil {
			return nodeagent.PreparedWorkload{}, err
		}
		preparation.ArtifactPresent = func(context.Context, nodejob.ArtifactRef) error { return nil }
	}
	j, err := a.options.Store.PrepareChecked(ctx, actor, accepted, q.ExpectedLeaseVersion, preparation, a.options.Clock())
	if err != nil {
		return nodeagent.PreparedWorkload{}, err
	}
	return a.options.Store.PrepareWorkload(ctx, actor, j.ID, j.Version, a.options.Workload)
}

func (a *NodeJobs) active(ctx context.Context, actor trust.AuthenticatedPrincipal, job string) (*nodeagent.LocalExecution, error) {
	if _, err := a.options.Store.JobStatus(actor, job, a.options.Clock()); err != nil {
		return nil, err
	}
	x, err := a.options.Store.ActiveExecution(ctx)
	if err != nil {
		return nil, err
	}
	if x == nil || x.Job == nil || x.Job.ID != job || x.Lease.State == nodeagent.Releasing || x.Job.CancelRequested || x.Outcome != nil && x.Outcome.Terminal() {
		return nil, fmt.Errorf("job no longer active")
	}
	return x, nil
}

func (a *NodeJobs) Transport(ctx context.Context, actor trust.AuthenticatedPrincipal, job string, q NodeTransportRequest) (out nodeagent.Job, err error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	v, err := a.current(ctx)
	if err != nil {
		return out, err
	}
	x, err := a.active(ctx, actor, job)
	if err != nil {
		return out, err
	}
	if x.Workload.Node != a.options.Node || x.Workload != q.Grant.Request.Request.Scope {
		return out, fmt.Errorf("transport grant differs from locally prepared workload")
	}
	accepted, err := grouptransport.Accept(a.options.Anchor, v, actor, x.Manifest, q.Credential.Chain, q.Signed, a.options.Clock())
	if err != nil {
		return out, err
	}
	if q.Signed.Plan.Members[x.Manifest.Rank].Endpoint != a.options.RelayAddress {
		return out, fmt.Errorf("transport listener differs from administrator policy")
	}
	dir, err := workloadDirectory(a.options.CredentialRoot, x.Workload)
	if err != nil {
		return out, err
	}
	c, err := nodecredentials.OpenTransport(dir, a.options.Anchor, x.Workload)
	if err != nil {
		return out, err
	}
	owned := false
	defer func() {
		if !owned {
			_ = c.Close()
		}
	}()
	if err := c.Install(ctx, q.Grant, q.Credential, v, a.options.Clock()); err != nil {
		return out, err
	}
	chain, key, err := c.Identity(v, a.options.Clock())
	if err != nil {
		return out, err
	}
	out, err = a.options.Store.ActivateTransport(ctx, actor, accepted, q.ExpectedVersion, a.options.Clock())
	if err != nil {
		return out, err
	}
	if a.relay != nil {
		if a.relay.job != job || a.relay.hash != out.TransportHash {
			return out, fmt.Errorf("another relay owns this node")
		}
		return out, a.relay.live()
	}
	// The journal now owns cleanup even if binding fails. Never compensate by
	// deleting a key that a retry could accidentally try to reuse.
	defer func() {
		if err != nil {
			fence, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			err = errors.Join(err, a.options.Store.InterruptActive(fence, "authority_unavailable"))
		}
	}()
	l, err := (&net.ListenConfig{}).Listen(ctx, "tcp", a.options.RelayAddress)
	if err != nil {
		return out, err
	}
	r, err := ringproxy.New(ringproxy.Options{Accepted: accepted, LocalRank: x.Manifest.Rank, Identity: ringtls.Identity{Chain: chain, Key: key}, CurrentTrust: a.options.CurrentTrust, Clock: a.options.Clock, Public: l, ConnectTimeout: 30 * time.Second, MaxConnections: 32})
	if err != nil {
		_ = l.Close()
		return out, err
	}
	a.relay = startNodeRelay(a.life, job, out.TransportHash, r, c)
	owned = true
	return out, nil
}

func (a *NodeJobs) Start(ctx context.Context, actor trust.AuthenticatedPrincipal, q nodeagent.StartCommand) (nodeagent.Job, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	v, err := a.current(ctx)
	if err != nil {
		return nodeagent.Job{}, err
	}
	if err := a.options.Store.CheckExecutionTrust(ctx, a.options.Anchor, v, a.options.Clock()); err != nil {
		return nodeagent.Job{}, err
	}
	x, err := a.active(ctx, actor, q.Job)
	if err != nil {
		return nodeagent.Job{}, err
	}
	if a.relay == nil || a.relay.job != q.Job || a.relay.hash != x.Job.TransportHash {
		return nodeagent.Job{}, fmt.Errorf("live committed relay required before worker start")
	}
	if err := a.relay.live(); err != nil {
		return nodeagent.Job{}, err
	}
	assignment, err := a.options.Store.ResolveAssignment(ctx, q.Job)
	if err != nil {
		return nodeagent.Job{}, err
	}
	if len(x.Manifest.Artifacts) > 0 {
		path, err := a.preparedInput(ctx, x.Manifest)
		if err != nil {
			return nodeagent.Job{}, err
		}
		ref, err := checkpointRef(x.Manifest)
		if err != nil {
			return nodeagent.Job{}, err
		}
		assignment.Resume, assignment.ResumePath = &ref, path
	}
	if _, err := a.options.Store.StartJob(ctx, actor, q, assignment, a.options.Clock()); err != nil {
		return nodeagent.Job{}, err
	}
	return a.options.Store.JobStatus(actor, q.Job, a.options.Clock())
}

var _ NodeJobOperations = (*NodeJobs)(nil)
