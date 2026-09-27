package clusterapp

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/nodecredentials"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/principal"
	"github.com/mrothroc/mixlab/trust/workload"
)

// NodeWorkloadPorts composes only local node-owned credential operations. The
// authority issuer and controller signer are deliberately absent. root must be
// an administrator-created private agent directory, never worker runtime state.
func NodeWorkloadPorts(p *principal.Store, root statehome.Path, clock func() time.Time) (nodeagent.WorkloadPorts, error) {
	if p == nil || clock == nil || root.Kind() != statehome.Agent {
		return nodeagent.WorkloadPorts{}, fmt.Errorf("node principal, private agent root and clock required")
	}
	if err := root.Validate(); err != nil {
		return nodeagent.WorkloadPorts{}, err
	}
	state, _, err := p.Active(clock())
	if err != nil {
		return nodeagent.WorkloadPorts{}, err
	}
	if state.Role != trust.Node {
		return nodeagent.WorkloadPorts{}, fmt.Errorf("node identity required for workload preparation")
	}
	a, err := trust.PinRoot(state.Root, state.Fingerprint, clock())
	if err != nil {
		return nodeagent.WorkloadPorts{}, err
	}
	path := func(scope workload.Scope) (statehome.Path, error) {
		if err := scope.Validate(); err != nil {
			return statehome.Path{}, err
		}
		if scope.Cluster != state.Cluster || scope.Node != state.Principal {
			return statehome.Path{}, fmt.Errorf("workload targets another node or cluster")
		}
		return workloadDirectory(root, scope)
	}
	initialize := func(ctx context.Context, scope workload.Scope) error {
		dir, err := path(scope)
		if err != nil {
			return err
		}
		credential, err := nodecredentials.InitializeTransport(ctx, dir, a, scope, state.Key.Backend)
		if errors.Is(err, statehome.ErrExists) {
			// Exact retry of empty-context publication, never a key repair.
			credential, err = nodecredentials.OpenTransport(dir, a, scope)
		}
		if err != nil {
			return err
		}
		return credential.Close()
	}
	request := func(ctx context.Context, scope workload.Scope) (workload.SignedRequest, error) {
		dir, err := path(scope)
		if err != nil {
			return workload.SignedRequest{}, err
		}
		credential, err := nodecredentials.OpenTransport(dir, a, scope)
		if err != nil {
			return workload.SignedRequest{}, err
		}
		defer func() { _ = credential.Close() }()
		now := clock()
		current, key, err := p.Active(now)
		if err != nil {
			return workload.SignedRequest{}, err
		}
		if current.Principal != scope.Node || current.Cluster != scope.Cluster || current.Role != trust.Node {
			return workload.SignedRequest{}, fmt.Errorf("node credential context changed")
		}
		view, err := trust.VerifySnapshot(a, current.Snapshot, now)
		if err != nil {
			return workload.SignedRequest{}, err
		}
		return credential.PrepareRequest(ctx, func(_ context.Context, q trust.SignRequest) (trust.SignedProof, error) {
			return trust.SignPrincipalProof(a, current.Chain, key, view, q, now)
		}, view, now)
	}
	return nodeagent.WorkloadPorts{Initialize: initialize, Request: request, Clock: clock}, nil
}

func workloadDirectory(root statehome.Path, scope workload.Scope) (statehome.Path, error) {
	if root.Kind() != statehome.Agent {
		return statehome.Path{}, fmt.Errorf("agent-owned credential root required")
	}
	if err := root.Validate(); err != nil {
		return statehome.Path{}, err
	}
	if err := scope.Validate(); err != nil {
		return statehome.Path{}, err
	}
	return statehome.Resolve(statehome.Options{ExactDir: filepath.Join(root.Dir(), "job-"+scope.Job+"-"+scope.Attempt)}, statehome.Context{Kind: statehome.Agent})
}

// DestroyNodeWorkload consumes only local terminal-cleanup evidence, never an
// HTTP body. The owner must stop the relay before calling it. Cleanup deliberately
// does not require a live principal certificate: expired/revoked nodes must still
// be able to destroy their own keys. The persisted cluster pin remains mandatory.
func DestroyNodeWorkload(ctx context.Context, root statehome.Path, anchor trust.Anchor, q nodeagent.CleanupRequest) error {
	if q.Workload == (workload.Scope{}) {
		if q.WorkloadInitialized {
			return fmt.Errorf("initialized workload scope missing from cleanup")
		}
		return nil
	}
	if q.Workload.Cluster != anchor.Cluster() || q.Workload.Job != q.Job || q.Workload.Attempt != q.Attempt {
		return fmt.Errorf("workload cleanup scope mismatch")
	}
	dir, err := workloadDirectory(root, q.Workload)
	if err != nil {
		return err
	}
	if !q.WorkloadInitialized {
		// No Request port can have run before the owning job's marker. An
		// absent empty-context publication therefore cannot conceal a key.
		if _, err := os.Lstat(dir.Dir()); errors.Is(err, os.ErrNotExist) {
			return nil
		} else if err != nil {
			return err
		}
	}
	c, err := nodecredentials.OpenTransport(dir, anchor, q.Workload)
	if err != nil {
		return err
	}
	defer func() { _ = c.Close() }()
	return c.Destroy(ctx)
}
