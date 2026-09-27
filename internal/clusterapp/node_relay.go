package clusterapp

import (
	"context"
	"errors"
	"fmt"

	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/nodecredentials"
	"github.com/mrothroc/mixlab/transport/ringproxy"
)

type nodeRelay struct {
	job, hash  string
	relay      *ringproxy.Relay
	credential *nodecredentials.Transport
	cancel     context.CancelFunc
	done       chan struct{}
	err        error // published by closing done
}

func startNodeRelay(ctx context.Context, job, hash string, relay *ringproxy.Relay, credential *nodecredentials.Transport) *nodeRelay {
	life, cancel := context.WithCancel(ctx)
	r := &nodeRelay{job: job, hash: hash, relay: relay, credential: credential, cancel: cancel, done: make(chan struct{})}
	go func() {
		r.err = relay.Run(life)
		close(r.done)
	}()
	return r
}

func (r *nodeRelay) live() error {
	select {
	case <-r.done:
		return errors.Join(fmt.Errorf("committed relay stopped; attempt cannot restart"), r.err)
	default:
		return nil
	}
}

func (a *NodeJobs) Authorize(ctx context.Context, x nodeagent.LocalExecution) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	v, err := a.current(ctx)
	if err != nil {
		return err
	}
	if err := a.options.Store.CheckExecutionTrust(ctx, a.options.Anchor, v, a.options.Clock()); err != nil {
		return err
	}
	current, err := a.options.Store.ActiveExecution(ctx)
	if err != nil {
		return err
	}
	if current == nil || current.Lease.ID != x.Lease.ID {
		return fmt.Errorf("execution lease changed during authorization")
	}
	if current.Transport != nil {
		if a.relay == nil || current.Job == nil || a.relay.job != current.Job.ID || a.relay.hash != current.Job.TransportHash {
			return fmt.Errorf("committed transport has no owning relay")
		}
		return a.relay.live()
	}
	return nil
}

// Cleanup is called only after durable hosting NoChild evidence. Join all TLS
// streams before invalidating the signer; release remains the node's decision.
func (a *NodeJobs) Cleanup(ctx context.Context, q nodeagent.CleanupRequest) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.relay != nil {
		r := a.relay
		if r.job != q.Job || r.hash != q.TransportHash {
			return fmt.Errorf("cleanup would affect another relay")
		}
		r.cancel()
		_ = r.relay.Close()
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-r.done:
		}
		if err := r.credential.Close(); err != nil {
			return err
		}
		a.relay = nil
	}
	return DestroyNodeWorkload(ctx, a.options.CredentialRoot, a.options.Anchor, q)
}
