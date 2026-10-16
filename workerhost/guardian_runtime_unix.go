//go:build darwin || linux

package workerhost

import (
	"context"
	"fmt"
	"github.com/mrothroc/mixlab/artifact"
	"github.com/mrothroc/mixlab/statehome"
	"time"
)

// GuardianRuntime resolves already-approved executable identities and durable
// per-attempt directories. Remote inputs are only canonical job/attempt IDs.
type GuardianRuntime struct {
	store          *RuntimeStore
	supervisor     *Supervisor
	binary, build  string
	startup, grace time.Duration
}

func (r *GuardianRuntime) Output(ctx context.Context, job, attempt string) (statehome.Path, artifact.Ref, error) {
	return r.store.Output(ctx, job, attempt)
}

func (r *GuardianRuntime) LearnerPrivate(ctx context.Context, run, worker string) (statehome.Path, error) {
	return r.store.LearnerPrivate(ctx, run, worker)
}

func NewGuardianRuntime(store *RuntimeStore, workerBinary, workerBuild, guardianBinary, guardianBuild string, startup, grace time.Duration) (*GuardianRuntime, error) {
	if store == nil {
		return nil, fmt.Errorf("initialized runtime store required")
	}
	s, err := New(workerBinary, workerBuild)
	if err != nil {
		return nil, err
	}
	// Validate helper identity and deadlines without allocating a job directory.
	l, err := NewLocalRunner(s, store.root, startup, grace)
	if err != nil {
		return nil, err
	}
	if _, err := NewGuardianRunner(l, guardianBinary, guardianBuild); err != nil {
		return nil, err
	}
	return &GuardianRuntime{store, s, guardianBinary, guardianBuild, startup, grace}, nil
}

func (r *GuardianRuntime) Open(ctx context.Context, job, attempt string) (*AttemptStore, Runner, error) {
	return r.open(ctx, job, attempt, nil, statehome.Path{})
}

// OpenLearner adds only an opaque relay and an administrator-owned private
// directory. Remote credentials and network policy never enter hosting.
func (r *GuardianRuntime) OpenLearner(ctx context.Context, job, attempt string, relay Relay, private statehome.Path) (*AttemptStore, Runner, error) {
	if relay == nil || private.Kind() != statehome.Worker {
		return nil, nil, fmt.Errorf("local learner relay and private directory required")
	}
	return r.open(ctx, job, attempt, relay, private)
}

func (r *GuardianRuntime) open(ctx context.Context, job, attempt string, relay Relay, private statehome.Path) (*AttemptStore, Runner, error) {
	p, err := r.store.Attempt(ctx, job, attempt)
	if err != nil {
		return nil, nil, err
	}
	host, err := NewAttemptStore(p)
	if err != nil {
		return nil, nil, err
	}
	l, err := NewLocalRunner(r.supervisor, p, r.startup, r.grace)
	if err != nil {
		return nil, nil, err
	}
	l.LearnerRelay, l.LearnerPrivateDirectory = relay, private
	guardian, err := NewGuardianRunner(l, r.binary, r.build)
	if err != nil {
		return nil, nil, err
	}
	return host, guardian, nil
}
