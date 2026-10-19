//go:build darwin || linux

package workerhost

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"time"

	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/workerhost/contract"
)

const physicalOwnerLock = "physical-owner.lock"
const physicalClaimFile = "physical-attempt.json"

// LocalRunner composes an approved execution with the physical supervisor.
// It never guesses that a saved PID is dead. A hot restart without a durable
// exit receipt remains fenced; after a host reboot, the changed boot identity
// proves that all processes from the old attempt are gone.
type LocalRunner struct {
	supervisor              *Supervisor
	directory               statehome.Path
	startup                 time.Duration
	grace                   time.Duration
	boot                    func() (string, error)
	LearnerRelay            Relay
	LearnerPrivateDirectory statehome.Path
}

func NewLocalRunner(s *Supervisor, dir statehome.Path, startup, grace time.Duration) (*LocalRunner, error) {
	if s == nil || dir.Kind() != statehome.Worker || startup <= 0 || startup > 10*time.Minute || grace <= 0 || grace > 30*time.Second {
		return nil, fmt.Errorf("approved supervisor, worker directory and bounded deadlines required")
	}
	if err := dir.Validate(); err != nil {
		return nil, err
	}
	return &LocalRunner{supervisor: s, directory: dir, startup: startup, grace: grace, boot: hostBootIdentity}, nil
}

func (r *LocalRunner) Run(ctx context.Context, a contract.Approved, started func(int) error) error {
	a, hash, err := freezeApproval(a)
	if err != nil {
		return err
	}
	if started == nil {
		return fmt.Errorf("durable started callback required")
	}
	return r.directory.WithProcessLock(ctx, physicalOwnerLock, func() error {
		boot, err := r.boot()
		if err != nil {
			return err
		}
		claim := physicalClaim{hash, boot}
		b, err := json.Marshal(claim)
		if err != nil {
			return err
		}
		// Permanent physical intent also fences direct adapter retries. Losing
		// an exit receipt cannot turn this into permission to launch again.
		if err := r.directory.CompareAndSwap(physicalClaimFile, nil, b); err != nil {
			return errors.Join(ErrReconciliationRequired, err)
		}
		result, runErr := r.supervisor.Run(ctx, LaunchPlan{Assignment: a.Assignment, Learner: a.Learner, Relay: r.LearnerRelay, LearnerPrivateDirectory: r.LearnerPrivateDirectory, Directory: r.directory, StartupTimeout: r.startup, ShutdownGrace: r.grace, Limits: a.Limits, Started: started})
		// Preflight can return before the supervisor installs its own observer.
		// Freeze the outer runtime before exit persistence can cross its deadline.
		runErr = withRuntimeDiagnostic(ctx, runErr)
		if errors.Is(runErr, ErrReconciliationRequired) {
			return runErr
		}
		// Supervisor returned only after fencing/reaping every owned child.
		// Persist that evidence before AttemptStore publishes its terminal state.
		b, err = json.Marshal(physicalExit{Claim: claim, PID: result.PID, Exit: result.Exit})
		if err == nil {
			err = r.directory.CompareAndSwap(physicalExitFile, nil, b)
		}
		if err != nil {
			return errors.Join(runErr, err, ErrReconciliationRequired)
		}
		return runErr
	})
}

func (r *LocalRunner) Reconcile(ctx context.Context, a Attempt) (noChild bool, err error) {
	if err := a.validate(); err != nil {
		return false, err
	}
	err = r.directory.WithProcessLock(ctx, physicalOwnerLock, func() error {
		b, err := r.directory.ReadFileLimit(physicalClaimFile, 1024)
		if err != nil {
			return errors.Join(ErrReconciliationRequired, err)
		}
		var claim physicalClaim
		if err := json.Unmarshal(b, &claim); err != nil {
			return err
		}
		canonical, _ := json.Marshal(claim)
		if !bytes.Equal(b, canonical) || claim.Approval != a.Digest || !validBootIdentity(claim.Boot) {
			return fmt.Errorf("physical ownership evidence changed")
		}
		exit, err := readPhysicalExit(r.directory, claim)
		if err == nil {
			if a.Started != nil && exit.PID != a.Started.PID {
				return fmt.Errorf("physical exit PID evidence changed")
			}
			noChild = true
			return nil
		}
		if !errors.Is(err, os.ErrNotExist) {
			return err
		}
		boot, err := r.boot()
		if err != nil {
			return err
		}
		noChild = boot != claim.Boot
		return nil
	})
	return noChild, err
}
