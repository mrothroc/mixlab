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
const physicalExitFile = "physical-exit.json"

type physicalClaim struct {
	Approval string `json:"approval"`
	Boot     string `json:"boot"`
}

type physicalExit struct {
	Claim physicalClaim `json:"claim"`
	PID   int           `json:"pid"`
}

// LocalRunner composes an approved execution with the physical supervisor.
// It never guesses that a saved PID is dead. A hot restart without a durable
// exit receipt remains fenced; after a host reboot, the changed boot identity
// proves that all processes from the old attempt are gone.
type LocalRunner struct {
	supervisor *Supervisor
	directory  statehome.Path
	startup    time.Duration
	grace      time.Duration
	boot       func() (string, error)
}

func NewLocalRunner(s *Supervisor, dir statehome.Path, startup, grace time.Duration) (*LocalRunner, error) {
	if s == nil || dir.Kind() != statehome.Worker || startup <= 0 || startup > 10*time.Minute || grace <= 0 || grace > 30*time.Second {
		return nil, fmt.Errorf("approved supervisor, worker directory and bounded deadlines required")
	}
	if err := dir.Validate(); err != nil {
		return nil, err
	}
	return &LocalRunner{s, dir, startup, grace, hostBootIdentity}, nil
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
		result, runErr := r.supervisor.Run(ctx, LaunchPlan{Assignment: a.Assignment, Directory: r.directory, StartupTimeout: r.startup, ShutdownGrace: r.grace, Limits: a.Limits, Started: started})
		if errors.Is(runErr, ErrReconciliationRequired) {
			return runErr
		}
		// Supervisor returned only after fencing/reaping every owned child.
		// Persist that evidence before AttemptStore publishes its terminal state.
		b, err = json.Marshal(physicalExit{claim, result.PID})
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

// Receipts are immutable and published only after owned-process cleanup. They
// can be read without taking the physical lock, even if its owner died before
// publishing the higher-level guardian result.
func readPhysicalExit(dir statehome.Path, claim physicalClaim) (physicalExit, error) {
	b, err := dir.ReadFileLimit(physicalExitFile, 2048)
	if err != nil {
		return physicalExit{}, err
	}
	var exit physicalExit
	if err := json.Unmarshal(b, &exit); err != nil {
		return exit, err
	}
	canonical, _ := json.Marshal(exit)
	if !bytes.Equal(b, canonical) || exit.Claim != claim || exit.PID < 0 {
		return exit, fmt.Errorf("physical exit evidence changed")
	}
	return exit, nil
}

func validBootIdentity(s string) bool {
	if len(s) != 36 {
		return false
	}
	for i, c := range s {
		if i == 8 || i == 13 || i == 18 || i == 23 {
			if c != '-' {
				return false
			}
		} else if (c < '0' || c > '9') && (c < 'a' || c > 'f') {
			return false
		}
	}
	return true
}
