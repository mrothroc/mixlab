//go:build darwin || linux

package workerhost

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"

	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/workerhost/contract"
)

const guardianFile = "guardian-attempt.json"
const guardianLock = "guardian-attempt.lock"

type guardianClaim struct {
	Approval string `json:"approval"`
	Boot     string `json:"boot"`
}
type guardianRecord struct {
	Claim guardianClaim `json:"claim"`
	State string        `json:"state"`
}

func loadGuardian(dir statehome.Path) ([]byte, guardianRecord, error) {
	b, err := dir.ReadFileLimit(guardianFile, 2048)
	if err != nil {
		return nil, guardianRecord{}, err
	}
	var r guardianRecord
	if err := json.Unmarshal(b, &r); err != nil {
		return nil, r, err
	}
	canonical, _ := json.Marshal(r)
	if !bytes.Equal(b, canonical) || !contract.Hash(r.Claim.Approval) || !validBootIdentity(r.Claim.Boot) {
		return nil, r, fmt.Errorf("invalid guardian intent")
	}
	switch r.State {
	case "pending", "owned", "done", "fenced":
	default:
		return nil, r, fmt.Errorf("invalid guardian stage")
	}
	return b, r, nil
}

func transitionGuardian(ctx context.Context, dir statehome.Path, claim guardianClaim, from, to string) error {
	return dir.WithProcessLock(ctx, guardianLock, func() error {
		old, r, err := loadGuardian(dir)
		if err != nil {
			return err
		}
		if r.Claim == claim && r.State == "done" && from == "owned" && to == "done" {
			return nil // Reconciliation may already have consumed the reap receipt.
		}
		if r.Claim != claim || r.State != from {
			return fmt.Errorf("guardian attempt already claimed, fenced or changed")
		}
		r.State = to
		b, err := json.Marshal(r)
		if err != nil {
			return err
		}
		return dir.CompareAndSwap(guardianFile, old, b)
	})
}

func reconcileGuardian(ctx context.Context, dir statehome.Path, hash string, startedPID int) (noChild bool, err error) {
	err = dir.WithProcessLock(ctx, guardianLock, func() error {
		old, r, err := loadGuardian(dir)
		if err != nil {
			return err
		}
		if r.Claim.Approval != hash {
			return fmt.Errorf("guardian approval mismatch")
		}
		switch r.State {
		case "fenced":
			if startedPID != 0 {
				return fmt.Errorf("fenced helper has published worker start")
			}
			noChild = true
		case "pending":
			if startedPID != 0 {
				return fmt.Errorf("unclaimed helper has published worker start")
			}
			// Serialize against the helper's claim. Even an already-spawned
			// helper cannot launch a worker after this cancellation fence.
			r.State = "fenced"
			b, err := json.Marshal(r)
			if err != nil {
				return err
			}
			if err := dir.CompareAndSwap(guardianFile, old, b); err != nil {
				return err
			}
			noChild = true
		case "owned", "done":
			exit, receiptErr := readPhysicalExit(dir, physicalClaim{r.Claim.Approval, r.Claim.Boot})
			if receiptErr == nil {
				if startedPID != 0 && exit.PID != startedPID {
					return fmt.Errorf("guardian cleanup PID differs from published start")
				}
				// Recover the crash window between physical reap publication and
				// the guardian's final journal write, without trusting a saved PID.
				r.State = "done"
				b, err := json.Marshal(r)
				if err != nil {
					return err
				}
				if err := dir.CompareAndSwap(guardianFile, old, b); err != nil {
					return err
				}
				noChild = true
				return nil
			}
			if !errors.Is(receiptErr, os.ErrNotExist) {
				return receiptErr
			}
			boot, err := hostBootIdentity()
			if err != nil {
				return err
			}
			noChild = boot != r.Claim.Boot
		}
		return nil
	})
	return noChild, err
}
