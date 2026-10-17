package nodeagent

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"

	"github.com/mrothroc/mixlab/workerhost/contract"
)

const learnerOutcomeBytes = 32 << 10

func learnerOutcomeFile(job, attempt string, version uint64) (string, error) {
	if !identifier(job) || !identifier(attempt) || version < 1 || version > 2 {
		return "", fmt.Errorf("invalid learner outcome identity")
	}
	return fmt.Sprintf("learner-outcome-%s-%s-%d.json", job, attempt, version), nil
}

// Diagnostic receipts never authorize lifecycle transitions. Publish before
// advancing the journal so a crash cannot erase evidence during a retry. Each
// immutable receipt is bounded separately from the hot-path learner journal.
func (s *Store) retainLearnerOutcome(o *contract.Outcome) error {
	if o == nil {
		return nil
	}
	if err := o.Validate(); err != nil {
		return err
	}
	name, err := learnerOutcomeFile(o.JobID, o.AttemptID, o.Version)
	if err != nil {
		return err
	}
	b, err := json.Marshal(o)
	if err != nil {
		return err
	}
	if len(b) > learnerOutcomeBytes {
		return fmt.Errorf("learner outcome receipt too large")
	}
	old, err := s.path.ReadFileLimit(name, learnerOutcomeBytes)
	if err == nil {
		if !bytes.Equal(old, b) {
			return fmt.Errorf("learner outcome receipt conflict")
		}
		return nil
	}
	if !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return s.path.CompareAndSwap(name, nil, b)
}

// LearnerOutcomes is a local diagnostic reader, not a remote status or recovery
// authorization input. Only known admitted attempts are inspected. Older
// journals may have no receipts; their latest available outcome is included.
func (s *Store) LearnerOutcomes(ctx context.Context, id string) (out []contract.Outcome, err error) {
	var j LearnerJob
	err = s.path.WithProcessLock(ctx, nodeLock, func() error {
		var e error
		_, j, e = s.loadLearner(id)
		return e
	})
	if err != nil {
		return nil, err
	}
	// Receipts are immutable; inspect the captured attempt list without holding
	// the process lock through historical I/O or delaying live learner RPCs.
	err = func() error {
		if j.Assignment == nil {
			return nil
		}
		attempts := []string{j.Signed.Manifest.Attempt}
		privateHashes := []string{""}
		for _, r := range j.History {
			attempts = append(attempts, r.Attempt.AttemptID)
			privateHashes = append(privateHashes, r.Attempt.PrivateHash)
		}
		for i, attempt := range attempts {
			var previous *contract.Outcome
			for version := uint64(1); version <= 2; version++ {
				if err := ctx.Err(); err != nil {
					return err
				}
				name, err := learnerOutcomeFile(id, attempt, version)
				if err != nil {
					return err
				}
				b, err := s.path.ReadFileLimit(name, learnerOutcomeBytes)
				var o contract.Outcome
				switch {
				case errors.Is(err, os.ErrNotExist):
					if j.Outcome == nil || j.Outcome.AttemptID != attempt || j.Outcome.Version != version {
						continue
					}
					o = *j.Outcome
				case err != nil:
					return err
				default:
					if err := json.Unmarshal(b, &o); err != nil {
						return err
					}
					canonical, _ := json.Marshal(o)
					if !bytes.Equal(canonical, b) {
						return fmt.Errorf("noncanonical learner outcome receipt")
					}
				}
				if err := o.Validate(); err != nil {
					return err
				}
				old := j
				a := *j.Assignment
				a.AttemptID, a.ExpectedPrivateHash = attempt, privateHashes[i]
				old.Assignment = &a
				approval, err := learnerApproval(old)
				if err != nil {
					return err
				}
				if o.JobID != id || o.AttemptID != attempt || o.Version != version || o.ManifestHash != approval.ManifestHash || o.ApprovalHash != digest(*approval) || (previous != nil && previous.Terminal()) {
					return fmt.Errorf("learner outcome history binding or ordering mismatch")
				}
				if j.Outcome != nil && j.Outcome.AttemptID == attempt && j.Outcome.Version == version && *j.Outcome != o {
					return fmt.Errorf("learner latest outcome differs from receipt")
				}
				out = append(out, o)
				previous = &o
			}
		}
		return nil
	}()
	return
}
