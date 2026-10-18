package nodeagent

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"time"

	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/trust"
)

func decodeLearnerPreparation(b []byte) (learnerPreparation, error) {
	var c learnerPreparation
	if err := json.Unmarshal(b, &c); err != nil {
		return c, err
	}
	canonical, err := json.Marshal(c)
	if err != nil || !bytes.Equal(b, canonical) {
		return c, fmt.Errorf("noncanonical learner preparation")
	}
	return c, nil
}

// AcceptLearnerPreparation permits historical evidence only for this store's
// exact durable acceptance. The caller supplies a freshly authenticated actor;
// neither a request nor a peer can supply the receipt used for continuation.
func (s *Store) AcceptLearnerPreparation(ctx context.Context, anchor trust.Anchor, view trust.VerifiedSnapshot, actor trust.AuthenticatedPrincipal, signed nodejob.SignedLearner, now time.Time) (out nodejob.AcceptedLearner, err error) {
	if anchor.Cluster() != s.cluster {
		return out, fmt.Errorf("learner trust belongs to another cluster")
	}
	name, err := learnerPreparationFile(signed.Manifest.Slot.Allocation.JobID)
	if err != nil {
		return out, err
	}
	err = s.path.WithProcessLock(ctx, nodeLock, func() error {
		if err := ctx.Err(); err != nil {
			return err
		}
		var receipt *trust.AcceptedProof
		var cache *learnerPreparation
		b, err := s.path.ReadFileLimit(name, 64<<10)
		switch {
		case err == nil:
			c, err := decodeLearnerPreparation(b)
			if err != nil {
				return err
			}
			if c.SignedHash != digest(signed) {
				return fmt.Errorf("learner preparation manifest changed")
			}
			cache = &c
			receipt = &trust.AcceptedProof{Digest: c.ProofDigest, Generation: c.ProofGeneration}
		case !errors.Is(err, os.ErrNotExist):
			return err
		default:
			_, j, err := s.loadLearner(signed.Manifest.Slot.Allocation.JobID)
			if err == nil {
				if digest(j.Signed) != digest(signed) {
					return fmt.Errorf("learner preparation manifest changed")
				}
				receipt = &j.Accepted
			} else if !errors.Is(err, os.ErrNotExist) {
				return err
			}
		}
		if receipt == nil {
			out, err = nodejob.AcceptLearner(anchor, view, actor, s.node, signed, now)
		} else {
			out, err = nodejob.ContinueLearner(anchor, view, actor, s.node, signed, executionCommit{*receipt}, now)
		}
		if err != nil {
			return err
		}
		frozen, proof, err := out.Value()
		if err != nil {
			return err
		}
		in := learnerPreparationInput{signed: frozen, proof: proof}
		state, err := s.guardLearnerPreparation(actor, in, now)
		if err != nil {
			return err
		}
		if cache != nil {
			state.cache = *cache
			return state.validate(in)
		}
		return nil
	})
	if err != nil {
		return nodejob.AcceptedLearner{}, err
	}
	return out, nil
}
