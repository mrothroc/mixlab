package nodejob

import (
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/trust"
)

// LearnerRecovery is node-owned authorization after confirmed old-child exit.
// Its admission deadline never extends the original allocation runtime.
type LearnerRecovery struct {
	Attempt     run.AttemptBinding `json:"attempt"`
	Coordinator run.Coordinator    `json:"coordinator"`
	Created     int64              `json:"created"`
	Expires     int64              `json:"expires"`
}

func (r LearnerRecovery) Digest() (string, error) {
	r.Attempt.NodeAuthorizationHash = ""
	return run.Digest(r)
}
func (r LearnerRecovery) Validate() error {
	if err := r.Attempt.Validate(); err != nil {
		return err
	}
	if err := r.Coordinator.Validate(); err != nil {
		return err
	}
	if r.Created <= 0 || r.Expires <= r.Created || r.Expires-r.Created > 600 {
		return fmt.Errorf("bounded retry authorization required")
	}
	h, err := r.Digest()
	if err != nil {
		return err
	}
	if h != r.Attempt.NodeAuthorizationHash {
		return fmt.Errorf("retry authorization hash mismatch")
	}
	return nil
}
func (r LearnerRecovery) SigningRequest() (trust.SignRequest, error) {
	if err := r.Validate(); err != nil {
		return trust.SignRequest{}, err
	}
	return trust.SignRequest{Version: trust.ProofVersion, Purpose: trust.LearnerRecovery, Digest: r.Attempt.NodeAuthorizationHash, Context: r.Attempt.Binding.RunID, Audience: r.Coordinator.Audience}, nil
}

type SignedLearnerRecovery struct {
	Body  LearnerRecovery   `json:"body"`
	Proof trust.SignedProof `json:"proof"`
}

func (r SignedLearnerRecovery) Verify(a trust.Anchor, v trust.VerifiedSnapshot, now time.Time) error {
	q, err := r.Body.SigningRequest()
	if err != nil {
		return err
	}
	if r.Body.Coordinator.ClusterID != a.Cluster() || r.Proof.Evidence.Role != trust.Node || r.Proof.Evidence.Principal != r.Body.Attempt.Allocation.NodeID || now.Unix() < r.Body.Created || now.Unix() >= r.Body.Expires {
		return fmt.Errorf("retry node, cluster or deadline mismatch")
	}
	_, err = trust.VerifyProof(a, v, r.Proof, q, now)
	return err
}
