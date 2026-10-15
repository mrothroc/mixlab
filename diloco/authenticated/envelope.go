// Package authenticated defines coordinator-side signed envelopes. It must not
// be linked into the training executable. Structural validation does not verify
// trust: adapters must use the existing trust verifier and current revocation
// evidence before returning the semantic body across the authenticated IPC port.
package authenticated

import (
	"crypto/ed25519"
	"fmt"
	"github.com/mrothroc/mixlab/diloco/internal/wire"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/trust"
)

type Commit struct {
	Body  run.Commit        `json:"body"`
	Proof trust.SignedProof `json:"proof"`
}
type Plan struct {
	Body  run.Plan          `json:"body"`
	Proof trust.SignedProof `json:"proof"`
}
type Assignment struct {
	Body  run.Assignment    `json:"body"`
	Proof trust.SignedProof `json:"proof"`
}

type Recovery struct {
	Body        run.AttemptBinding `json:"body"`
	Coordinator run.Coordinator    `json:"coordinator"`
	Proof       trust.SignedProof  `json:"proof"`
}

func (r Recovery) Validate() error {
	if err := r.Body.Validate(); err != nil {
		return err
	}
	if err := r.Coordinator.Validate(); err != nil {
		return err
	}
	h, err := run.Digest(r.Body)
	if err != nil {
		return err
	}
	return binding(r.Proof, trust.RecoveryGrant, h, r.Body.Binding.RunID, r.Coordinator)
}

func (p Plan) Validate() error {
	if e := p.Body.Validate(); e != nil {
		return e
	}
	return binding(p.Proof, trust.RunPlan, p.Body.SHA256, p.Body.Cohort.RunID, p.Body.Coordinator)
}
func (a Assignment) Validate() error {
	if e := a.Body.Validate(); e != nil {
		return e
	}
	h, e := run.Digest(a.Body)
	if e != nil {
		return e
	}
	return binding(a.Proof, trust.RoundGrant, h, a.Body.Binding.RunID, a.Body.Coordinator)
}

type Grant struct {
	Body  run.Grant         `json:"body"`
	Proof trust.SignedProof `json:"proof"`
}

func (c Commit) Validate() error {
	if e := c.Body.Validate(); e != nil {
		return e
	}
	return binding(c.Proof, trust.RunCommit, c.Body.Binding.CommitHash, c.Body.Binding.RunID, c.Body.Coordinator)
}
func (g Grant) Validate() error {
	if e := g.Body.Validate(); e != nil {
		return e
	}
	return binding(g.Proof, trust.ArtifactGrant, g.Body.SHA256, g.Body.Binding.RunID, g.Body.Coordinator)
}
func Decode[T interface{ Validate() error }](b []byte, dst T) error { return wire.Decode(b, dst) }
func binding(p trust.SignedProof, purpose trust.Purpose, hash, context string, c run.Coordinator) error {
	if p.Request != (trust.SignRequest{Version: trust.ProofVersion, Purpose: purpose, Digest: hash, Context: context, Audience: c.Audience}) || p.Evidence.Cluster != c.ClusterID || p.Evidence.Principal != c.PrincipalID || p.Evidence.Role != trust.Coordinator || len(p.Signature) != ed25519.SignatureSize {
		return fmt.Errorf("signed record coordinator/purpose binding mismatch")
	}
	return nil
}
