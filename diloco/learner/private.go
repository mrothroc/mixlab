// Package learner owns pure private-state and update contracts, not the learner
// application service. Hosting credentials/attempts never enter replay identity.
package learner

import (
	"fmt"
	"github.com/mrothroc/mixlab/artifact"
	"github.com/mrothroc/mixlab/diloco/internal/wire"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/precision"
)

func Decode[T interface{ Validate() error }](b []byte, dst T) error { return wire.Decode(b, dst) }

// Replay is restored together with the sampler. AttemptStep means a logical
// optimizer attempt, not a process launch. Initial weights use the common seed;
// private stochastic streams additionally use this worker/partition identity.
type Replay struct {
	Version          string `json:"version"`
	RunID            string `json:"run_id"`
	WorkerID         string `json:"worker_id"`
	PartitionID      string `json:"partition_id"`
	Seed             int64  `json:"seed"`
	AttemptStep      uint64 `json:"attempt_step"`
	Microstep        uint64 `json:"microstep"`
	OperationOrdinal uint64 `json:"operation_ordinal"`
}

func (r Replay) Validate() error {
	if r.Version != "mixlab_learner_rng_v1" || !wire.Text(r.RunID) || !wire.Text(r.WorkerID) || !wire.Text(r.PartitionID) {
		return fmt.Errorf("invalid replay identity")
	}
	return nil
}
func (r Replay) Digest() (string, error) {
	if e := r.Validate(); e != nil {
		return "", e
	}
	return wire.Digest(r)
}

// PrivateState references complete engine snapshots. Metadata includes direct
// sampler cursor, scheduler, valid-target/record state and counters; no component
// may be reconstructed by consuming additional training data on restore.
type PrivateState struct {
	Format            string           `json:"format"`
	Assignment        run.Assignment   `json:"assignment"`
	AssignmentHash    string           `json:"assignment_hash"`
	UpdateID          string           `json:"update_id"`
	Stage             string           `json:"stage"`
	Model             artifact.Ref     `json:"model"`
	Optimizer         artifact.Ref     `json:"optimizer"`
	Sampler           artifact.Ref     `json:"sampler"`
	Schedule          artifact.Ref     `json:"schedule"`
	TrainingMetadata  artifact.Ref     `json:"training_metadata"`
	Replay            Replay           `json:"replay"`
	Precision         precision.Policy `json:"precision"`
	Attempted         uint64           `json:"attempted_steps"`
	Committed         uint64           `json:"committed_steps"`
	Skipped           uint64           `json:"skipped_steps"`
	EffectiveTokens   uint64           `json:"effective_tokens"`
	EffectiveExamples uint64           `json:"effective_examples"`
}

func (p PrivateState) Validate() error {
	if e := p.Assignment.Validate(); e != nil {
		return e
	}
	if e := p.Replay.Validate(); e != nil {
		return e
	}
	h, e := wire.Digest(p.Assignment)
	if e != nil {
		return e
	}
	if p.Format != "mixlab_learner_private_v1" || h != p.AssignmentHash || !wire.Text(p.UpdateID) || (p.Stage != "start" && p.Stage != "completed") || p.Skipped != 0 || p.Attempted != p.Committed {
		return fmt.Errorf("invalid private checkpoint binding/counters")
	}
	for _, ref := range []artifact.Ref{p.Model, p.Optimizer, p.Sampler, p.Schedule, p.TrainingMetadata} {
		if e := ref.Validate(); e != nil {
			return e
		}
	}
	if p.Stage == "start" && (p.Attempted != 0 || p.EffectiveTokens != 0 || p.EffectiveExamples != 0) {
		return fmt.Errorf("start snapshot has round work")
	}
	if p.Stage == "completed" && (p.Committed != p.Assignment.InnerSteps || p.EffectiveTokens == 0 || p.EffectiveExamples == 0) {
		return fmt.Errorf("incomplete private snapshot")
	}
	h, e = p.Precision.Digest()
	if e != nil {
		return e
	}
	if h != p.Assignment.PrecisionHash || p.Precision.Backend != "metal" {
		return fmt.Errorf("private precision differs from assignment")
	}
	if p.Replay.RunID != p.Assignment.Binding.RunID || p.Replay.WorkerID != p.Assignment.WorkerID || p.Replay.PartitionID != p.Assignment.PartitionID || p.Replay.Microstep != 0 || p.Replay.OperationOrdinal != 0 {
		return fmt.Errorf("private replay identity is not at an attempt boundary")
	}
	if p.Assignment.Round > (^uint64(0)-p.Committed)/p.Assignment.InnerSteps || p.Replay.AttemptStep != p.Assignment.Round*p.Assignment.InnerSteps+p.Committed {
		return fmt.Errorf("logical RNG counter mismatch")
	}
	return nil
}

// Completion is published last after fsync of the named immutable records. It
// joins hashes without introducing a private-state <-> update hash cycle.
type Completion struct {
	Format         string `json:"format"`
	AssignmentHash string `json:"assignment_hash"`
	UpdateID       string `json:"update_id"`
	PrivateHash    string `json:"private_hash"`
	UpdateHash     string `json:"update_hash"`
	SHA256         string `json:"sha256"`
}

func (c Completion) Digest() (string, error) { c.SHA256 = ""; return wire.Digest(c) }
func (c Completion) Validate() error {
	if c.Format != "mixlab_learner_completion_v1" || !wire.Text(c.UpdateID) || !wire.Hash(c.AssignmentHash) || !wire.Hash(c.PrivateHash) || !wire.Hash(c.UpdateHash) {
		return fmt.Errorf("invalid completion marker")
	}
	h, e := c.Digest()
	if e != nil {
		return e
	}
	if h != c.SHA256 {
		return fmt.Errorf("completion checksum mismatch")
	}
	return nil
}

func (c Completion) ValidateFor(p PrivateState, u Update) error {
	if e := c.Validate(); e != nil {
		return e
	}
	if e := p.Validate(); e != nil {
		return e
	}
	if e := u.ValidateFor(p.Assignment); e != nil {
		return e
	}
	ph, e := wire.Digest(p)
	if e != nil {
		return e
	}
	uh, e := wire.Digest(u)
	if e != nil {
		return e
	}
	if p.Stage != "completed" || c.PrivateHash != ph || c.UpdateHash != uh || c.AssignmentHash != p.AssignmentHash || c.UpdateID != p.UpdateID || u.ID != p.UpdateID || u.PrivateHash != ph || u.EffectiveTokens != p.EffectiveTokens || u.EffectiveExamples != p.EffectiveExamples {
		return fmt.Errorf("completion files do not bind the same finished round")
	}
	return nil
}
