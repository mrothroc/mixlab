package main

import (
	"context"
	"encoding/json"
	"fmt"
	"math"

	private "github.com/mrothroc/mixlab/diloco/learner"
	"github.com/mrothroc/mixlab/diloco/optimizer"
	"github.com/mrothroc/mixlab/learner"
	"github.com/mrothroc/mixlab/train"
)

// This adapter is in the training executable's composition root. Neither the
// numerical engine nor the management executable imports the learner service.
type learnerEngine struct{ session *train.LocalLearnerSession }
type learnerMetadata struct {
	Format   string `json:"format"`
	Contract string `json:"contract"`
	Attempt  uint64 `json:"attempt"`
	Tokens   uint64 `json:"tokens"`
	Examples uint64 `json:"examples"`
}

func (e *learnerEngine) Snapshot() (learner.Snapshot, error) {
	v, err := e.session.Snapshot()
	if err != nil {
		return learner.Snapshot{}, err
	}
	id := v.Sampler.Identity
	out := learner.Snapshot{Model: train.LocalLearnerWireTensors(v.Model), Precision: v.Precision, Replay: private.Replay{Version: "mixlab_learner_rng_v1", RunID: id.RunID, WorkerID: id.WorkerID, PartitionID: id.PartitionID, Seed: id.Seed, AttemptStep: v.Attempt}}
	for _, part := range []struct {
		v   any
		dst *[]byte
	}{
		{v.Optimizer, &out.Optimizer}, {v.Sampler, &out.Sampler},
		{v.Contract, &out.Schedule},
		{learnerMetadata{v.Format, v.Contract, v.Attempt, v.EffectiveTokens, v.EffectiveExamples}, &out.Metadata},
	} {
		*part.dst, err = json.Marshal(part.v)
		if err != nil {
			return learner.Snapshot{}, err
		}
	}
	return out, nil
}
func (e *learnerEngine) Restore(v learner.Snapshot) error {
	var meta learnerMetadata
	if err := json.Unmarshal(v.Metadata, &meta); err != nil {
		return err
	}
	var contract string
	if err := json.Unmarshal(v.Schedule, &contract); err != nil {
		return err
	}
	if contract != meta.Contract {
		return fmt.Errorf("private schedule contract mismatch")
	}
	model, err := train.LocalLearnerModelFromWire(v.Model)
	if err != nil {
		return err
	}
	s := train.LocalLearnerSnapshot{Format: meta.Format, Contract: meta.Contract, Attempt: meta.Attempt, EffectiveTokens: meta.Tokens, EffectiveExamples: meta.Examples, Precision: v.Precision, Model: model}
	if err := json.Unmarshal(v.Optimizer, &s.Optimizer); err != nil {
		return err
	}
	if err := json.Unmarshal(v.Sampler, &s.Sampler); err != nil {
		return err
	}
	id := s.Sampler.Identity
	want := private.Replay{Version: "mixlab_learner_rng_v1", RunID: id.RunID, WorkerID: id.WorkerID, PartitionID: id.PartitionID, Seed: id.Seed, AttemptStep: s.Attempt}
	if v.Replay != want {
		return fmt.Errorf("private RNG and sampler boundary differ")
	}
	return e.session.Restore(s)
}
func (e *learnerEngine) Install(v []optimizer.Tensor) error {
	model, err := train.LocalLearnerModelFromWire(v)
	if err != nil {
		return err
	}
	return e.session.InstallGlobal(model)
}
func (e *learnerEngine) Train(ctx context.Context, n uint64) (learner.Counts, error) {
	if n > math.MaxInt {
		return learner.Counts{}, fmt.Errorf("inner attempt count overflows")
	}
	v, err := e.session.Train(ctx, int(n))
	return learner.Counts{Attempted: v.Attempted, Committed: v.Committed, Skipped: v.Skipped, Tokens: v.EffectiveTokens, Examples: v.EffectiveExamples, Loss: v.Loss}, err
}
func (e *learnerEngine) Close() { e.session.Close() }
