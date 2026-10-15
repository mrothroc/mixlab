package data

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
)

// LearnerIdentity has no process attempt, credential generation or clock.
// Index is the immutable cohort position, not an MLX collective rank.
type LearnerIdentity struct {
	RunID       string `json:"run_id"`
	WorkerID    string `json:"worker_id"`
	PartitionID string `json:"partition_id"`
	Index       int    `json:"index"`
	Count       int    `json:"count"`
	Seed        int64  `json:"seed"`
}

func (i LearnerIdentity) Validate() error {
	if i.RunID == "" || i.WorkerID == "" || i.PartitionID == "" || len(i.RunID) > 256 || len(i.WorkerID) > 256 || len(i.PartitionID) > 256 || i.Count < 1 || i.Count > 64 || i.Index < 0 || i.Index >= i.Count {
		return fmt.Errorf("invalid immutable learner identity")
	}
	return nil
}

func (i LearnerIdentity) RandomSeed() uint64 {
	b, _ := json.Marshal(i)
	h := sha256.Sum256(append([]byte("mixlab-learner-rng-v1\x00"), b...))
	return binary.LittleEndian.Uint64(h[:8])
}

type LearnerSamplerState struct {
	Identity LearnerIdentity         `json:"identity"`
	Counter  DistributedSamplerState `json:"counter"`
}

type LearnerLoader struct {
	identity LearnerIdentity
	loader   *DistributedLoader
}

func NewLearnerLoader(pattern string, identity LearnerIdentity, seqLen, chunkTokens, vocab int) (*LearnerLoader, error) {
	if e := identity.Validate(); e != nil {
		return nil, e
	}
	// All workers share the permutation seed, but consume disjoint strided
	// ordinals. Their stochastic model streams use RandomSeed instead.
	l, e := newPartitionedLoader(pattern, identity.Seed, identity.Count, identity.Index, seqLen, chunkTokens, vocab, identity.PartitionID)
	if e != nil {
		return nil, e
	}
	return &LearnerLoader{identity, l}, nil
}

func (l *LearnerLoader) State() LearnerSamplerState {
	return LearnerSamplerState{l.identity, l.loader.State()}
}
func (l *LearnerLoader) Restore(s LearnerSamplerState) error {
	if s.Identity != l.identity {
		return fmt.Errorf("learner sampler identity mismatch")
	}
	return l.loader.Restore(s.Counter)
}
func (l *LearnerLoader) NextBatch(tokens int) (Batch, error) { return l.loader.NextBatch(tokens) }

// ValidateConsumedRows binds a boundary cursor to logical work without replaying
// samples. Record mode has one row per chunk; stream chunks may contain many.
func (l *LearnerLoader) ValidateConsumedRows(s LearnerSamplerState, rows uint64) error {
	perChunk := uint64(l.loader.state.ChunkTokens / l.loader.seqLen)
	chunks := rows / perChunk
	perEpoch := uint64(l.loader.usable / l.identity.Count)
	if s.Identity != l.identity || s.Counter.Epoch != chunks/perEpoch || s.Counter.GlobalCursor != (chunks%perEpoch)*uint64(l.identity.Count) || s.Counter.ChunkOffset != int(rows%perChunk)*l.loader.seqLen {
		return fmt.Errorf("learner cursor differs from logical consumed rows")
	}
	return nil
}
