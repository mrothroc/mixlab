package data

import (
	"fmt"
	"math/rand"
	"slices"
)

type GridSamplerState struct {
	Epoch  int   `json:"epoch"`
	Cursor int   `json:"cursor"`
	Order  []int `json:"order"`
}

// GridSampler preserves the original grid shuffle stream; augmentation never
// consumes it. Restoring replays permutations, not data reads or training steps.
type GridSampler struct {
	rng   *rand.Rand
	state GridSamplerState
}

func NewGridSampler(records int, seed int64) (*GridSampler, error) {
	if records <= 0 {
		return nil, fmt.Errorf("grid sampler requires records")
	}
	s := &GridSampler{rng: rand.New(rand.NewSource(seed))}
	s.state.Order = s.rng.Perm(records)
	return s, nil
}

func (s *GridSampler) Snapshot() GridSamplerState {
	state := s.state
	state.Order = slices.Clone(state.Order)
	return state
}

func (s *GridSampler) Restore(state GridSamplerState, step, batchSize int) error {
	n := len(s.state.Order)
	if step <= 0 || batchSize <= 0 || len(state.Order) != n {
		return fmt.Errorf("invalid grid sampler checkpoint")
	}
	batches := 1 + (n-1)/batchSize
	wantEpoch := (step - 1) / batches
	wantCursor := min(n, (1+(step-1)%batches)*batchSize)
	if state.Epoch != wantEpoch || state.Cursor != wantCursor {
		return fmt.Errorf("grid sampler epoch/cursor do not match checkpoint step")
	}
	for s.state.Epoch < state.Epoch {
		s.state.Order = s.rng.Perm(n)
		s.state.Epoch++
	}
	if !slices.Equal(s.state.Order, state.Order) {
		return fmt.Errorf("grid sampler order does not match seed/epoch")
	}
	s.state.Cursor = state.Cursor
	return nil
}

// Next returns a read-only index slice, epoch, and first occurrence in that epoch.
func (s *GridSampler) Next(batchSize int) ([]int, int, int, error) {
	if batchSize <= 0 {
		return nil, 0, 0, fmt.Errorf("invalid grid batch size")
	}
	if s.state.Cursor == len(s.state.Order) {
		s.state.Order = s.rng.Perm(len(s.state.Order))
		s.state.Cursor = 0
		s.state.Epoch++
	}
	start := s.state.Cursor
	s.state.Cursor = min(start+batchSize, len(s.state.Order))
	return s.state.Order[start:s.state.Cursor], s.state.Epoch, start, nil
}
