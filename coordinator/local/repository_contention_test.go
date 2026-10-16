package local

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/mrothroc/mixlab/coordinator"
)

func TestRepositoryDistinguishesCASContention(t *testing.T) {
	r, err := Open(protected(t, t.TempDir()))
	if err != nil {
		t.Fatal(err)
	}
	s := fixtureState(t)
	id := s.Plan.Cohort.RunID
	ctx := context.Background()
	if err := r.Save(ctx, id, 0, coordinator.Change{Next: s}); err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct {
		name       string
		revision   uint64
		next       uint64
		runID      string
		concurrent bool
	}{
		{"lost-create-CAS", 0, 1, id, true},
		{"future-revision", 2, 3, id, false},
		{"invalid-next-revision", 0, 2, id, false},
		{"invalid-run-scope", 0, 1, "other-run", false},
		{"overflow", ^uint64(0), 0, id, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			next := s
			next.Revision = tt.next
			next.Plan.Cohort.RunID = tt.runID
			err := r.Save(ctx, id, tt.revision, coordinator.Change{Next: next})
			if !errors.Is(err, coordinator.ErrConflict) || errors.Is(err, coordinator.ErrConcurrent) != tt.concurrent {
				t.Fatalf("wrong classification: %v, want concurrent=%v", err, tt.concurrent)
			}
		})
	}
	// A real existing-head race has the same classification as competing creates.
	next := s
	next.Revision++
	if err := r.Save(ctx, id, s.Revision, coordinator.Change{Next: next}); err != nil {
		t.Fatal(err)
	}
	err = r.Save(ctx, id, s.Revision, coordinator.Change{Next: next})
	if !errors.Is(err, coordinator.ErrConcurrent) || !errors.Is(fmt.Errorf("adapter: %w", err), coordinator.ErrConflict) {
		t.Fatalf("lost update CAS did not retain conflict compatibility: %v", err)
	}
	if errors.Is(coordinator.ErrConflict, coordinator.ErrConcurrent) {
		t.Fatal("semantic conflicts must not match contention")
	}
	got, err := r.Load(ctx, id)
	if err != nil || got.Revision != 2 {
		t.Fatalf("rejected saves changed durable state: revision=%d err=%v", got.Revision, err)
	}
}
