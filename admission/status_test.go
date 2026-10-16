package admission

import (
	"context"
	"errors"
	"testing"
)

func TestSnapshotOwnsDurableInputs(t *testing.T) {
	x := setup(t)
	ctx := context.Background()
	a, err := x.s.Snapshot(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if a.Committed || a.Done || a.Plan != nil || a.Commit != nil || a.Aborting {
		t.Fatalf("new saga has effects: %+v", a)
	}
	a.Request.Selection.Selected[0].Endpoint = "changed.invalid:1"
	a.Specification.Config[0] = 'x'
	b, err := x.s.Snapshot(ctx)
	if err != nil || b.Request.Selection.Selected[0].Endpoint == "changed.invalid:1" || b.Specification.Config[0] != '{' {
		t.Fatalf("snapshot aliases journal: %+v %v", b, err)
	}
	for range 100 {
		done, err := x.s.Step(ctx, x.p)
		if err != nil {
			t.Fatal(err)
		}
		if done {
			break
		}
	}
	b, err = x.s.Snapshot(ctx)
	if err != nil || !b.Committed || !b.Done || b.Commit == nil || b.Plan == nil {
		t.Fatalf("missing committed snapshot: %+v %v", b, err)
	}
	if err := x.s.Cancel(ctx); err == nil {
		t.Fatal("committed saga allowed cancellation")
	}
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := x.s.Snapshot(canceled); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled snapshot: %v", err)
	}
}
