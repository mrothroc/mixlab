package admission

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/recruitment"
)

func TestCompensationTimeoutRotatesAndSurvivesRestart(t *testing.T) {
	for _, lostACK := range []bool{false, true} {
		name := "no_effect"
		if lostACK {
			name = "lost_ack"
		}
		t.Run(name, func(t *testing.T) {
			x := setup(t)
			ctx := context.Background()
			for len(x.leases) < 2 {
				if _, err := x.s.Step(ctx, x.p); err != nil {
					t.Fatal(err)
				}
			}
			if err := x.s.Cancel(ctx); err != nil {
				t.Fatal(err)
			}
			if _, err := x.s.Step(ctx, x.p); err != nil { // durable coordinator fence
				t.Fatal(err)
			}
			v, err := x.s.Snapshot(ctx)
			if err != nil {
				t.Fatal(err)
			}
			x.now = v.Deadline + 1 // compensation is not capped by forward admission
			first := v.Request.Selection.Selected[0].Capabilities.Node
			calls := make(map[string]int)
			original := x.p.AbortReservation
			x.p.AbortReservation = func(ctx context.Context, c recruitment.Candidate, q nodeagent.Reserve) error {
				calls[c.Capabilities.Node]++
				if q != v.Nodes[0].Reserve && q != v.Nodes[1].Reserve {
					t.Error("abort retry changed immutable reservation")
				}
				if c.Capabilities.Node == first && calls[first] <= 2 {
					if lostACK {
						if err := original(ctx, c, q); err != nil {
							return err
						}
					}
					<-ctx.Done()
					return ctx.Err()
				}
				return original(ctx, c, q)
			}
			for attempt := range 5 {
				x.s, err = Open(x.s.path)
				if err != nil {
					t.Fatal(err)
				}
				bounded, cancel := context.WithTimeout(ctx, 100*time.Millisecond)
				_, err = x.s.Step(bounded, x.p)
				cancel()
				if attempt == 0 || attempt == 2 {
					if !errors.Is(err, context.DeadlineExceeded) {
						t.Fatalf("attempt %d: %v", attempt, err)
					}
				} else if err != nil {
					t.Fatal(err)
				}
				v, err = x.s.Snapshot(ctx)
				if err != nil {
					t.Fatal(err)
				}
				if attempt < 3 && (v.Nodes[0].Released || v.Done) {
					t.Fatal("timeout treated as confirmed cleanup")
				}
				if attempt >= 1 && !v.Nodes[1].Released {
					t.Fatal("slow first node starved later compensation")
				}
			}
			if !v.Done || len(x.leases) != 0 || calls[first] != 3 {
				t.Fatalf("cleanup stranded: done=%v leases=%d first attempts=%d", v.Done, len(x.leases), calls[first])
			}
		})
	}
}

func TestCompensationCursorValidation(t *testing.T) {
	x := setup(t)
	_, r, err := x.s.load()
	if err != nil {
		t.Fatal(err)
	}
	for _, next := range []int{-1, 1, len(r.Nodes)} {
		r.AbortNext = next
		if err := r.validate(); err == nil {
			t.Fatalf("accepted invalid noncompensating cursor %d", next)
		}
	}
}
