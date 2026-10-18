package nodeagent

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/workerprobe"
)

func TestLearnerPreparationNumericalRejectsInvalidBeforeCaching(t *testing.T) {
	for _, kind := range []string{"build", "config", "program", "layout", "optimizer", "outer", "precision", "missing_learner", "batch"} {
		t.Run(kind, func(t *testing.T) {
			f := newLearnerPreparationFixture(t)
			_, err := f.s.PrepareLearnerChecked(context.Background(), f.actor, f.accepted, f.lease.Version, f.ports())
			preparationCheck(t, err)
			bad := f.numerical()
			switch kind {
			case "build":
				bad.BuildID = strings.Repeat("f", 64)
			case "config":
				bad.ConfigHash = strings.Repeat("f", 64)
			case "program":
				bad.ProgramHash = strings.Repeat("f", 64)
			case "layout":
				bad.WeightLayoutHash = strings.Repeat("f", 64)
			case "optimizer":
				bad.OptimizerHash = strings.Repeat("f", 64)
			case "outer":
				bad.Learner.Outer = workerprobe.LearnerPlan{}.Outer
			case "precision":
				bad.Learner.Precision = nil
			case "missing_learner":
				bad.Learner = nil
			case "batch":
				bad.Learner.BatchTokens = 0
			}
			_, err = f.s.PrepareLearnerNumerical(context.Background(), f.actor, f.accepted, func(context.Context, []byte) (workerprobe.Plan, error) { return bad, nil }, f.clock)
			if err == nil || f.cache(t).Numerical != nil {
				t.Fatalf("invalid %s poisoned cache: %v", kind, err)
			}
			want := f.numerical()
			got, err := f.s.PrepareLearnerNumerical(context.Background(), f.actor, f.accepted, func(_ context.Context, config []byte) (workerprobe.Plan, error) {
				if string(config) != string(f.manifest.Config) {
					t.Fatal("wrong numerical config")
				}
				return want, nil
			}, f.clock)
			preparationCheck(t, err)
			got.Learner.Precision.Backend = "changed"
			f.s, err = Open(f.s.path, f.s.cluster, f.s.node)
			preparationCheck(t, err)
			replay, err := f.s.PrepareLearnerNumerical(context.Background(), f.actor, f.accepted, nil, f.clock)
			preparationCheck(t, err)
			if digest(replay) != digest(want) {
				t.Fatal("numerical replay changed")
			}
		})
	}
}

func TestLearnerPreparationNumericalRechecksLiveLease(t *testing.T) {
	for _, abort := range []bool{false, true} {
		name := "renew"
		if abort {
			name = "abort"
		}
		t.Run(name, func(t *testing.T) {
			f := newLearnerPreparationFixture(t)
			_, err := f.s.PrepareLearnerChecked(context.Background(), f.actor, f.accepted, f.lease.Version, f.ports())
			preparationCheck(t, err)
			_, err = f.s.PrepareLearnerNumerical(context.Background(), f.actor, f.accepted, func(context.Context, []byte) (workerprobe.Plan, error) {
				ctx, cancel := context.WithTimeout(context.Background(), time.Second)
				defer cancel()
				if abort {
					_, err := f.s.AbortReservation(ctx, f.actor, f.reserve, f.now)
					preparationCheck(t, err)
				} else {
					l, err := f.s.LeaseStatus(ctx, f.actor, f.lease.ID, f.now)
					preparationCheck(t, err)
					_, err = f.s.RenewLease(ctx, f.actor, LeaseCommand{Lease: l.ID, ExpectedVersion: l.Version, IdempotencyKey: strings.Repeat("a", 32), TTLSeconds: 600}, f.now)
					preparationCheck(t, err)
				}
				return f.numerical(), nil
			}, f.clock)
			if abort {
				if err == nil || f.cache(t).Numerical != nil {
					t.Fatalf("cached after abort: %v", err)
				}
			} else {
				preparationCheck(t, err)
			}
		})
	}
}
