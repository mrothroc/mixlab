package learner

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/mrothroc/mixlab/diloco/optimizer"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/precision"
)

func replayEvaluation(f *fixture) run.EvaluationAssignment {
	return run.EvaluationAssignment{Format: "mixlab_evaluation_assignment_v1", ID: "eval0", Binding: f.a.Binding,
		Coordinator: f.a.Coordinator, Round: 0, Model: f.a.BaseModel, ValidationHash: f.a.Contract.DatasetHash,
		Split: "validation", BatchTokens: 8, SequenceLength: 4, Batches: 1, WorkerID: f.a.WorkerID,
		Precision: precision.Policy{ComputeDType: "float32", Backend: "metal"}, PrecisionHash: f.a.PrecisionHash, IssuedAt: 1200, ExpiresAt: 1800}
}

func TestPrivatePublicationProcessCrashReplay(t *testing.T) {
	if root := os.Getenv("MIXLAB_P7_PRIVATE_CRASH_ROOT"); root != "" {
		f := setupAt(t, root)
		point := os.Getenv("MIXLAB_P7_PRIVATE_CRASH_POINT")
		f.r.Store.fault = func(got string) error {
			if got == point {
				os.Exit(77)
			}
			return nil
		}
		f.run(t)
		if point == "evaluation" {
			_, err := f.r.Evaluate(context.Background(), replayEvaluation(f), func(context.Context, run.EvaluationAssignment, []optimizer.Tensor) (float64, uint64, error) {
				return 2, 8, nil
			})
			if err != nil {
				t.Fatal(err)
			}
		}
		t.Fatal("publication fault was not reached")
	}
	baseline := setup(t)
	wantReceipt := baseline.run(t)
	_, wantJournal, err := baseline.r.Store.load()
	if err != nil {
		t.Fatal(err)
	}
	want, err := baseline.r.Store.restore(context.Background(), *wantJournal.Completed)
	if err != nil {
		t.Fatal(err)
	}
	for _, point := range []string{"start", "completed", "receipt", "evaluation"} {
		t.Run(point, func(t *testing.T) {
			f := setup(t)
			cmd := exec.Command(os.Args[0], "-test.run=^TestPrivatePublicationProcessCrashReplay$")
			cmd.Env = append(os.Environ(), "MIXLAB_P7_PRIVATE_CRASH_ROOT="+filepath.Dir(f.r.Store.path.Dir()), "MIXLAB_P7_PRIVATE_CRASH_POINT="+point)
			var exit *exec.ExitError
			if output, err := cmd.CombinedOutput(); !errors.As(err, &exit) || exit.ExitCode() != 77 {
				t.Fatalf("expected abrupt publication exit: %v\n%s", err, output)
			}
			f.r.Store, err = Open(f.r.Store.path, f.r.Store.limit)
			if err != nil {
				t.Fatal(err)
			}
			_, before, err := f.r.Store.load()
			if err != nil {
				t.Fatal(err)
			}
			f.r.Scope.ExpectedPrivateHash, err = run.Digest(before.Start)
			if err != nil {
				t.Fatal(err)
			}
			if got := f.run(t); got != wantReceipt {
				t.Fatal("recovery changed authoritative receipt", got, wantReceipt)
			}
			wantCalls := 0
			if point == "start" {
				wantCalls = 1
			}
			if f.calls != wantCalls {
				t.Fatalf("recovery consumed extra training: %d want %d", f.calls, wantCalls)
			}
			_, after, err := f.r.Store.load()
			if err != nil {
				t.Fatal(err)
			}
			got, err := f.r.Store.restore(context.Background(), *after.Completed)
			if err != nil || !reflect.DeepEqual(want, got) || !reflect.DeepEqual(wantJournal.Update, after.Update) {
				t.Fatal("process recovery changed model, moments, sampler, RNG, policy or update", err)
			}
			if point == "evaluation" {
				result, err := f.r.Evaluate(context.Background(), replayEvaluation(f), func(context.Context, run.EvaluationAssignment, []optimizer.Tensor) (float64, uint64, error) {
					t.Fatal("published evaluation reran after process crash")
					return 0, 0, nil
				})
				if err != nil || result.Loss != 2 || result.PrivateBeforeHash != result.PrivateAfterHash {
					t.Fatal("evaluation replay lost private-state invariance", err)
				}
			}
		})
	}
}
