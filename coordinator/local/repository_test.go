package local

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"sync"
	"testing"

	artifactlocal "github.com/mrothroc/mixlab/artifact/local"
	"github.com/mrothroc/mixlab/coordinator"
	"github.com/mrothroc/mixlab/diloco/learner"
	"github.com/mrothroc/mixlab/diloco/optimizer"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/statehome"
)

func protected(t *testing.T, dir string) statehome.Path {
	t.Helper()
	dir, e := filepath.EvalSymlinks(dir)
	if e != nil {
		t.Fatal(e)
	}
	if e = os.Chmod(dir, 0700); e != nil {
		t.Fatal(e)
	}
	p, e := statehome.Resolve(statehome.Options{ExactDir: dir}, statehome.Context{Kind: statehome.Principal, Role: "coordinator"})
	if e != nil {
		t.Fatal(e)
	}
	return p
}
func fixtureState(t *testing.T) coordinator.State {
	t.Helper()
	var p run.Plan
	b, e := os.ReadFile("../../diloco/testdata/plan.json")
	if e != nil {
		t.Fatal(e)
	}
	if e = json.Unmarshal(b, &p); e != nil {
		t.Fatal(e)
	}
	return coordinator.State{Revision: 1, Phase: coordinator.Staged, Plan: p, Policy: coordinator.EvaluationPolicy{WorkerID: p.Slots[0].Member.WorkerID, ValidationHash: p.Contract.DatasetHash, Split: "validation", BatchTokens: 8, SequenceLength: 4, Batches: 2, TimeoutSeconds: 100}}
}
func TestRepositoryCrashPublication(t *testing.T) {
	if dir := os.Getenv("MIXLAB_P3_CRASH_DIR"); dir != "" {
		r, e := Open(protected(t, dir))
		if e != nil {
			t.Fatal(e)
		}
		r.fault = func(stage string) error {
			if stage == os.Getenv("MIXLAB_P3_CRASH_AT") {
				os.Exit(77)
			}
			return nil
		}
		s := fixtureState(t)
		if e = r.Save(context.Background(), s.Plan.Cohort.RunID, 0, coordinator.Change{Next: s}); e != nil {
			t.Fatal(e)
		}
		return
	}
	for _, stage := range []string{"record", "head"} {
		t.Run(stage, func(t *testing.T) {
			dir := t.TempDir()
			cmd := exec.Command(os.Args[0], "-test.run=^TestRepositoryCrashPublication$")
			cmd.Env = append(os.Environ(), "MIXLAB_P3_CRASH_DIR="+dir, "MIXLAB_P3_CRASH_AT="+stage)
			if e := cmd.Run(); e == nil {
				t.Fatal("child did not crash")
			}
			r, e := Open(protected(t, dir))
			if e != nil {
				t.Fatal(e)
			}
			s := fixtureState(t)
			id := s.Plan.Cohort.RunID
			got, e := r.Load(context.Background(), id)
			if stage == "record" {
				if !errors.Is(e, coordinator.ErrNotFound) {
					t.Fatal("unpublished visible", e)
				}
				if n, e := r.PruneOrphans(context.Background(), id); e != nil || n != 1 {
					t.Fatal(n, e)
				}
				if e = r.Save(context.Background(), id, 0, coordinator.Change{Next: s}); e != nil {
					t.Fatal(e)
				}
			} else if e != nil || !reflect.DeepEqual(s, got) {
				t.Fatal("published state lost", e)
			}
			if n, e := r.PruneOrphans(context.Background(), id); e != nil || n != 0 {
				t.Fatal("removed live history", n, e)
			}
		})
	}
}
func TestRepositoryCASAndCorruption(t *testing.T) {
	p := protected(t, t.TempDir())
	r, e := Open(p)
	if e != nil {
		t.Fatal(e)
	}
	s := fixtureState(t)
	id := s.Plan.Cohort.RunID
	var wg sync.WaitGroup
	results := make(chan error, 12)
	for range 12 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			other, _ := Open(p)
			results <- other.Save(context.Background(), id, 0, coordinator.Change{Next: s})
		}()
	}
	wg.Wait()
	close(results)
	count := 0
	for e := range results {
		if e == nil {
			count++
		} else if !errors.Is(e, coordinator.ErrConflict) {
			t.Fatal(e)
		}
	}
	if count != 1 {
		t.Fatal(count)
	}
	h, _, e := r.current(id)
	if e != nil {
		t.Fatal(e)
	}
	k, _ := key(id)
	if e = p.WriteFile(k+"-"+h.Hash+".record", []byte(`{}`)); e != nil {
		t.Fatal(e)
	}
	if _, e = r.Load(context.Background(), id); e == nil {
		t.Fatal("corrupt history trusted")
	}
}

type clock struct{}

func (clock) Unix() int64 { return 1200 }

type allocations struct{}

func (allocations) Ready(context.Context, run.Slot) error { return nil }

type evaluator struct{ repo coordinator.Repository }

func (e evaluator) Evaluate(ctx context.Context, a run.EvaluationAssignment) (run.EvaluationResult, error) {
	s, err := e.repo.Load(ctx, a.Binding.RunID)
	if err != nil {
		return run.EvaluationResult{}, err
	}
	private := ""
	for i, slot := range s.Plan.Slots {
		if slot.Member.WorkerID == a.WorkerID {
			private = s.PrivateHashes[i]
		}
	}
	h, _ := run.Digest(a)
	return run.EvaluationResult{Format: "mixlab_evaluation_result_v1", Assignment: a, AssignmentHash: h, Loss: 2, EffectiveTokens: 8, StartedAt: 1200, FinishedAt: 1200, PrivateBeforeHash: private, PrivateAfterHash: private}, nil
}
func TestDurableCoordinatorRoundAndReceiptRecovery(t *testing.T) {
	ctx := context.Background()
	p := protected(t, t.TempDir())
	repo, _ := Open(p)
	bs, _ := artifactlocal.Open(protected(t, t.TempDir()))
	a := Artifacts{bs}
	s := fixtureState(t)
	base, e := a.Put(ctx, "model_f32", []optimizer.Tensor{{Name: "w0", Shape: []int{1}, Values: []float32{4}}})
	if e != nil {
		t.Fatal(e)
	}
	s.Plan.BaseModel = base
	s.Plan.Outer.TotalSteps = s.Plan.Outer.InnerSteps
	s.Plan.SHA256, _ = s.Plan.Digest()
	var commit run.Commit
	b, e := os.ReadFile("../../diloco/testdata/commit.json")
	if e != nil {
		t.Fatal(e)
	}
	if e = json.Unmarshal(b, &commit); e != nil {
		t.Fatal(e)
	}
	commit.Binding.PlanHash = s.Plan.SHA256
	for i := range commit.Registrations {
		commit.Registrations[i].PlanHash = s.Plan.SHA256
		commit.Registrations[i].BaseModelHash = base.Ref.SHA256
	}
	commit.Binding.CommitHash, _ = commit.Digest()
	m := coordinator.Model{Repository: repo, Artifacts: a, Allocations: allocations{}, Clock: clock{}, Evaluator: evaluator{repo}}
	if e = m.Stage(ctx, s.Plan, s.Policy); e != nil {
		t.Fatal(e)
	}
	for _, reg := range commit.Registrations {
		if e = repo.Register(ctx, reg); e != nil {
			t.Fatal(e)
		}
		if e = repo.Register(ctx, reg); e != nil {
			t.Fatal(e)
		}
	}
	regs, e := repo.Registrations(ctx, s.Plan.Cohort.RunID)
	if e != nil || !reflect.DeepEqual(regs, commit.Registrations) {
		t.Fatal(regs, e)
	}
	if e = m.Commit(ctx, s.Plan.Cohort.RunID, commit); e != nil {
		t.Fatal(e)
	}
	if e = repo.Register(ctx, commit.Registrations[0]); e != nil {
		t.Fatal("lost registration ack after commit", e)
	}
	assignments, e := m.OpenRound(ctx, s.Plan.Cohort.RunID, 100)
	if e != nil {
		t.Fatal(e)
	}
	delta, e := a.Put(ctx, "delta_f32", []optimizer.Tensor{{Name: "w0", Shape: []int{1}, Values: []float32{1}}})
	if e != nil {
		t.Fatal(e)
	}
	var receipts []learner.Receipt
	for i, assignment := range assignments {
		h, _ := run.Digest(assignment)
		member := s.Plan.Slots[i].Member
		u := learner.Update{Format: "mixlab_outer_update_v1", ID: fmt.Sprint("update-", i), AssignmentHash: h, Binding: assignment.Binding, Round: 0, BaseRound: 0, BaseModelHash: base.Ref.SHA256, WorkerID: member.WorkerID, ParticipantID: member.ParticipantID, IslandID: member.IslandID, Contract: assignment.Contract, Delta: delta, DType: "float32", TensorCount: 1, PrecisionHash: assignment.PrecisionHash, PrivateHash: s.Plan.Contract.ConfigHash, Attempted: assignment.InnerSteps, Committed: assignment.InnerSteps, EffectiveTokens: 8, EffectiveExamples: 2, Loss: 1, DeltaL2: 1, DeltaMaxAbs: 1, StartedAt: 1200, FinishedAt: 1200}
		receipt, e := m.Accept(ctx, s.Plan.Cohort.RunID, u)
		if e != nil {
			t.Fatal(e)
		}
		receipts = append(receipts, receipt)
	}
	// Restart at every state transition, including an ambiguous post-head error.
	for i := 0; i < 4; i++ {
		repo, _ = Open(p)
		m.Repository = repo
		m.Evaluator = evaluator{repo}
		if i == 2 {
			repo.fault = func(stage string) error {
				if stage == "head" {
					return errors.New("lost ack")
				}
				return nil
			}
		}
		e = m.Advance(ctx, s.Plan.Cohort.RunID)
		if (e != nil) != (i == 2) {
			t.Fatal(i, e)
		}
	}
	repo, _ = Open(p)
	got, e := repo.Load(ctx, s.Plan.Cohort.RunID)
	if e != nil || got.Phase != coordinator.Complete || got.Round != 1 {
		t.Fatal(got.Phase, e)
	}
	t.Run("global_checkpoint_catalog_recovery", func(t *testing.T) {
		cp := protected(t, t.TempDir())
		catalog, err := OpenCheckpoints(cp)
		if err != nil {
			t.Fatal(err)
		}
		id := s.Plan.Cohort.RunID
		// No catalog writer ran during publication/evaluation. Reconstruct the
		// durable global reference after process restart, without private data.
		for range 2 {
			if err = catalog.Sync(ctx, repo, id); err != nil {
				t.Fatal(err)
			}
		}
		k, _ := key(id)
		b, err := cp.ReadFileLimit(fmt.Sprintf("%s-%020d.global.json", k, 0), maxRecord)
		if err != nil {
			t.Fatal(err)
		}
		var published run.Round
		if err = json.Unmarshal(b, &published); err != nil {
			t.Fatal(err)
		}
		if got.Candidate == nil || !reflect.DeepEqual(published, *got.Candidate) {
			t.Fatal("catalog differs from published global checkpoint")
		}
		if err = cp.WriteFile(k+".catalog.head", []byte(`{"revision":999999,"hash":"changed"}`)); err != nil {
			t.Fatal(err)
		}
		if err = catalog.Sync(ctx, repo, id); err == nil {
			t.Fatal("accepted catalog ahead of authoritative history")
		}
	})
	for _, want := range receipts {
		got, e := repo.Receipt(ctx, s.Plan.Cohort.RunID, want.UpdateID)
		if e != nil || got != want {
			t.Fatal(got, e)
		}
	}
	if n, e := repo.PruneOrphans(ctx, s.Plan.Cohort.RunID); e != nil || n != 0 {
		t.Fatal(n, e)
	}
}
