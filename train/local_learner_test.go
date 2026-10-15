package train

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/mrothroc/mixlab/data"
	"github.com/mrothroc/mixlab/gpu"
)

func learnerFixture(t *testing.T, k int) (*ArchConfig, string, data.LearnerIdentity, []LocalLearnerTensor) {
	t.Helper()
	cfg, e := ParseArchConfig([]byte(fmt.Sprintf(`{"model_dim":8,"vocab_size":16,"seq_len":4,"dropout":0.1,"blocks":[{"type":"plain","heads":2}],"training":{"objective":"causal","optimizer":"adamw","steps":8,"batch_tokens":8,"seed":19,"lr":0.001,"weight_decay":0,"distributed":{"mode":"diloco","inner_steps":2,"outer_optimizer":"nesterov","outer_lr":1,"outer_momentum":0,"aggregation":"uniform","gradient_accumulation_steps":%d}}}`, k)), "local_learner")
	if e != nil {
		t.Fatal(e)
	}
	dir := t.TempDir()
	tokens := make([]uint16, 513)
	for i := range tokens {
		tokens[i] = uint16((i*7 + 3) % 16)
	}
	writeTestShard(t, dir, "train.bin", tokens)
	shapes, e := computeWeightShapes(cfg)
	if e != nil {
		t.Fatal(e)
	}
	w := initWeightData(shapes, cfg.Training.Seed, cfg.Training.WeightInit, cfg.Training.WeightInitStd)
	return cfg, filepath.Join(dir, "train.bin"), data.LearnerIdentity{RunID: "run", WorkerID: "worker", PartitionID: "partition", Count: 1, Seed: 19}, localModel(w, shapes)
}

func cloneLearnerValue[T any](v T) T {
	b, e := json.Marshal(v)
	if e != nil {
		panic(e)
	}
	var out T
	if e = json.Unmarshal(b, &out); e != nil {
		panic(e)
	}
	return out
}

// The fake isolates session scheduling and transaction semantics. Numerical
// AdamW/accumulation correctness is covered by the real MLX oracle tests.
type fakeLocalLearner struct {
	w         [][]float32
	s         gpu.TrainerStateSnapshot
	k, micro  int
	skip      *bool
	afterStep func()
}

func (f *fakeLocalLearner) step(b objectiveBatch, step int, lr float32) (float32, error) {
	f.micro++
	if f.afterStep != nil {
		f.afterStep()
	}
	if f.micro%f.k == 0 {
		f.s.Optimizer.AttemptedSteps++
		if *f.skip {
			f.s.Optimizer.SkippedSteps++
			return 1, nil
		}
		f.s.Optimizer.CommittedSteps++
		f.w[0][0] -= lr * float32(b.x[0]+step+1)
		f.s.Tensors[0].Data[0] += float32(step + 1)
	}
	return 1, nil
}
func (f *fakeLocalLearner) weights() ([][]float32, error) { return cloneLearnerValue(f.w), nil }
func (f *fakeLocalLearner) state() (gpu.TrainerStateSnapshot, error) {
	return cloneLearnerValue(f.s), nil
}
func (f *fakeLocalLearner) stats() (gpu.TrainerOptimizerStats, error) { return f.s.Optimizer, nil }
func (f *fakeLocalLearner) install(w [][]float32) error               { f.w = cloneLearnerValue(w); return nil }
func (f *fakeLocalLearner) restore(s gpu.TrainerStateSnapshot, step int) error {
	if len(s.Tensors) != 1 || s.Tensors[0].WeightIndex != 0 || !reflect.DeepEqual(s.Tensors[0].Shape, []int{1}) || len(s.Tensors[0].Data) != 1 {
		return fmt.Errorf("invalid moments")
	}
	f.s = cloneLearnerValue(s)
	f.micro = step
	return nil
}
func (f *fakeLocalLearner) evaluate(objectiveBatch) (float32, error) { return 2, nil }
func (f *fakeLocalLearner) close()                                   {}
func fakeLocalFactory(k int, skip *bool) localLearnerFactory {
	return func(w [][]float32, _ bool) (localLearnerEngine, error) {
		return &fakeLocalLearner{w: cloneLearnerValue(w), k: k, skip: skip, s: gpu.TrainerStateSnapshot{Tensors: []gpu.TrainerOptimizerStateTensor{{Kind: gpu.OptimizerStateAdamM, WeightIndex: 0, Shape: []int{1}, Data: []float32{0}}}}}, nil
	}
}

func TestLocalLearnerBoundaryReplayAndInstall(t *testing.T) {
	cfg, path, id, w := learnerFixture(t, 3)
	skip := false
	makeSession := func() *LocalLearnerSession {
		s, e := newLocalLearnerSession(cfg, path, id, w, fakeLocalFactory(3, &skip))
		if e != nil {
			t.Fatal(e)
		}
		t.Cleanup(s.Close)
		return s
	}
	s := makeSession()
	ctx := context.Background()
	if _, e := s.Train(ctx, 2); e != nil {
		t.Fatal(e)
	}
	start, e := s.Snapshot()
	if e != nil {
		t.Fatal(e)
	}
	if start.Attempt != 2 || start.EffectiveTokens != 48 || start.EffectiveExamples != 12 {
		t.Fatal("microsteps changed optimizer count", start.Attempt, start.EffectiveTokens)
	}
	if _, e = s.Train(ctx, 2); e != nil {
		t.Fatal(e)
	}
	want, _ := s.Snapshot()
	resumed := makeSession()
	if e = resumed.Restore(cloneLearnerValue(start)); e != nil {
		t.Fatal(e)
	}
	if _, e = resumed.Train(ctx, 2); e != nil {
		t.Fatal(e)
	}
	got, _ := resumed.Snapshot()
	if !reflect.DeepEqual(want, got) {
		t.Fatal("boundary replay differs")
	}
	if e = s.InstallGlobal(w); e != nil {
		t.Fatal(e)
	}
	installed, _ := s.Snapshot()
	if !reflect.DeepEqual(installed.Optimizer, want.Optimizer) || installed.Sampler != want.Sampler || installed.Attempt != want.Attempt {
		t.Fatal("global install reset private state")
	}
	b, e := s.loader.NextBatch(8)
	if e != nil {
		t.Fatal(e)
	}
	if e = s.loader.Restore(installed.Sampler); e != nil {
		t.Fatal(e)
	}
	if _, e = s.EvaluateGlobal(w, []data.Batch{b}); e != nil {
		t.Fatal(e)
	}
	after, _ := s.Snapshot()
	if !reflect.DeepEqual(installed, after) {
		t.Fatal("eval changed private state")
	}
}

func TestLocalLearnerSkipAndInterruptedMicrostep(t *testing.T) {
	for _, cancelMidway := range []bool{false, true} {
		cfg, path, id, w := learnerFixture(t, 3)
		skip := false
		s, e := newLocalLearnerSession(cfg, path, id, w, fakeLocalFactory(3, &skip))
		if e != nil {
			t.Fatal(e)
		}
		defer s.Close()
		before, _ := s.Snapshot()
		ctx, cancel := context.WithCancel(context.Background())
		if cancelMidway {
			s.engine.(*fakeLocalLearner).afterStep = cancel
		} else {
			skip = true
		}
		if _, e = s.Train(ctx, 2); e == nil {
			t.Fatal("failure hidden")
		}
		cancel()
		if _, e = s.Train(context.Background(), 1); e == nil {
			t.Fatal("failed session retrained")
		}
		if _, e = s.Snapshot(); e == nil {
			t.Fatal("partial snapshot accepted")
		}
		skip = false
		if e = s.Restore(before); e != nil {
			t.Fatal(e)
		}
		if _, e = s.Train(context.Background(), 2); e != nil {
			t.Fatal(e)
		}
	}
}

func TestLocalLearnerRejectsSnapshotsTransactionally(t *testing.T) {
	cfg, path, id, w := learnerFixture(t, 2)
	skip := false
	s, e := newLocalLearnerSession(cfg, path, id, w, fakeLocalFactory(2, &skip))
	if e != nil {
		t.Fatal(e)
	}
	defer s.Close()
	if _, e = s.Train(context.Background(), 2); e != nil {
		t.Fatal(e)
	}
	before, _ := s.Snapshot()
	for name, change := range map[string]func(*LocalLearnerSnapshot){
		"identity":       func(v *LocalLearnerSnapshot) { v.Sampler.Identity.WorkerID = "other" },
		"precision":      func(v *LocalLearnerSnapshot) { v.Precision.TF32Allowed = !v.Precision.TF32Allowed },
		"schedule":       func(v *LocalLearnerSnapshot) { v.Contract = "other" },
		"cursor":         func(v *LocalLearnerSnapshot) { v.Sampler.Counter.ChunkOffset = -1 },
		"layout":         func(v *LocalLearnerSnapshot) { v.Model[0].Name = "other" },
		"nan_weight":     func(v *LocalLearnerSnapshot) { v.Model[0].Data[0] = float32(math.NaN()) },
		"nan_moment":     func(v *LocalLearnerSnapshot) { v.Optimizer.Tensors[0].Data[0] = float32(math.Inf(1)) },
		"missing_moment": func(v *LocalLearnerSnapshot) { v.Optimizer.Tensors = nil },
		"count":          func(v *LocalLearnerSnapshot) { v.Attempt++ },
	} {
		t.Run(name, func(t *testing.T) {
			bad := cloneLearnerValue(before)
			change(&bad)
			if e = s.Restore(bad); e == nil {
				t.Fatal("invalid snapshot restored")
			}
			after, _ := s.Snapshot()
			if !reflect.DeepEqual(before, after) {
				t.Fatal("failed restore mutated state")
			}
		})
	}
	bad := cloneLearnerValue(w)
	bad[len(bad)-1].Data[0] = float32(math.Inf(1))
	if e = s.InstallGlobal(bad); e == nil {
		t.Fatal("invalid install")
	}
	after, _ := s.Snapshot()
	if !reflect.DeepEqual(before, after) {
		t.Fatal("partial global install")
	}
}

func TestLocalLearnerRNGIdentity(t *testing.T) {
	_, _, id, _ := learnerFixture(t, 2)
	a, b := make([]int32, 12), make([]int32, 12)
	fillDistributedDropoutKeys(a, id.RandomSeed(), 3, 0, 1)
	fillDistributedDropoutKeys(b, id.RandomSeed(), 3, 0, 1)
	if !reflect.DeepEqual(a, b) {
		t.Fatal("replay changed RNG")
	}
	id.WorkerID = "other"
	fillDistributedDropoutKeys(b, id.RandomSeed(), 3, 0, 1)
	if reflect.DeepEqual(a, b) {
		t.Fatal("worker streams alias")
	}
}

func TestLocalLearnerEvaluationRejectsInvalidBatches(t *testing.T) {
	cfg, path, id, w := learnerFixture(t, 2)
	skip := false
	s, err := newLocalLearnerSession(cfg, path, id, w, fakeLocalFactory(2, &skip))
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	before, err := s.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	for name, batch := range map[string]data.Batch{
		"shape":         {X: []int{1}, Y: []int{2}, SeqLen: 4},
		"token":         {X: []int{16, 1, 1, 1, 1, 1, 1, 1}, Y: make([]int, 8), SeqLen: 4},
		"target":        {X: make([]int, 8), Y: []int{-1, 1, 1, 1, 1, 1, 1, 1}, SeqLen: 4},
		"mask_contract": {X: make([]int, 8), Y: make([]int, 8), LossMask: make([]float32, 8), SeqLen: 4},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := s.EvaluateGlobal(w, []data.Batch{batch}); err == nil {
				t.Fatal("invalid evaluation batch accepted")
			}
		})
	}
	if _, err := s.EvaluateGlobal(w, nil); err == nil {
		t.Fatal("empty evaluation accepted")
	}
	after, err := s.Snapshot()
	if err != nil || !reflect.DeepEqual(before, after) {
		t.Fatal("invalid evaluation changed private state", err)
	}
}
