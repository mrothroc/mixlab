//go:build mlx && cgo && (darwin || linux)

package train

import (
	"context"
	"math"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/data"
	"github.com/mrothroc/mixlab/gpu"
	"github.com/mrothroc/mixlab/workerjob"
)

func TestLocalLearnerMLXReplayInstallAndEvaluation(t *testing.T) {
	if !gpu.Available() {
		t.Skip("MLX unavailable")
	}
	cfg, path, id, w := learnerFixture(t, 2)
	s, e := NewLocalLearnerSession(cfg, path, id, w)
	if e != nil {
		t.Fatal(e)
	}
	defer s.Close()
	var optionsErr error
	s.engine.(*threadedLocalLearner).call(func(e localLearnerEngine) {
		optionsErr = gpu.TrainerSetDistributedOptions(e.(*mlxLocalLearner).t.handle, 1024, 3)
	})
	if optionsErr == nil {
		t.Fatal("distributed options overwrote active local accumulation")
	}
	ctx := context.Background()
	if _, e = s.Train(ctx, 2); e != nil {
		t.Fatal(e)
	}
	start, e := s.Snapshot()
	if e != nil {
		t.Fatal(e)
	}
	if _, e = s.Train(ctx, 2); e != nil {
		t.Fatal(e)
	}
	want, e := s.Snapshot()
	if e != nil {
		t.Fatal(e)
	}
	if e = s.Restore(start); e != nil {
		t.Fatal(e)
	}
	if _, e = s.Train(ctx, 2); e != nil {
		t.Fatal(e)
	}
	got, e := s.Snapshot()
	if e != nil {
		t.Fatal(e)
	}
	assertLocalSnapshotNear(t, want, got)
	if e = s.InstallGlobal(w); e != nil {
		t.Fatal(e)
	}
	installed, e := s.Snapshot()
	if e != nil {
		t.Fatal(e)
	}
	if !reflect.DeepEqual(installed.Optimizer, got.Optimizer) || installed.Sampler != got.Sampler {
		t.Fatal("global install changed moments/cursor")
	}
	if _, e = s.Train(ctx, 2); e != nil {
		t.Fatal(e)
	}
	before, e := s.Snapshot()
	if e != nil {
		t.Fatal(e)
	}
	l, e := data.NewLearnerLoader(path, id, 4, 4, 16)
	if e != nil {
		t.Fatal(e)
	}
	batch, e := l.NextBatch(8)
	if e != nil {
		t.Fatal(e)
	}
	loss, e := s.EvaluateGlobal(w, []data.Batch{batch})
	if e != nil || math.IsNaN(loss) || math.IsInf(loss, 0) {
		t.Fatal("evaluation", loss, e)
	}
	after, e := s.Snapshot()
	if e != nil {
		t.Fatal(e)
	}
	if !reflect.DeepEqual(before, after) {
		t.Fatal("evaluation mutated private state")
	}
	hash, e := data.DistributedDatasetIdentity(path)
	if e != nil {
		t.Fatal(e)
	}
	validation := workerjob.LearnerValidation{Pattern: path, Hash: hash, Split: "validation", BatchTokens: 8, SequenceLength: 4, Batches: 2}
	loss, tokens, e := EvaluateLocalLearnerModel(ctx, cfg, w, validation, before.Precision)
	if e != nil || math.IsNaN(loss) || math.IsInf(loss, 0) || tokens != 16 {
		t.Fatal("managed global evaluation", loss, tokens, e)
	}
	otherLoss, otherTokens, e := EvaluateLocalLearnerModel(ctx, cfg, w, validation, before.Precision)
	if e != nil || otherLoss != loss || otherTokens != tokens {
		t.Fatal("global evaluation replay changed", e)
	}
	after, e = s.Snapshot()
	if e != nil || !reflect.DeepEqual(before, after) {
		t.Fatal("isolated global evaluation changed private state", e)
	}
	// Exercise native all-or-nothing installation, bypassing Go's preflight.
	bad, _ := localModelData(w, s.shapes)
	bad[len(bad)-1][0] = float32(math.NaN())
	if e = s.engine.install(bad); e == nil {
		t.Fatal("native install accepted NaN")
	}
	after, e = s.Snapshot()
	if e != nil || !reflect.DeepEqual(before, after) {
		t.Fatal("native install partially mutated model", e)
	}
	badSnapshot := cloneLearnerValue(before)
	badSnapshot.Optimizer.Tensors[len(badSnapshot.Optimizer.Tensors)-1].Shape = []int{999}
	if e = s.Restore(badSnapshot); e == nil {
		t.Fatal("malformed moments restored")
	}
	after, e = s.Snapshot()
	if e != nil || !reflect.DeepEqual(before, after) {
		t.Fatal("failed native restore changed live state", e)
	}
}

func assertLocalSnapshotNear(t *testing.T, want, got LocalLearnerSnapshot) {
	t.Helper()
	if want.Sampler != got.Sampler || want.Attempt != got.Attempt || want.EffectiveTokens != got.EffectiveTokens || want.Optimizer.Optimizer != got.Optimizer.Optimizer {
		t.Fatal("replay counters/cursor differ")
	}
	for i := range want.Model {
		for j, v := range want.Model[i].Data {
			if math.Abs(float64(v-got.Model[i].Data[j])) > 1e-6 {
				t.Fatalf("weight replay mismatch %d/%d", i, j)
			}
		}
	}
	for i := range want.Optimizer.Tensors {
		for j, v := range want.Optimizer.Tensors[i].Data {
			if math.Abs(float64(v-got.Optimizer.Tensors[i].Data[j])) > 1e-6 {
				t.Fatalf("moment replay mismatch %d/%d", i, j)
			}
		}
	}
}

func TestLocalLearnerMLXAccumulationOracle(t *testing.T) {
	t.Run("dense", func(t *testing.T) { testLocalAccumulationOracle(t, false) })
	t.Run("unequal_valid_tokens", func(t *testing.T) { testLocalAccumulationOracle(t, true) })
}

func testLocalAccumulationOracle(t *testing.T, masked bool) {
	if !gpu.Available() {
		t.Skip("MLX unavailable")
	}
	cfg, _, id, w := learnerFixture(t, 2)
	cfg.Dropout = 0
	cfg.Training.DatasetRecordFraming = masked
	program, e := BuildIRProgramFromConfig(cfg)
	if e != nil {
		t.Fatal(e)
	}
	factory, e := makeLocalLearnerFactory(cfg, program, id.RandomSeed())
	if e != nil {
		t.Fatal(e)
	}
	shapes, e := computeWeightShapes(cfg)
	if e != nil {
		t.Fatal(e)
	}
	weights, e := localModelData(w, shapes)
	if e != nil {
		t.Fatal(e)
	}
	engine, e := factory(weights, false)
	if e != nil {
		t.Fatal(e)
	}
	defer engine.close()
	refCfg := *cfg
	refCfg.Training = cfg.Training
	refCfg.Training.Distributed = nil
	refCfg.Training.BatchTokens = 16
	refProgram, e := BuildIRProgramFromConfig(&refCfg)
	if e != nil {
		t.Fatal(e)
	}
	ref, e := initGPUTrainer(refProgram, &refCfg, weights, nil)
	if e != nil {
		t.Fatal(e)
	}
	defer ref.CloseTrainer()
	for round := 0; round < 3; round++ {
		x, y := make([]int, 16), make([]int, 16)
		for i := range x {
			x[i] = (i + round) % 16
			y[i] = (i + round + 1) % 16
		}
		var mask []float32
		if masked {
			mask = make([]float32, 16)
			for i := 0; i < 10; i++ {
				mask[i] = 1
			}
		}
		for micro := 0; micro < 2; micro++ {
			b := trainBatch{x: x[micro*8 : (micro+1)*8], y: y[micro*8 : (micro+1)*8]}
			if masked {
				b.lossMask = mask[micro*8 : (micro+1)*8]
			}
			p, e := prepareObjectiveBatch(cfg, b, round, arch.ObjectiveCausal)
			if e != nil {
				t.Fatal(e)
			}
			if _, e = engine.step(p, round*2+micro, .001); e != nil {
				t.Fatal(e)
			}
			if micro == 0 {
				if e = engine.install(weights); e == nil {
					t.Fatal("installed global model during incomplete accumulation")
				}
			}
		}
		prepared, e := prepareObjectiveBatch(&refCfg, trainBatch{x: x, y: y, lossMask: mask}, round, arch.ObjectiveCausal)
		if e != nil {
			t.Fatal(e)
		}
		if e = submitPreparedStepGPU(ref, prepared, 4, 4, .001); e != nil {
			t.Fatal(e)
		}
		if _, e = ref.CollectLossGPU(); e != nil {
			t.Fatal(e)
		}
		actual, e := engine.weights()
		if e != nil {
			t.Fatal(e)
		}
		expected, e := readTrainerWeights(ref)
		if e != nil {
			t.Fatal(e)
		}
		if diff := maxWeightDifference(actual, expected); diff > 1e-5 {
			t.Fatalf("round %d accumulation diff %g", round, diff)
		}
		stats, e := engine.stats()
		if e != nil || stats.AttemptedSteps != uint64(round+1) || stats.CommittedSteps != uint64(round+1) {
			t.Fatal("wrong attempt semantics", stats, e)
		}
		// LR=1, outer momentum=0, singleton DiLoCo installs its local result.
		if e = engine.install(actual); e != nil {
			t.Fatal(e)
		}
	}
	groupActive := false
	engine.(*threadedLocalLearner).call(func(e localLearnerEngine) { groupActive = e.(*mlxLocalLearner).t.distributed != nil })
	if groupActive {
		t.Fatal("local learner acquired a collective group")
	}
}

func TestLocalLearnerMLXRecordReplay(t *testing.T) {
	if !gpu.Available() {
		t.Skip("MLX unavailable")
	}
	cfg, _, id, w := learnerFixture(t, 2)
	cfg.SeqLen = 6
	cfg.Training.BatchTokens = 12
	dir := t.TempDir()
	path := filepath.Join(dir, "train_000.bin")
	writeNucleotideMLXSequenceShard(t, path, [][]uint16{{3}, {4}, {5, 6}, {7}, {8, 9}, {10}})
	writeTextRecordDatasetContract(t, dir, 16, 6)
	s, e := NewLocalLearnerSession(cfg, path, id, w)
	if e != nil {
		t.Fatal(e)
	}
	defer s.Close()
	start, e := s.Snapshot()
	if e != nil {
		t.Fatal(e)
	}
	l, e := data.NewLearnerLoader(path, id, 6, 6, 16)
	if e != nil {
		t.Fatal(e)
	}
	var count uint64
	for range 4 {
		b, e := l.NextBatch(12)
		if e != nil {
			t.Fatal(e)
		}
		for _, m := range b.LossMask {
			count += uint64(m)
		}
	}
	r, e := s.Train(context.Background(), 2)
	if e != nil {
		t.Fatal(e)
	}
	if r.EffectiveTokens != count || r.EffectiveExamples != 8 {
		t.Fatal("record valid-token accounting", r, count)
	}
	want, e := s.Snapshot()
	if e != nil {
		t.Fatal(e)
	}
	if e = s.Restore(start); e != nil {
		t.Fatal(e)
	}
	if _, e = s.Train(context.Background(), 2); e != nil {
		t.Fatal(e)
	}
	got, e := s.Snapshot()
	if e != nil {
		t.Fatal(e)
	}
	assertLocalSnapshotNear(t, want, got)
}
