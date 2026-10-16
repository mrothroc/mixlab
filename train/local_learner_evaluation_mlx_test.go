//go:build mlx && cgo && (darwin || linux)

package train

import (
	"context"
	"errors"
	"math"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/data"
	"github.com/mrothroc/mixlab/gpu"
	"github.com/mrothroc/mixlab/precision"
	"github.com/mrothroc/mixlab/workerjob"
)

func TestLocalLearnerEvaluationMLXPreservesContinuation(t *testing.T) {
	if !gpu.Available() {
		t.Skip("MLX unavailable")
	}
	cfg, path, id, _ := learnerFixture(t, 2)
	cfg.Dropout = .4
	cfg.Blocks = append(cfg.Blocks, cfg.Blocks[0])
	_, wire := workerInitialArtifact(t, cfg)
	model, err := LocalLearnerModelFromWire(wire)
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	baseline, err := NewLocalLearnerSession(cfg, path, id, model)
	if err != nil {
		t.Fatal(err)
	}
	defer baseline.Close()
	observed, err := NewLocalLearnerSession(cfg, path, id, model)
	if err != nil {
		t.Fatal(err)
	}
	defer observed.Close()
	for _, s := range []*LocalLearnerSession{baseline, observed} {
		if _, err := s.Train(ctx, 2); err != nil {
			t.Fatal(err)
		}
	}
	before, err := observed.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	hash, err := data.DistributedDatasetIdentity(path)
	if err != nil {
		t.Fatal(err)
	}
	v := workerjob.LearnerValidation{Pattern: path, Hash: hash, Split: "validation", BatchTokens: 8, SequenceLength: 4, Batches: 3}
	// Evaluate the global weights, not the learner's already-updated local model.
	loss, tokens, err := EvaluateLocalLearnerModel(ctx, cfg, model, v, before.Precision)
	if err != nil || math.IsNaN(loss) || math.IsInf(loss, 0) || tokens != 24 {
		t.Fatal(loss, tokens, err)
	}
	noDropout := cloneLearnerValue(cfg)
	noDropout.Dropout = 0
	refLoss, _, err := EvaluateLocalLearnerModel(ctx, noDropout, model, v, before.Precision)
	if err != nil || refLoss != loss {
		t.Fatal("evaluation did not disable dropout", loss, refLoss, err)
	}

	for _, failure := range []string{"precision", "digest", "model", "canceled"} {
		t.Run(failure, func(t *testing.T) {
			policy, validation, weights, callCtx := before.Precision, v, model, ctx
			switch failure {
			case "precision":
				policy.ComputeDType = "bf16"
			case "digest":
				validation.Hash = strings.Repeat("f", 64)
			case "model":
				weights = cloneLearnerValue(model)
				weights[len(weights)-1].Data[0] = float32(math.Inf(1))
			case "canceled":
				var cancel context.CancelFunc
				callCtx, cancel = context.WithCancel(ctx)
				cancel()
			}
			l, n, err := EvaluateLocalLearnerModel(callCtx, cfg, weights, validation, policy)
			if err == nil || l != 0 || n != 0 {
				t.Fatal("failed evaluation returned metrics", l, n, err)
			}
			if failure == "canceled" && !errors.Is(err, context.Canceled) {
				t.Fatal(err)
			}
		})
	}
	after, err := observed.Snapshot()
	if err != nil || !reflect.DeepEqual(before, after) {
		t.Fatal("evaluation changed private snapshot", err)
	}
	for _, s := range []*LocalLearnerSession{baseline, observed} {
		if _, err := s.Train(ctx, 2); err != nil {
			t.Fatal(err)
		}
	}
	want, err := baseline.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	got, err := observed.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	assertLocalSnapshotNear(t, want, got)
	if !reflect.DeepEqual(want, got) {
		t.Fatal("evaluation changed subsequent sampler/RNG/optimizer evolution")
	}
}

func TestLocalLearnerEvaluationRecordWeightedMetrics(t *testing.T) {
	cfg, _, _, model := learnerFixture(t, 2)
	cfg.SeqLen, cfg.Training.BatchTokens = 6, 12
	dir := t.TempDir()
	path := filepath.Join(dir, "train_000.bin")
	writeNucleotideMLXSequenceShard(t, path, [][]uint16{{3}, {4}, {5, 6}, {7, 8, 9}})
	writeTextRecordDatasetContract(t, dir, 16, 6)
	hash, err := data.DistributedDatasetIdentity(path)
	if err != nil {
		t.Fatal(err)
	}
	v := workerjob.LearnerValidation{Pattern: path, Hash: hash, Split: "validation", BatchTokens: 12, SequenceLength: 6, Batches: 2}
	var seen [][]int
	closed := false
	loss, tokens, err := evaluateLocalLearnerModel(context.Background(), cfg, model, v, precision.Policy{}, func(w [][]float32, _ bool) (localLearnerEngine, error) {
		return &observedLocalEvaluation{fakeLocalLearner: &fakeLocalLearner{w: w}, seen: &seen, closed: &closed}, nil
	})
	// BOS->content and content->EOS: 4 valid targets in batch one, 7 in batch two.
	if err != nil || tokens != 11 || loss != 18.0/11 || !closed {
		t.Fatal("unweighted or padded validation metric", loss, tokens, err)
	}
	if cfg.Training.DatasetRecordFraming {
		t.Fatal("evaluation leaked runtime dataset state into caller config")
	}
}
