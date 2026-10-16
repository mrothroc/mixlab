package train

import (
	"context"
	"encoding/json"
	"fmt"
	"math"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/data"
	"github.com/mrothroc/mixlab/precision"
	"github.com/mrothroc/mixlab/workerjob"
)

// EvaluateLocalLearnerModel evaluates a verified global model in a separate
// engine. It has no private learner store, training sampler, or optimizer port.
// Validation reads one fixed batch at a time rather than retaining a whole split.
func EvaluateLocalLearnerModel(ctx context.Context, cfg *ArchConfig, model []LocalLearnerTensor, validation workerjob.LearnerValidation, policy precision.Policy) (float64, uint64, error) {
	return evaluateLocalLearnerModel(ctx, cfg, model, validation, policy, nil)
}

func evaluateLocalLearnerModel(ctx context.Context, cfg *ArchConfig, model []LocalLearnerTensor, v workerjob.LearnerValidation, policy precision.Policy, testFactory localLearnerFactory) (float64, uint64, error) {
	if err := ctx.Err(); err != nil {
		return 0, 0, err
	}
	if cfg == nil {
		return 0, 0, fmt.Errorf("evaluation config required")
	}
	if err := v.Validate(); err != nil {
		return 0, 0, err
	}
	if v.BatchTokens != uint64(cfg.Training.BatchTokens) || v.SequenceLength != uint64(cfg.SeqLen) {
		return 0, 0, fmt.Errorf("validation shapes differ from learner contract")
	}
	hash, err := data.DistributedDatasetIdentity(v.Pattern)
	if err != nil {
		return 0, 0, err
	}
	if hash != v.Hash {
		return 0, 0, fmt.Errorf("validation dataset digest mismatch")
	}
	b, err := json.Marshal(cfg)
	if err != nil {
		return 0, 0, err
	}
	var owned ArchConfig
	if err := json.Unmarshal(b, &owned); err != nil {
		return 0, 0, err
	}
	cfg = &owned
	if err := arch.ValidateDistributedConfig(cfg); err != nil {
		return 0, 0, err
	}
	if cfg.Training.Distributed == nil || cfg.Training.Distributed.Mode != "diloco" {
		return 0, 0, fmt.Errorf("DiLoCo evaluation config required")
	}
	if err := configureDatasetForTraining(cfg, v.Pattern, cfg.Name); err != nil {
		return 0, 0, err
	}
	if cfg.CharVocabSize > 0 {
		if _, err := configureCharFeaturesForTraining(cfg, v.Pattern); err != nil {
			return 0, 0, err
		}
	}
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		return 0, 0, err
	}
	w, err := localModelData(model, shapes)
	if err != nil {
		return 0, 0, err
	}
	factory := testFactory
	if factory == nil {
		actual, err := trainingPrecision(cfg)
		if err != nil {
			return 0, 0, err
		}
		if actual != policy || actual.Backend != "metal" {
			return 0, 0, fmt.Errorf("evaluation precision differs from registration")
		}
		p, err := BuildIRProgramFromConfig(cfg)
		if err != nil {
			return 0, 0, err
		}
		factory, err = makeLocalLearnerFactory(cfg, p, 0)
		if err != nil {
			return 0, 0, err
		}
	}
	engine, err := factory(w, true)
	if err != nil {
		return 0, 0, err
	}
	defer engine.close()
	loader, err := data.NewLoaderWithOptions(v.Pattern, 0, data.LoaderOptions{NoShardShuffle: true, ChunkSize: cfg.SeqLen})
	if err != nil {
		return 0, 0, err
	}
	var total float64
	var tokens uint64
	for range v.Batches {
		if err := ctx.Err(); err != nil {
			return 0, 0, err
		}
		batch, err := loader.NextBatchDetailed(cfg.Training.BatchTokens, cfg.SeqLen)
		if err != nil {
			return 0, 0, err
		}
		batch.SeqLen = cfg.SeqLen
		loss, n, err := evaluateLocalBatch(cfg, engine, batch, 0)
		if err != nil {
			return 0, 0, err
		}
		total += loss * float64(n)
		tokens += n
	}
	if err := ctx.Err(); err != nil {
		return 0, 0, err
	}
	if tokens == 0 || math.IsNaN(total) || math.IsInf(total, 0) {
		return 0, 0, fmt.Errorf("invalid global evaluation total")
	}
	return total / float64(tokens), tokens, nil
}

func evaluateLocalBatch(cfg *ArchConfig, engine localLearnerEngine, b data.Batch, step int) (float64, uint64, error) {
	if len(b.X) != cfg.Training.BatchTokens || len(b.Y) != len(b.X) || b.SeqLen != cfg.SeqLen {
		return 0, 0, fmt.Errorf("evaluation batch shape mismatch")
	}
	if (cfg.Training.DatasetRecordFraming && len(b.LossMask) != len(b.X)) || (!cfg.Training.DatasetRecordFraming && len(b.LossMask) != 0) {
		return 0, 0, fmt.Errorf("evaluation loss mask differs from program contract")
	}
	for i, x := range b.X {
		if x < 0 || x >= cfg.VocabSize || b.Y[i] < 0 || b.Y[i] >= cfg.VocabSize {
			return 0, 0, fmt.Errorf("evaluation token outside vocabulary")
		}
	}
	for _, mask := range b.LossMask {
		if mask != 0 && mask != 1 {
			return 0, 0, fmt.Errorf("evaluation loss mask must be binary")
		}
	}
	p, err := prepareObjectiveBatch(cfg, trainBatchFromDataBatch(b, nil), step, arch.ObjectiveCausal)
	if err != nil {
		return 0, 0, err
	}
	loss, err := engine.evaluate(p)
	if err != nil {
		return 0, 0, err
	}
	if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) || loss < 0 {
		return 0, 0, fmt.Errorf("non-finite or negative global evaluation")
	}
	return float64(loss), uint64(p.lossNormalizer), nil
}
