package train

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"sync"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/data"
	"github.com/mrothroc/mixlab/gpu"
)

// LocalLearnerSession owns one numerical engine and sampler, not a network
// participant. Calls are serialized; no lookahead or collective group is used.
type LocalLearnerSession struct {
	mu                        sync.Mutex
	cfg                       *ArchConfig
	loader                    *data.LearnerLoader
	engine                    localLearnerEngine
	factory                   localLearnerFactory
	shapes                    []WeightShape
	policy                    gpu.PrecisionPolicy
	contract                  string
	schedule                  trainingScheduler
	attempt, tokens, examples uint64
	failed                    bool
}

type LocalLearnerTensor struct {
	Name  string    `json:"name"`
	Shape []int     `json:"shape"`
	Data  []float32 `json:"data"`
}

// LocalLearnerSnapshot is an owned boundary snapshot. Durable completion markers
// and artifact serialization belong to the learner application, not this engine.
type LocalLearnerSnapshot struct {
	Format            string                   `json:"format"`
	Contract          string                   `json:"contract"`
	Precision         gpu.PrecisionPolicy      `json:"precision"`
	Sampler           data.LearnerSamplerState `json:"sampler"`
	Model             []LocalLearnerTensor     `json:"model"`
	Optimizer         gpu.TrainerStateSnapshot `json:"optimizer"`
	Attempt           uint64                   `json:"attempt"`
	EffectiveTokens   uint64                   `json:"effective_tokens"`
	EffectiveExamples uint64                   `json:"effective_examples"`
}

type LocalLearnerResult struct {
	Attempted, Committed, Skipped      uint64
	EffectiveTokens, EffectiveExamples uint64
	Loss                               float64
}

type localLearnerEngine interface {
	step(objectiveBatch, int, float32) (float32, error)
	weights() ([][]float32, error)
	state() (gpu.TrainerStateSnapshot, error)
	stats() (gpu.TrainerOptimizerStats, error)
	install([][]float32) error
	restore(gpu.TrainerStateSnapshot, int) error
	evaluate(objectiveBatch) (float32, error)
	close()
}
type localLearnerFactory func([][]float32, bool) (localLearnerEngine, error)

// NewLocalLearnerSession requires the committed initial model. Identity seeds
// model stochastic work only; it never changes those initial parameters.
func NewLocalLearnerSession(cfg *ArchConfig, pattern string, identity data.LearnerIdentity, initial []LocalLearnerTensor) (*LocalLearnerSession, error) {
	return newLocalLearnerSession(cfg, pattern, identity, initial, nil)
}

func newLocalLearnerSession(cfg *ArchConfig, pattern string, identity data.LearnerIdentity, initial []LocalLearnerTensor, testFactory localLearnerFactory) (*LocalLearnerSession, error) {
	if cfg == nil || cfg.Training.Distributed == nil || cfg.Training.Distributed.Mode != "diloco" {
		return nil, fmt.Errorf("local learner requires DiLoCo config")
	}
	if e := arch.ValidateDistributedConfig(cfg); e != nil {
		return nil, e
	}
	if e := identity.Validate(); e != nil {
		return nil, e
	}
	if identity.Seed != cfg.Training.Seed {
		return nil, fmt.Errorf("learner seed differs from config")
	}
	// Freeze caller-owned config without re-defaulting; reload runtime artifacts below.
	b, e := json.Marshal(cfg)
	if e != nil {
		return nil, e
	}
	var clone ArchConfig
	if e = json.Unmarshal(b, &clone); e != nil {
		return nil, e
	}
	cfg = &clone
	if e = configureDatasetForTraining(cfg, pattern, cfg.Name); e != nil {
		return nil, e
	}
	if cfg.CharVocabSize > 0 {
		if _, e = configureCharFeaturesForTraining(cfg, pattern); e != nil {
			return nil, e
		}
	}
	l, e := data.NewLearnerLoader(pattern, identity, cfg.SeqLen, effectiveShuffleChunkTokens(cfg), cfg.VocabSize)
	if e != nil {
		return nil, e
	}
	p, e := BuildIRProgramFromConfig(cfg)
	if e != nil {
		return nil, e
	}
	for _, op := range p.Ops {
		if op.Code == arch.OpRandomNormal {
			return nil, fmt.Errorf("local learner cannot replay unkeyed random_normal")
		}
	}
	shapes, e := computeWeightShapes(cfg)
	if e != nil {
		return nil, e
	}
	w, e := localModelData(initial, shapes)
	if e != nil {
		return nil, e
	}
	schedule, steps := buildTrainingScheduler(cfg.Training)
	if steps != cfg.Training.Steps || steps > math.MaxInt32/cfg.Training.Distributed.GradientAccumulationSteps {
		return nil, fmt.Errorf("unsupported learner step horizon")
	}
	if uint64(steps) > math.MaxUint64/uint64(cfg.Training.Distributed.GradientAccumulationSteps)/uint64(cfg.Training.BatchTokens) {
		return nil, fmt.Errorf("learner effective count horizon overflows")
	}
	policy := gpu.PrecisionPolicy{ComputeDType: "float32", Backend: "metal"}
	factory := testFactory
	if factory == nil {
		policy, e = trainingPrecision(cfg)
		if e != nil {
			return nil, e
		}
		if policy.Backend != "metal" {
			return nil, fmt.Errorf("R2 local learner requires Metal")
		}
		factory, e = makeLocalLearnerFactory(cfg, p, identity.RandomSeed())
		if e != nil {
			return nil, e
		}
	}
	// Bind the complete configured schedule, graph, model layout and metadata.
	b, e = json.Marshal(struct {
		Config       *ArchConfig
		Program      *arch.Program
		Shapes       []WeightShape
		CharFeatures []int32
	}{cfg, p, shapes, cfg.CharFeatureIDs})
	if e != nil {
		return nil, e
	}
	h := sha256.Sum256(b)
	engine, e := factory(w, false)
	if e != nil {
		return nil, e
	}
	return &LocalLearnerSession{cfg: cfg, loader: l, engine: engine, factory: factory, shapes: shapes, policy: policy, contract: hex.EncodeToString(h[:]), schedule: schedule}, nil
}

func localModelData(model []LocalLearnerTensor, shapes []WeightShape) ([][]float32, error) {
	if len(model) != len(shapes) {
		return nil, fmt.Errorf("global model tensor count mismatch")
	}
	w := make([][]float32, len(model))
	for i, t := range model {
		if t.Name != shapes[i].Name || !reflect.DeepEqual(t.Shape, shapes[i].Shape) || len(t.Data) != shapeProduct(t.Shape) {
			return nil, fmt.Errorf("global model tensor %d layout mismatch", i)
		}
		for _, v := range t.Data {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return nil, fmt.Errorf("global model tensor %d non-finite", i)
			}
		}
		w[i] = append([]float32(nil), t.Data...)
	}
	return w, nil
}

func localModel(weights [][]float32, shapes []WeightShape) []LocalLearnerTensor {
	out := make([]LocalLearnerTensor, len(shapes))
	for i, s := range shapes {
		out[i] = LocalLearnerTensor{s.Name, append([]int(nil), s.Shape...), append([]float32(nil), weights[i]...)}
	}
	return out
}

func (s *LocalLearnerSession) available() error {
	if s.engine == nil || s.failed {
		return fmt.Errorf("local learner closed or failed; restore a boundary snapshot")
	}
	return nil
}

func (s *LocalLearnerSession) InstallGlobal(model []LocalLearnerTensor) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if e := s.available(); e != nil {
		return e
	}
	w, e := localModelData(model, s.shapes)
	if e != nil {
		return e
	}
	return s.engine.install(w)
}

// Train performs exactly attempts optimizer attempts, each with K microsteps.
// A skip or interrupted microstep poisons this session until explicit restore;
// it never consumes extra batches to compensate for a skipped update.
func (s *LocalLearnerSession) Train(ctx context.Context, attempts int) (LocalLearnerResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out LocalLearnerResult
	if e := s.available(); e != nil {
		return out, e
	}
	if attempts < 1 || uint64(attempts) > uint64(s.cfg.Training.Steps)-s.attempt {
		return out, fmt.Errorf("invalid local attempt count")
	}
	for range attempts {
		if e := ctx.Err(); e != nil {
			return out, e
		}
		var numerator, denominator float64
		var examples uint64
		for micro := 0; micro < s.cfg.Training.Distributed.GradientAccumulationSteps; micro++ {
			if e := ctx.Err(); e != nil {
				s.failed = true
				return out, e
			}
			b, e := s.loader.NextBatch(s.cfg.Training.BatchTokens)
			if e != nil {
				s.failed = true
				return out, e
			}
			p, e := prepareObjectiveBatch(s.cfg, trainBatchFromDataBatch(b, nil), int(s.attempt), arch.ObjectiveCausal)
			if e != nil {
				s.failed = true
				return out, e
			}
			step := int(s.attempt)*s.cfg.Training.Distributed.GradientAccumulationSteps + micro
			loss, e := s.engine.step(p, step, s.schedule.At(int(s.attempt)))
			if e != nil {
				s.failed = true
				return out, e
			}
			if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
				s.failed = true
				return out, fmt.Errorf("non-finite local loss")
			}
			numerator += float64(loss) * float64(p.lossNormalizer)
			denominator += float64(p.lossNormalizer)
			examples += uint64(b.ExampleCount)
		}
		state, e := s.engine.stats()
		if e != nil {
			s.failed = true
			return out, e
		}
		out.Attempted++
		if state.AttemptedSteps != s.attempt+1 || state.CommittedSteps != s.attempt+1 || state.SkippedSteps != 0 || denominator <= 0 {
			s.failed = true
			out.Skipped++
			return out, fmt.Errorf("local learner requires exactly one committed update and zero skips per attempt")
		}
		if uint64(denominator) > ^uint64(0)-s.tokens || examples > ^uint64(0)-s.examples {
			s.failed = true
			return out, fmt.Errorf("learner effective count overflow")
		}
		s.attempt++
		s.tokens += uint64(denominator)
		s.examples += examples
		out.Committed++
		out.EffectiveTokens += uint64(denominator)
		out.EffectiveExamples += examples
		out.Loss += numerator / denominator
	}
	out.Loss /= float64(attempts)
	return out, nil
}

func (s *LocalLearnerSession) Close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.engine != nil {
		s.engine.close()
		s.engine = nil
	}
}
