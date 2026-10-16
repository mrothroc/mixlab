//go:build mlx && cgo && darwin

package integration

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/artifact"
	"github.com/mrothroc/mixlab/artifact/tensors"
	"github.com/mrothroc/mixlab/data"
	private "github.com/mrothroc/mixlab/diloco/learner"
	"github.com/mrothroc/mixlab/diloco/optimizer"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/gpu"
	"github.com/mrothroc/mixlab/learner"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/train"
	wc "github.com/mrothroc/mixlab/workercontrol"
)

// Mirrors cmd/mixlab's composition adapter without importing main. The engine
// receives only tensors, config and logical sampler identity, never TLS ports.
type relayMetalEngine struct {
	session *train.LocalLearnerSession
	calls   *int
}

type relayMetalMetadata struct {
	Format   string `json:"format"`
	Contract string `json:"contract"`
	Attempt  uint64 `json:"attempt"`
	Tokens   uint64 `json:"tokens"`
	Examples uint64 `json:"examples"`
}

func (e *relayMetalEngine) Snapshot() (learner.Snapshot, error) {
	v, err := e.session.Snapshot()
	if err != nil {
		return learner.Snapshot{}, err
	}
	id := v.Sampler.Identity
	out := learner.Snapshot{Precision: v.Precision, Replay: private.Replay{Version: "mixlab_learner_rng_v1", RunID: id.RunID, WorkerID: id.WorkerID, PartitionID: id.PartitionID, Seed: id.Seed, AttemptStep: v.Attempt}}
	for _, tensor := range v.Model {
		out.Model = append(out.Model, optimizer.Tensor{Name: tensor.Name, Shape: tensor.Shape, Values: tensor.Data})
	}
	for _, part := range []struct {
		value any
		dst   *[]byte
	}{
		{v.Optimizer, &out.Optimizer}, {v.Sampler, &out.Sampler}, {v.Contract, &out.Schedule},
		{relayMetalMetadata{v.Format, v.Contract, v.Attempt, v.EffectiveTokens, v.EffectiveExamples}, &out.Metadata},
	} {
		*part.dst, err = json.Marshal(part.value)
		if err != nil {
			return learner.Snapshot{}, err
		}
	}
	return out, nil
}

func (e *relayMetalEngine) Restore(v learner.Snapshot) error {
	var meta relayMetalMetadata
	if err := json.Unmarshal(v.Metadata, &meta); err != nil {
		return err
	}
	var contract string
	if err := json.Unmarshal(v.Schedule, &contract); err != nil {
		return err
	}
	if contract != meta.Contract {
		return fmt.Errorf("private schedule contract mismatch")
	}
	s := train.LocalLearnerSnapshot{Format: meta.Format, Contract: meta.Contract, Attempt: meta.Attempt, EffectiveTokens: meta.Tokens, EffectiveExamples: meta.Examples, Precision: v.Precision, Model: relayLocalTensors(v.Model)}
	if err := json.Unmarshal(v.Optimizer, &s.Optimizer); err != nil {
		return err
	}
	if err := json.Unmarshal(v.Sampler, &s.Sampler); err != nil {
		return err
	}
	id := s.Sampler.Identity
	want := private.Replay{Version: "mixlab_learner_rng_v1", RunID: id.RunID, WorkerID: id.WorkerID, PartitionID: id.PartitionID, Seed: id.Seed, AttemptStep: s.Attempt}
	if v.Replay != want {
		return fmt.Errorf("private RNG and sampler boundary differ")
	}
	if err := e.session.Restore(s); err != nil {
		return err
	}
	again, err := e.Snapshot()
	if err == nil && !reflect.DeepEqual(v, again) {
		return fmt.Errorf("native restore changed private snapshot")
	}
	return err
}

func (e *relayMetalEngine) Install(v []optimizer.Tensor) error {
	before, err := e.Snapshot()
	if err != nil {
		return err
	}
	if err = e.session.InstallGlobal(relayLocalTensors(v)); err != nil {
		return err
	}
	after, err := e.Snapshot()
	before.Model, after.Model = nil, nil
	if err == nil && !reflect.DeepEqual(before, after) {
		return fmt.Errorf("global install reset private AdamW/sampler/counters")
	}
	return err
}

func (e *relayMetalEngine) Train(ctx context.Context, n uint64) (learner.Counts, error) {
	if n > math.MaxInt {
		return learner.Counts{}, fmt.Errorf("inner attempt count overflows")
	}
	*e.calls++
	v, err := e.session.Train(ctx, int(n))
	return learner.Counts{Attempted: v.Attempted, Committed: v.Committed, Skipped: v.Skipped, Tokens: v.EffectiveTokens, Examples: v.EffectiveExamples, Loss: v.Loss}, err
}

func (e *relayMetalEngine) Close() { e.session.Close() }

func relayLocalTensors(v []optimizer.Tensor) []train.LocalLearnerTensor {
	out := make([]train.LocalLearnerTensor, len(v))
	for i, tensor := range v {
		out[i] = train.LocalLearnerTensor{Name: tensor.Name, Shape: tensor.Shape, Data: tensor.Values}
	}
	return out
}

func TestLearnerRelayMetalIPCRecovery(t *testing.T) {
	if !gpu.Available() {
		t.Skip("Metal unavailable")
	}
	cfg, err := arch.ParseArchConfig([]byte(`{"model_dim":8,"vocab_size":16,"seq_len":4,"dropout":0.1,"blocks":[{"type":"plain","heads":2}],"training":{"objective":"causal","optimizer":"adamw","steps":4,"batch_tokens":8,"seed":19,"lr":0.001,"weight_decay":0,"distributed":{"mode":"diloco","inner_steps":2,"outer_optimizer":"nesterov","outer_lr":1,"outer_momentum":0,"aggregation":"uniform","gradient_accumulation_steps":1}}}`), "relay-metal")
	check(t, err)
	root, err := filepath.EvalSymlinks(t.TempDir())
	check(t, err)
	path := filepath.Join(root, "train.bin")
	b := make([]byte, 1024+513*2)
	binary.LittleEndian.PutUint32(b, 20240520)
	binary.LittleEndian.PutUint32(b[4:], 1)
	binary.LittleEndian.PutUint32(b[8:], 513)
	for i := range 513 {
		binary.LittleEndian.PutUint16(b[1024+i*2:], uint16((i*7+3)%16))
	}
	check(t, os.WriteFile(path, b, 0600))
	metas, err := arch.CollectWeightShapesFromConfig(cfg)
	check(t, err)
	rng := rand.New(rand.NewSource(19))
	model := make([]optimizer.Tensor, len(metas))
	for i, m := range metas {
		n := 1
		for _, d := range m.Shape {
			n *= d
		}
		values := make([]float32, n)
		for j := range values {
			if m.IsNormScale || m.InitOne {
				values[j] = 1
			} else if len(m.Shape) > 1 {
				values[j] = float32(rng.NormFloat64() * .05)
			}
		}
		model[i] = optimizer.Tensor{Name: m.Name, Shape: m.Shape, Values: values}
	}
	policy, err := gpu.ResolvePrecisionPolicy("float32", "metal", nil, os.LookupEnv)
	check(t, err)
	x := newLearnerRelayModel(t, model, policy)
	identity := func(i int) data.LearnerIdentity {
		a := x.s.Assignments[i]
		return data.LearnerIdentity{RunID: a.Binding.RunID, WorkerID: a.WorkerID, PartitionID: a.PartitionID, Index: i, Count: 2, Seed: 19}
	}
	var frames atomic.Uint64
	evaluations := 0
	runLearnerRelayRecovery(t, x, relayRuntimeScenario{
		factory: func(i int, calls *int) learner.Factory {
			return func(m []optimizer.Tensor) (learner.Engine, error) {
				s, err := train.NewLocalLearnerSession(cfg, path, identity(i), relayLocalTensors(m))
				if err != nil {
					return nil, err
				}
				return &relayMetalEngine{session: s, calls: calls}, nil
			}
		},
		evaluate: func(_ context.Context, a run.EvaluationAssignment, m []optimizer.Tensor) (float64, uint64, error) {
			evaluations++
			if reflect.DeepEqual(m, model) {
				t.Fatal("native training did not change the global model")
			}
			s, err := train.NewLocalLearnerSession(cfg, path, identity(0), relayLocalTensors(m))
			check(t, err)
			defer s.Close()
			loader, err := data.NewLearnerLoader(path, identity(0), cfg.SeqLen, cfg.SeqLen, cfg.VocabSize)
			check(t, err)
			var batches []data.Batch
			for range a.Batches {
				batch, err := loader.NextBatch(int(a.BatchTokens))
				check(t, err)
				batches = append(batches, batch)
			}
			before, err := s.Snapshot()
			check(t, err)
			loss, err := s.EvaluateGlobal(relayLocalTensors(m), batches)
			check(t, err)
			after, err := s.Snapshot()
			check(t, err)
			if !reflect.DeepEqual(before, after) || loss <= 0 || math.IsNaN(loss) || math.IsInf(loss, 0) {
				t.Fatal("native evaluation changed private state or produced invalid loss")
			}
			return loss, a.BatchTokens * a.Batches, nil
		},
		verify: func(i int, p statehome.Path, v private.PrivateState) {
			read := func(ref artifact.Ref) []byte {
				b, err := p.ReadFileLimit(ref.SHA256, 1<<20)
				check(t, err)
				return b
			}
			m, err := tensors.Read(bytes.NewReader(read(v.Model)), v.Model.Bytes)
			check(t, err)
			snapshot := learner.Snapshot{Model: m, Optimizer: read(v.Optimizer), Sampler: read(v.Sampler), Schedule: read(v.Schedule), Metadata: read(v.TrainingMetadata), Replay: v.Replay, Precision: v.Precision}
			s, err := train.NewLocalLearnerSession(cfg, path, identity(i), relayLocalTensors(model))
			check(t, err)
			defer s.Close()
			e := &relayMetalEngine{session: s}
			check(t, e.Restore(snapshot))
			native, err := s.Snapshot()
			check(t, err)
			if native.Attempt != 4 || native.EffectiveTokens != 32 || native.EffectiveExamples != 8 || native.Optimizer.Optimizer.CommittedSteps != 4 {
				t.Fatal("native AdamW/sampler counters did not survive both rounds")
			}
			nonzero := map[gpu.OptimizerStateKind]bool{}
			for _, tensor := range native.Optimizer.Tensors {
				for _, value := range tensor.Data {
					if value != 0 {
						nonzero[tensor.Kind] = true
					}
				}
			}
			if !nonzero[gpu.OptimizerStateAdamM] || !nonzero[gpu.OptimizerStateAdamV] {
				t.Fatal("real AdamW first/second moments absent")
			}
		},
		observe: func(e wc.Envelope) {
			frames.Add(1)
			var v any
			if err := json.Unmarshal(e.Payload, &v); err != nil {
				t.Error(err)
				return
			}
			var walk func(any)
			walk = func(v any) {
				switch v := v.(type) {
				case map[string]any:
					for key, child := range v {
						switch strings.ToLower(key) {
						case "workload", "grant", "proof", "private_key", "certificate", "credentials", "chain", "signature":
							t.Errorf("remote credential/capability field crossed local IPC: %s", key)
						}
						walk(child)
					}
				case []any:
					for _, child := range v {
						walk(child)
					}
				}
			}
			walk(v)
		},
	})
	if evaluations != 2 || frames.Load() < 50 {
		t.Fatal("missing native evaluation or observed IPC traffic", evaluations, frames.Load())
	}
}
