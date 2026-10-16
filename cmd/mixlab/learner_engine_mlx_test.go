//go:build mlx && cgo && darwin

package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"path/filepath"
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
)

type metalPeer struct {
	models  map[string][]byte
	deltas  map[string][]byte
	updates []private.Update
	lose    bool
}

func (p *metalPeer) Read(_ context.Context, _ run.Assignment, b run.Blob, w io.Writer) error {
	_, err := w.Write(p.models[b.Ref.SHA256])
	return err
}
func (p *metalPeer) Write(_ context.Context, _ run.Assignment, b run.Blob, r io.Reader) error {
	v, err := io.ReadAll(r)
	p.deltas[b.Ref.SHA256] = v
	return err
}
func (p *metalPeer) Start(context.Context, private.Start) error { return nil }
func (p *metalPeer) Submit(_ context.Context, u private.Update) (private.Receipt, error) {
	p.updates = append(p.updates, u)
	if p.lose {
		p.lose = false
		return private.Receipt{}, io.ErrUnexpectedEOF
	}
	h, _ := run.Digest(u)
	return private.Receipt{Format: "mixlab_update_receipt_v1", Binding: u.Binding, UpdateID: u.ID, UpdateHash: h, PrivateHash: u.PrivateHash}, nil
}
func (p *metalPeer) model(t *testing.T, m []optimizer.Tensor) run.Blob {
	t.Helper()
	var b bytes.Buffer
	if err := tensors.Write(&b, m); err != nil {
		t.Fatal(err)
	}
	h := sha256.Sum256(b.Bytes())
	ref := artifact.Ref{SHA256: hex.EncodeToString(h[:]), Bytes: uint64(b.Len())}
	p.models[ref.SHA256] = b.Bytes()
	return run.Blob{Ref: ref, Kind: "model_f32"}
}

type interruptedMetal struct {
	learner.Engine
	fail  *bool
	calls *int
}

func (e *interruptedMetal) Train(ctx context.Context, n uint64) (learner.Counts, error) {
	*e.calls++
	if *e.fail {
		*e.fail = false
		if _, err := e.Engine.Train(ctx, 1); err != nil {
			return learner.Counts{}, err
		}
		return learner.Counts{}, context.Canceled
	}
	return e.Engine.Train(ctx, n)
}
func TestManagedLearnerMetalDurableRecovery(t *testing.T) {
	if !gpu.Available() {
		t.Skip("Metal unavailable")
	}
	cfg, err := arch.ParseArchConfig([]byte(`{"model_dim":8,"vocab_size":16,"seq_len":4,"dropout":0.1,"blocks":[{"type":"plain","heads":2}],"training":{"objective":"causal","optimizer":"adamw","steps":8,"batch_tokens":8,"seed":19,"lr":0.001,"weight_decay":0,"distributed":{"mode":"diloco","inner_steps":2,"outer_optimizer":"nesterov","outer_lr":1,"outer_momentum":0,"aggregation":"uniform","gradient_accumulation_steps":2}}}`), "managed-metal")
	if err != nil {
		t.Fatal(err)
	}
	root, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(root, "train.bin")
	b := make([]byte, 1024+513*2)
	binary.LittleEndian.PutUint32(b, 20240520)
	binary.LittleEndian.PutUint32(b[4:], 1)
	binary.LittleEndian.PutUint32(b[8:], 513)
	for i := 0; i < 513; i++ {
		binary.LittleEndian.PutUint16(b[1024+i*2:], uint16((i*7+3)%16))
	}
	if err = os.WriteFile(path, b, 0600); err != nil {
		t.Fatal(err)
	}
	metas, err := arch.CollectWeightShapesFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(19))
	model := make([]optimizer.Tensor, len(metas))
	for i, m := range metas {
		n := 1
		for _, d := range m.Shape {
			n *= d
		}
		v := make([]float32, n)
		for j := range v {
			if m.IsNormScale || m.InitOne {
				v[j] = 1
			} else if len(m.Shape) > 1 {
				v[j] = float32(rng.NormFloat64() * .05)
			}
		}
		model[i] = optimizer.Tensor{Name: fmt.Sprintf("w%d_%s", i, m.Name), Shape: m.Shape, Values: v}
	}
	fixture, err := os.ReadFile("../../diloco/testdata/assignment.json")
	if err != nil {
		t.Fatal(err)
	}
	for worker := 0; worker < 2; worker++ {
		t.Run(fmt.Sprintf("worker%d", worker), func(t *testing.T) {
			var a run.Assignment
			if err = json.Unmarshal(fixture, &a); err != nil {
				t.Fatal(err)
			}
			a.WorkerID = fmt.Sprintf("w%d", worker)
			a.PartitionID = fmt.Sprintf("p%d", worker)
			a.InnerSteps = 2
			policy, e := gpu.ResolvePrecisionPolicy("float32", "metal", nil, os.LookupEnv)
			if e != nil {
				t.Fatal(e)
			}
			a.PrecisionHash, e = policy.Digest()
			if e != nil {
				t.Fatal(e)
			}
			peer := &metalPeer{models: map[string][]byte{}, deltas: map[string][]byte{}}
			a.BaseModel = peer.model(t, model)
			identity := data.LearnerIdentity{RunID: a.Binding.RunID, WorkerID: a.WorkerID, PartitionID: a.PartitionID, Index: worker, Count: 2, Seed: 19}
			// Compare uninterrupted and replayed engines from identical logical RNG,
			// sampler and model state. Hosting attempts are intentionally absent.
			var reference []optimizer.Tensor
			for _, replay := range []bool{false, true} {
				p, e := statehome.Resolve(statehome.Options{ExactDir: filepath.Join(root, fmt.Sprintf("worker%d-%t", worker, replay))}, statehome.Context{Kind: statehome.Worker})
				if e != nil {
					t.Fatal(e)
				}
				if e = p.Ensure(); e != nil {
					t.Fatal(e)
				}
				store, e := learner.Open(p, 16<<20)
				if e != nil {
					t.Fatal(e)
				}
				fail, calls := replay, 0
				r := learner.Runtime{Store: store, Peer: peer, Now: func() int64 { return 1300 }, Scope: learner.Scope{RunID: a.Binding.RunID, PlanHash: a.Binding.PlanHash, CohortHash: a.Binding.CohortHash, WorkerID: a.WorkerID, PartitionID: a.PartitionID, ParticipantID: fmt.Sprintf("n%d", worker), IslandID: fmt.Sprintf("i%d", worker), Contract: a.Contract, Coordinator: a.Coordinator}}
				r.Factory = func(m []optimizer.Tensor) (learner.Engine, error) {
					local, e := train.LocalLearnerModelFromWire(m)
					if e != nil {
						return nil, e
					}
					s, e := train.NewLocalLearnerSession(cfg, path, identity, local)
					if e != nil {
						return nil, e
					}
					snap, e := s.Snapshot()
					if e != nil {
						s.Close()
						return nil, e
					}
					hash, e := snap.Precision.Digest()
					if e != nil || hash != a.PrecisionHash {
						s.Close()
						return nil, fmt.Errorf("precision: %s %s %v", hash, a.PrecisionHash, e)
					}
					return &interruptedMetal{Engine: &learnerEngine{session: s}, fail: &fail, calls: &calls}, nil
				}
				if replay {
					if _, e = r.Execute(context.Background(), a); !errors.Is(e, context.Canceled) {
						t.Fatal("expected interrupted inner window", e)
					}
				}
				peer.lose = true
				if _, e = r.Execute(context.Background(), a); !errors.Is(e, io.ErrUnexpectedEOF) {
					t.Fatal("expected lost receipt", e)
				}
				trained := calls
				receipt, e := r.Execute(context.Background(), a)
				if e != nil || calls != trained {
					t.Fatal("receipt replay reran training", calls, trained, e)
				}
				last := peer.updates[len(peer.updates)-1]
				delta, e := tensors.Read(bytes.NewReader(peer.deltas[last.Delta.Ref.SHA256]), last.Delta.Ref.Bytes)
				if e != nil {
					t.Fatal(e)
				}
				if replay {
					for i := range delta {
						for j, v := range delta[i].Values {
							if math.Abs(float64(v-reference[i].Values[j])) > 1e-6 {
								t.Fatal("native replay delta differs", i, j)
							}
						}
					}
				} else {
					reference = delta
				}
				next := a
				next.ID = "round1"
				next.Round = 1
				next.BaseRound = 1
				next.PreviousPrivateHash = receipt.PrivateHash
				if _, e = r.Execute(context.Background(), next); e != nil {
					t.Fatal("successor failed", e)
				}
				if calls != trained+1 {
					t.Fatal("successor did not train exactly once")
				}
			}
		})
	}
}
