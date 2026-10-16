package learner

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"

	"github.com/mrothroc/mixlab/artifact"
	local "github.com/mrothroc/mixlab/artifact/local"
	"github.com/mrothroc/mixlab/artifact/tensors"
	private "github.com/mrothroc/mixlab/diloco/learner"
	"github.com/mrothroc/mixlab/diloco/optimizer"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/statehome"
)

type Store struct {
	path  statehome.Path
	blobs *local.Store
	limit uint64
	fault func(string) error
}

// Open never creates the worker-owned directory or repairs missing history.
// Hosting supplies the same private directory across authorized attempts.
func Open(path statehome.Path, limit uint64) (*Store, error) {
	if path.Kind() != statehome.Worker || limit == 0 || limit > 8<<30 {
		return nil, fmt.Errorf("bounded worker private context required")
	}
	if err := path.Validate(); err != nil {
		return nil, err
	}
	return &Store{path: path, limit: limit}, nil
}
func (s *Store) locked(ctx context.Context, fn func() error) error {
	return s.path.WithProcessLock(ctx, "learner.lock", func() error {
		var err error
		s.blobs, err = local.Open(s.path)
		if err != nil {
			return err
		}
		return fn()
	})
}
func ref(b []byte) artifact.Ref {
	h := sha256.Sum256(b)
	return artifact.Ref{SHA256: hex.EncodeToString(h[:]), Bytes: uint64(len(b))}
}
func (s *Store) budget(extra uint64) error {
	entries, err := os.ReadDir(s.path.Dir())
	if err != nil {
		return err
	}
	used := extra
	for _, e := range entries {
		info, err := e.Info()
		if err != nil {
			return err
		}
		if !info.Mode().IsRegular() {
			return fmt.Errorf("unexpected private directory entry")
		}
		if uint64(info.Size()) > s.limit || used > s.limit-uint64(info.Size()) {
			return fmt.Errorf("learner private disk budget exhausted")
		}
		used += uint64(info.Size())
	}
	return nil
}
func (s *Store) put(ctx context.Context, b []byte) (artifact.Ref, error) {
	r := ref(b)
	if err := r.Validate(); err != nil {
		return r, err
	}
	if f, err := s.blobs.OpenVerified(ctx, r); err == nil {
		_ = f.Close()
		return r, nil
	} else if !os.IsNotExist(err) {
		return r, err
	}
	if err := s.budget(r.Bytes); err != nil {
		return r, err
	}
	return r, s.blobs.PutVerified(ctx, r, bytes.NewReader(b))
}
func (s *Store) get(ctx context.Context, r artifact.Ref) ([]byte, error) {
	if r.Bytes > s.limit {
		return nil, fmt.Errorf("private artifact exceeds budget")
	}
	var b bytes.Buffer
	err := s.blobs.Copy(ctx, r, &b)
	return b.Bytes(), err
}
func encodeModel(v []optimizer.Tensor) ([]byte, error) {
	var b bytes.Buffer
	err := tensors.Write(&b, v)
	return b.Bytes(), err
}
func (s *Store) snapshot(ctx context.Context, a run.Assignment, id, stage string, v Snapshot, c Counts) (private.PrivateState, error) {
	h, err := run.Digest(a)
	if err != nil {
		return private.PrivateState{}, err
	}
	p := private.PrivateState{Format: "mixlab_learner_private_v1", Assignment: a, AssignmentHash: h, UpdateID: id, Stage: stage, Replay: v.Replay, Precision: v.Precision, Attempted: c.Attempted, Committed: c.Committed, Skipped: c.Skipped, EffectiveTokens: c.Tokens, EffectiveExamples: c.Examples}
	model, err := encodeModel(v.Model)
	if err != nil {
		return p, err
	}
	for _, part := range []struct {
		b []byte
		r *artifact.Ref
	}{{model, &p.Model}, {v.Optimizer, &p.Optimizer}, {v.Sampler, &p.Sampler}, {v.Schedule, &p.Schedule}, {v.Metadata, &p.TrainingMetadata}} {
		*part.r, err = s.put(ctx, part.b)
		if err != nil {
			return p, err
		}
	}
	return p, p.Validate()
}
func (s *Store) restore(ctx context.Context, p private.PrivateState) (Snapshot, error) {
	v := Snapshot{Replay: p.Replay, Precision: p.Precision}
	if err := p.Validate(); err != nil {
		return v, err
	}
	model, err := s.get(ctx, p.Model)
	if err != nil {
		return v, err
	}
	v.Model, err = tensors.Read(bytes.NewReader(model), p.Model.Bytes)
	if err != nil {
		return v, err
	}
	for _, part := range []struct {
		r artifact.Ref
		b *[]byte
	}{{p.Optimizer, &v.Optimizer}, {p.Sampler, &v.Sampler}, {p.Schedule, &v.Schedule}, {p.TrainingMetadata, &v.Metadata}} {
		*part.b, err = s.get(ctx, part.r)
		if err != nil {
			return v, err
		}
	}
	return v, nil
}

type journal struct {
	Format     string                `json:"format"`
	Assignment run.Assignment        `json:"assignment"`
	Start      private.PrivateState  `json:"start"`
	Completed  *private.PrivateState `json:"completed,omitempty"`
	Update     *private.Update       `json:"update,omitempty"`
	Marker     *private.Completion   `json:"marker,omitempty"`
	Receipt    *private.Receipt      `json:"receipt,omitempty"`
	Evaluation *run.EvaluationResult `json:"evaluation,omitempty"`
}

func (j journal) Validate() error {
	if j.Evaluation != nil {
		e := j.Evaluation
		if err := e.Validate(); err != nil {
			return err
		}
		if j.Marker == nil || e.PrivateBeforeHash != j.Marker.PrivateHash || e.Assignment.Binding != j.Assignment.Binding || e.Assignment.Round != j.Assignment.Round || e.Assignment.WorkerID != j.Assignment.WorkerID {
			return fmt.Errorf("evaluation differs from completed private state")
		}
	}
	if j.Format != "mixlab_learner_journal_v1" {
		return fmt.Errorf("unknown learner journal")
	}
	if err := j.Start.Validate(); err != nil {
		return err
	}
	if j.Start.Assignment != j.Assignment || j.Start.Stage != "start" {
		return fmt.Errorf("journal start differs from assignment")
	}
	if j.Completed != nil || j.Update != nil || j.Marker != nil {
		if j.Completed == nil || j.Update == nil || j.Marker == nil {
			return fmt.Errorf("partial completion marker")
		}
		if err := j.Marker.ValidateFor(*j.Completed, *j.Update); err != nil {
			return err
		}
		if j.Completed.Assignment != j.Assignment || j.Completed.UpdateID != j.Start.UpdateID {
			return fmt.Errorf("completion changed assignment")
		}
	}
	if j.Receipt != nil {
		if err := j.Receipt.Validate(); err != nil {
			return err
		}
		if j.Marker == nil || j.Receipt.UpdateHash != j.Marker.UpdateHash || j.Receipt.PrivateHash != j.Marker.PrivateHash || j.Receipt.UpdateID != j.Update.ID || j.Receipt.Binding != j.Update.Binding {
			return fmt.Errorf("receipt differs from completed work")
		}
	}
	return nil
}
func (s *Store) load() (b []byte, j journal, err error) {
	b, err = s.path.ReadFileLimit("head.json", 4<<20)
	if err != nil {
		return
	}
	err = json.Unmarshal(b, &j)
	if err != nil {
		return
	}
	canonical, e := json.Marshal(j)
	if e != nil || !bytes.Equal(b, canonical) {
		err = fmt.Errorf("noncanonical learner journal")
		return
	}
	err = j.Validate()
	return
}
func (s *Store) save(old []byte, j journal, point string) error {
	if err := j.Validate(); err != nil {
		return err
	}
	b, err := json.Marshal(j)
	if err != nil {
		return err
	}
	if len(b) > 4<<20 {
		return fmt.Errorf("learner journal too large")
	}
	if err = s.budget(uint64(len(b))); err != nil {
		return err
	}
	if err = s.path.CompareAndSwap("head.json", old, b); err != nil {
		return err
	}
	if s.fault != nil {
		return s.fault(point)
	}
	return nil
}
