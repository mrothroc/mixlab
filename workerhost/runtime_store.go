package workerhost

import (
	"bytes"
	"context"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"

	"github.com/mrothroc/mixlab/statehome"
)

const runtimeIndexFile = "runtime-attempts.json"
const runtimeIndexVersion = "mixlab_runtime_attempts_v1"

type runtimeEntry struct {
	Job     string `json:"job"`
	Attempt string `json:"attempt"`
	Ready   bool   `json:"ready"`
	Kind    string `json:"kind,omitempty"`
}
type runtimeIndex struct {
	Version string         `json:"version"`
	Entries []runtimeEntry `json:"entries"`
}

// RuntimeStore keeps allocation evidence outside per-attempt directories. Losing
// a published directory must not let a restarted agent create a fresh journal
// and launch the same job again. Initialization is administrator-local only.
type RuntimeStore struct{ root statehome.Path }

func InitializeRuntimeStore(root statehome.Path) (*RuntimeStore, error) {
	if root.Kind() != statehome.Worker {
		return nil, fmt.Errorf("worker runtime root required")
	}
	b, err := json.Marshal(runtimeIndex{runtimeIndexVersion, []runtimeEntry{}})
	if err != nil {
		return nil, err
	}
	if err := root.Publish(func(p statehome.Path) error { return p.CompareAndSwap(runtimeIndexFile, nil, b) }); err != nil {
		return nil, err
	}
	return OpenRuntimeStore(root)
}

func OpenRuntimeStore(root statehome.Path) (*RuntimeStore, error) {
	if root.Kind() != statehome.Worker {
		return nil, fmt.Errorf("worker runtime root required")
	}
	s := &RuntimeStore{root}
	_, _, err := s.load()
	if err != nil {
		return nil, err
	}
	return s, nil
}

func runtimeID(s string) bool {
	b, err := hex.DecodeString(s)
	return err == nil && len(b) == 16 && hex.EncodeToString(b) == s
}

func (s *RuntimeStore) load() ([]byte, runtimeIndex, error) {
	b, err := s.root.ReadFileLimit(runtimeIndexFile, 1<<20)
	if err != nil {
		return nil, runtimeIndex{}, err
	}
	var index runtimeIndex
	if err := json.Unmarshal(b, &index); err != nil {
		return nil, index, err
	}
	canonical, _ := json.Marshal(index)
	if !bytes.Equal(b, canonical) || index.Version != runtimeIndexVersion || index.Entries == nil || len(index.Entries) > 4096 {
		return nil, index, fmt.Errorf("invalid runtime allocation journal")
	}
	seen := make(map[string]bool)
	for _, e := range index.Entries {
		key := e.Kind + "/" + e.Job + "/" + e.Attempt
		if (e.Kind != "" && e.Kind != "learner") || !runtimeID(e.Job) || !runtimeID(e.Attempt) || seen[key] {
			return nil, index, fmt.Errorf("invalid runtime allocation entry")
		}
		seen[key] = true
	}
	return b, index, nil
}

// Attempt publishes an empty child context once, before returning its path to
// a launcher. Pending publication is retryable; a ready context is never rebuilt.
func (s *RuntimeStore) Attempt(ctx context.Context, job, attempt string) (out statehome.Path, err error) {
	return s.allocate(ctx, "", job, attempt)
}

// LearnerPrivate survives authorized attempts but never crosses run/worker
// identities. The allocation journal makes a missing published path fatal.
func (s *RuntimeStore) LearnerPrivate(ctx context.Context, run, worker string) (statehome.Path, error) {
	return s.allocate(ctx, "learner", run, worker)
}

func (s *RuntimeStore) allocate(ctx context.Context, kind, job, attempt string) (out statehome.Path, err error) {
	if !runtimeID(job) || !runtimeID(attempt) {
		return out, fmt.Errorf("canonical job and attempt IDs required")
	}
	err = s.root.WithProcessLock(ctx, "runtime-attempts.lock", func() error {
		old, index, err := s.load()
		if err != nil {
			return err
		}
		at := -1
		for i, e := range index.Entries {
			if e.Kind == kind && e.Job == job && e.Attempt == attempt {
				at = i
				break
			}
		}
		if at < 0 {
			if len(index.Entries) == 4096 {
				return fmt.Errorf("runtime allocation history capacity reached")
			}
			at = len(index.Entries)
			index.Entries = append(index.Entries, runtimeEntry{Job: job, Attempt: attempt, Kind: kind})
			b, err := json.Marshal(index)
			if err != nil {
				return err
			}
			if err := s.root.CompareAndSwap(runtimeIndexFile, old, b); err != nil {
				return err
			}
			old = b
		}
		name := job + "-" + attempt
		if kind != "" {
			name = kind + "-" + name
		}
		p, err := statehome.Resolve(statehome.Options{ExactDir: filepath.Join(s.root.Dir(), name)}, statehome.Context{Kind: statehome.Worker})
		if err != nil {
			return err
		}
		identity := []byte(job + "/" + attempt)
		if kind != "" {
			identity = []byte(kind + "/" + string(identity))
		}
		if !index.Entries[at].Ready {
			err := p.Publish(func(p statehome.Path) error {
				if err := p.CompareAndSwap("runtime-identity", nil, identity); err != nil {
					return err
				}
				if kind == "learner" {
					private, err := learnerPrivatePath(p)
					if err != nil {
						return err
					}
					return private.Ensure()
				}
				return nil
			})
			if err != nil && !errors.Is(err, statehome.ErrExists) {
				return err
			}
		}
		found, err := p.ReadFileLimit("runtime-identity", 128)
		if err != nil {
			return fmt.Errorf("published runtime context missing: %w", err)
		}
		if !bytes.Equal(found, identity) {
			return fmt.Errorf("runtime identity changed")
		}
		out = p
		if kind == "learner" {
			// Publish both ownership contexts atomically. Recovery validates the
			// learner child, never recreating lost private history.
			out, err = learnerPrivatePath(p)
			if err != nil {
				return err
			}
			if err := out.Validate(); err != nil {
				return fmt.Errorf("published learner private context missing or unsafe: %w", err)
			}
		}
		if !index.Entries[at].Ready {
			index.Entries[at].Ready = true
			b, err := json.Marshal(index)
			if err != nil {
				return err
			}
			if err := s.root.CompareAndSwap(runtimeIndexFile, old, b); err != nil {
				return err
			}
		}
		return nil
	})
	return out, err
}

func learnerPrivatePath(parent statehome.Path) (statehome.Path, error) {
	return statehome.Resolve(statehome.Options{ExactDir: filepath.Join(parent.Dir(), "private")}, statehome.Context{Kind: statehome.Worker})
}
