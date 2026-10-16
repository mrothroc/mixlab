package admission

import (
	"bytes"
	"context"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"

	"github.com/mrothroc/mixlab/recruitment"
	"github.com/mrothroc/mixlab/statehome"
)

const maxSubmissions = 1024

// Submission is the immutable, service-owned admission intent. Owner is the
// authenticated submitting principal; Launch.Controller is the service's
// separately enrolled admission identity. Neither carries a private key/path.
type Submission struct {
	Owner         string                 `json:"owner"`
	Launch        recruitment.LaunchPlan `json:"launch"`
	Specification Specification          `json:"specification"`
	Deadline      int64                  `json:"deadline"`
}

func (q Submission) Validate() error {
	for _, id := range []string{q.Owner, q.Launch.Run, q.Launch.Attempt, q.Launch.Controller} {
		b, err := hex.DecodeString(id)
		if err != nil || len(b) != 16 || hex.EncodeToString(b) != id {
			return fmt.Errorf("canonical submission identity required")
		}
	}
	r := record{Format: version, Request: q.Launch, Specification: q.Specification, Deadline: q.Deadline, Nodes: make([]node, len(q.Launch.Selection.Selected))}
	initializeReservations(&r)
	return r.validate()
}

// Inbox owns durable acceptance and discovery, separate from coordinator state.
// Reopening never creates a missing root. Accepted records are not deleted by
// terminal-client cancellation or service shutdown.
type Inbox struct{ path statehome.Path }

func OpenInbox(path statehome.Path) (*Inbox, error) {
	if path.Kind() != statehome.Principal {
		return nil, fmt.Errorf("admission-owned context required")
	}
	if err := path.Validate(); err != nil {
		return nil, err
	}
	return &Inbox{path: path}, nil
}

func (i *Inbox) Accept(ctx context.Context, q Submission) error {
	if err := q.Validate(); err != nil {
		return err
	}
	b, err := json.Marshal(q)
	if err != nil || len(b) > 8<<20 {
		return fmt.Errorf("submission exceeds durable limit")
	}
	return i.path.WithProcessLock(ctx, "inbox.lock", func() error {
		name := q.Launch.Run + ".submission"
		old, err := i.path.ReadFileLimit(name, 8<<20)
		if err == nil {
			if !bytes.Equal(old, b) {
				return fmt.Errorf("run ID already belongs to another submission")
			}
			return nil
		}
		if !errors.Is(err, os.ErrNotExist) {
			return err
		}
		ids, err := i.IDs(ctx)
		if err != nil {
			return err
		}
		if len(ids) >= maxSubmissions {
			return fmt.Errorf("admission inbox full; retain/archive completed runs before new submission")
		}
		return i.path.CompareAndSwap(name, nil, b)
	})
}

func (i *Inbox) IDs(ctx context.Context) ([]string, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := i.path.Validate(); err != nil {
		return nil, err
	}
	entries, err := os.ReadDir(i.path.Dir())
	if err != nil {
		return nil, err
	}
	var ids []string
	for _, e := range entries {
		if filepath.Ext(e.Name()) != ".submission" {
			continue
		}
		id := e.Name()[:len(e.Name())-len(".submission")]
		if _, err := i.Read(ctx, id); err != nil {
			return nil, err
		}
		ids = append(ids, id)
		if len(ids) > maxSubmissions {
			return nil, fmt.Errorf("admission inbox exceeds limit")
		}
	}
	sort.Strings(ids)
	return ids, nil
}

func (i *Inbox) Read(ctx context.Context, id string) (Submission, error) {
	var q Submission
	if err := ctx.Err(); err != nil {
		return q, err
	}
	decoded, err := hex.DecodeString(id)
	if err != nil || len(decoded) != 16 || hex.EncodeToString(decoded) != id {
		return q, fmt.Errorf("canonical run ID required")
	}
	b, err := i.path.ReadFileLimit(id+".submission", 8<<20)
	if err != nil {
		return q, err
	}
	if err := json.Unmarshal(b, &q); err != nil {
		return q, err
	}
	want, err := json.Marshal(q)
	if err != nil || !bytes.Equal(want, b) || q.Launch.Run != id {
		return q, fmt.Errorf("noncanonical or substituted submission")
	}
	return q, q.Validate()
}

// Resume publishes the saga only after its immutable intent is accepted. A
// crash between these writes is replayable without repeating recruitment.
func (i *Inbox) Resume(ctx context.Context, id string) (*Saga, error) {
	q, err := i.Read(ctx, id)
	if err != nil {
		return nil, err
	}
	p, err := statehome.Resolve(statehome.Options{ExactDir: filepath.Join(i.path.Dir(), id)}, statehome.Context{Kind: statehome.Principal})
	if err != nil {
		return nil, err
	}
	var s *Saga
	err = i.path.WithProcessLock(ctx, "inbox.lock", func() error {
		var err error
		s, err = Open(p)
		if errors.Is(err, os.ErrNotExist) {
			// An existing directory with missing journal is corruption, not a
			// fresh intent. Never repair it into an apparently unstarted run.
			if _, statErr := os.Lstat(p.Dir()); !errors.Is(statErr, os.ErrNotExist) {
				return fmt.Errorf("accepted admission state is missing its journal")
			}
			s, err = Begin(p, q.Launch, q.Specification, q.Deadline)
		}
		if err != nil {
			return err
		}
		v, err := s.Snapshot(ctx)
		if err != nil {
			return err
		}
		got, _ := json.Marshal(Submission{Owner: q.Owner, Launch: v.Request, Specification: v.Specification, Deadline: v.Deadline})
		want, _ := json.Marshal(q)
		if !bytes.Equal(got, want) {
			return fmt.Errorf("admission saga differs from accepted submission")
		}
		return nil
	})
	return s, err
}
