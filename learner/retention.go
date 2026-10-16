package learner

import (
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"

	"github.com/mrothroc/mixlab/artifact"
	private "github.com/mrothroc/mixlab/diloco/learner"
)

// collect runs only under the learner lifecycle lock, after a validated journal
// was read or durably replaced. An incomplete window retains its start; a
// completed window retains BOTH boundaries and its exact pending delta. The
// successor assignment proves global commit before the previous set is released.
func (s *Store) collect(j *journal) error {
	keep := map[string]bool{}
	add := func(p private.PrivateState) {
		for _, r := range []artifact.Ref{p.Model, p.Optimizer, p.Sampler, p.Schedule, p.TrainingMetadata} {
			keep[r.SHA256] = true
		}
	}
	if j != nil {
		add(j.Start)
		if j.Completed != nil {
			add(*j.Completed)
			keep[j.Update.Delta.Ref.SHA256] = true
		}
	}
	entries, err := os.ReadDir(s.path.Dir())
	if err != nil {
		return err
	}
	for _, entry := range entries {
		name := entry.Name()
		if name == "head.json" || name == "learner.lock" || keep[name] {
			continue
		}
		b, e := hex.DecodeString(name)
		if e != nil || len(b) != 32 || hex.EncodeToString(b) != name {
			return fmt.Errorf("unexpected learner private file %q", name)
		}
		f, e := s.path.OpenRead(name)
		if e != nil {
			return e
		}
		if e = f.Close(); e != nil {
			return e
		}
		if e = os.Remove(filepath.Join(s.path.Dir(), name)); e != nil {
			return e
		}
	}
	return nil
}
