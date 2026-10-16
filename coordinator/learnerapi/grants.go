package learnerapi

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"

	"github.com/mrothroc/mixlab/diloco/authenticated"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/statehome"
)

// Grants durably replays exact signed envelopes: P3's use ledger binds proof
// bytes, not only the semantic grant. Refresh creates a distinct generation's
// grant; upload identity and content identity remain unchanged.
type Grants struct{ path statehome.Path }

func OpenGrants(path statehome.Path) (*Grants, error) {
	if err := path.Validate(); err != nil {
		return nil, err
	}
	if _, err := path.CleanupTemporaryFiles(); err != nil {
		return nil, err
	}
	return &Grants{path}, nil
}
func (s *Grants) issue(ctx context.Context, b run.Grant, sign func() (authenticated.Grant, error)) (out authenticated.Grant, err error) {
	if err = b.Validate(); err != nil {
		return out, err
	}
	key, err := run.Digest(b.ID)
	if err != nil {
		return out, err
	}
	err = s.path.WithProcessLock(ctx, key+".lock", func() error {
		data, e := s.path.ReadFileLimit(key+".grant", MaxMessage)
		if e == nil {
			if e = authenticated.Decode(data, &out); e != nil {
				return e
			}
			if out.Body != b {
				return fmt.Errorf("grant ID rebound")
			}
			return nil
		}
		if !errors.Is(e, os.ErrNotExist) {
			return e
		}
		out, e = sign()
		if e != nil {
			return e
		}
		if e = out.Validate(); e != nil {
			return e
		}
		if out.Body != b {
			return fmt.Errorf("signer changed grant")
		}
		data, e = json.Marshal(out)
		if e != nil {
			return e
		}
		if len(data) > MaxMessage {
			return fmt.Errorf("grant too large")
		}
		return s.path.CompareAndSwap(key+".grant", nil, data)
	})
	return out, err
}
