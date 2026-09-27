package principal

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"time"

	"github.com/mrothroc/mixlab/trust/enrollment"
)

const renewalFile = "principal-renewal.json"

type pendingRenewal struct {
	Chain   [][]byte                        `json:"chain"`
	Request enrollment.SignedRenewalRequest `json:"request"`
}

// BeginRenewal persists the exact signed request before any network effect.
// Retry after a lost response or process restart uses the same nonce. Once the
// credential is atomically installed, a later renewal can replace this journal.
func (s *Store) BeginRenewal(ctx context.Context, now time.Time) (out enrollment.SignedRenewalRequest, err error) {
	err = s.path.WithProcessLock(ctx, lockname, func() error {
		r, k, err := s.Active(now)
		if err != nil {
			return err
		}
		old, err := s.path.ReadFileLimit(renewalFile, 64<<10)
		if err != nil && !errors.Is(err, os.ErrNotExist) {
			return err
		}
		if err == nil {
			var p pendingRenewal
			if err := json.Unmarshal(old, &p); err != nil {
				return err
			}
			again, err := json.Marshal(p)
			if err != nil || !bytes.Equal(old, again) || p.Request.ValidateFor(p.Chain) != nil {
				return fmt.Errorf("invalid persisted principal renewal")
			}
			if p.Request.Request.Cluster != r.Cluster || p.Request.Request.Principal != r.Principal {
				return fmt.Errorf("renewal journal identity mismatch")
			}
			if bytes.Equal(p.Chain[0], r.Chain[0]) {
				out = p.Request
				return nil
			}
		}
		out, err = enrollment.NewRenewalRequest(r.Chain, k)
		if err != nil {
			return err
		}
		b, err := json.Marshal(pendingRenewal{r.Chain, out})
		if err != nil {
			return err
		}
		return s.path.CompareAndSwap(renewalFile, old, b)
	})
	return out, err
}
