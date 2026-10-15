// Package recoveryauth verifies node recovery evidence before the coordinator
// changes its operational attempt binding. Transport adapters authenticate the
// caller separately; this is not a bearer authorization or a new allocation.
package recoveryauth

import (
	"context"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/trust"
)

type Authorization struct {
	Anchor      trust.Anchor
	Current     func(context.Context) (trust.VerifiedSnapshot, error)
	Clock       func() time.Time
	Coordinator run.Coordinator
	Evidence    nodejob.SignedLearnerRecovery
	// Confirm checks that node management still has this exact pending fence
	// under the live original lease. A stale signed proof is insufficient.
	Confirm func(context.Context, nodejob.LearnerRecovery) error
}

func (a Authorization) Check(ctx context.Context, want run.AttemptBinding) error {
	if a.Current == nil || a.Clock == nil || a.Confirm == nil || a.Evidence.Body.Attempt != want || a.Evidence.Body.Coordinator != a.Coordinator {
		return fmt.Errorf("recovery authorization binding mismatch")
	}
	v, err := a.Current(ctx)
	if err != nil {
		return err
	}
	if err = a.Evidence.Verify(a.Anchor, v, a.Clock()); err != nil {
		return err
	}
	return a.Confirm(ctx, a.Evidence.Body)
}
