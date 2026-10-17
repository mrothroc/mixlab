package trust

import (
	"context"
	"errors"
	"fmt"
	"time"
)

// ErrTrustBehind is retryable only within the caller's existing operation
// budget. It never authorizes using peer evidence as the local trust store.
var ErrTrustBehind = errors.New("peer evidence awaits current authority trust")

// CurrentForEvidence validates ahead-of-local evidence before using it as a
// refresh hint. Refresh must fetch only from locally pinned authority endpoints.
// The callback must verify the complete object, including its expected binding.
// Acceptance always uses independently read local trust, including revocations.
// This helper owns no retry loop and cannot extend the caller's deadline.
func CurrentForEvidence(ctx context.Context, anchor Anchor, snapshot SignedSnapshot, current func(context.Context) (VerifiedSnapshot, error), refresh func(context.Context) error, clock func() time.Time, verify func(VerifiedSnapshot) error) (VerifiedSnapshot, error) {
	var zero VerifiedSnapshot
	if err := ctx.Err(); err != nil {
		return zero, err
	}
	v, err := current(ctx)
	if err != nil {
		return zero, err
	}
	if snapshot.Payload.Generation > v.Generation() {
		evidence, err := VerifySnapshot(anchor, snapshot, clock())
		if err != nil {
			return zero, err
		}
		if err = verify(evidence); err != nil {
			return zero, err
		}
		var refreshErr error
		if refresh != nil {
			refreshErr = refresh(ctx)
		}
		// A concurrent maintenance refresh may have superseded our fetch. Read
		// local state even on fetch failure; never install the peer snapshot.
		v, err = current(ctx)
		if err != nil {
			return zero, err
		}
		if err = ctx.Err(); err != nil {
			return zero, err
		}
		if v.Generation() < evidence.Generation() {
			return zero, errors.Join(fmt.Errorf("%w (local %d, evidence %d)", ErrTrustBehind, v.Generation(), evidence.Generation()), refreshErr)
		}
	}
	if err = verify(v); err != nil {
		return zero, err
	}
	return v, ctx.Err()
}
