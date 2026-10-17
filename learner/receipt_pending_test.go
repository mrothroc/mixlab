package learner

import (
	"context"
	"errors"
	"fmt"
	"testing"

	private "github.com/mrothroc/mixlab/diloco/learner"
)

type pendingLookupPeer struct {
	*fakePeer
	err     error
	lookups int
}

func (p *pendingLookupPeer) Receipt(context.Context, private.Update) (private.Receipt, error) {
	p.lookups++
	return private.Receipt{}, p.err
}

func TestRuntimeReceiptLookupOnlyPendingPermitsSubmission(t *testing.T) {
	for _, tc := range []struct {
		name    string
		err     error
		pending bool
	}{
		{"pending", private.ErrReceiptPending, true},
		{"wrapped-pending", fmt.Errorf("lookup: %w", private.ErrReceiptPending), true},
		{"conflict", errors.New("HTTP 409"), false},
		{"authorization", errors.New("HTTP 403"), false},
		{"unavailable", private.ErrUnavailable, false},
		{"canceled", context.Canceled, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := setup(t)
			peer := &pendingLookupPeer{fakePeer: f.p, err: tc.err}
			f.r.Peer = peer
			_, err := f.r.Execute(context.Background(), f.a)
			if peer.lookups != 1 || f.calls != 1 {
				t.Fatalf("lookup replayed work: lookups=%d train=%d", peer.lookups, f.calls)
			}
			if tc.pending {
				if err != nil || len(f.p.deltas) != 1 || len(f.p.updates) != 1 {
					t.Fatalf("pending did not proceed to exactly one upload/submission: %v", err)
				}
			} else if !errors.Is(err, tc.err) || len(f.p.deltas) != 0 || len(f.p.updates) != 0 {
				t.Fatalf("lookup failure was swallowed or uploaded bytes: deltas=%d updates=%d err=%v", len(f.p.deltas), len(f.p.updates), err)
			}
		})
	}
}
