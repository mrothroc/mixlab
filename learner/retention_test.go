package learner

import (
	"context"
	"fmt"
	"os"
	"testing"

	private "github.com/mrothroc/mixlab/diloco/learner"
)

func TestRuntimeRetentionBoundedAcrossRounds(t *testing.T) {
	f := setup(t)
	for round := uint64(0); round < 20; round++ {
		f.a.Round, f.a.BaseRound = round, round
		f.a.ID = fmt.Sprintf("round%d", round)
		receipt := f.run(t)
		f.a.PreviousPrivateHash = receipt.PrivateHash
		entries, err := os.ReadDir(f.r.Store.path.Dir())
		if err != nil {
			t.Fatal(err)
		}
		// Two five-part snapshots, the pending delta, head and lock. Shared
		// immutable parts reduce this further; old rounds must not accumulate.
		if len(entries) > 13 {
			t.Fatalf("round %d retained %d files", round, len(entries))
		}
	}
}

type receiptOnlyPeer struct {
	*fakePeer
}

func (p receiptOnlyPeer) Receipt(context.Context, private.Update) (private.Receipt, error) {
	return p.receipt, nil
}

func TestRuntimeLostReceiptRecoveryAfterAssignmentDeadline(t *testing.T) {
	f := setup(t)
	f.p.failSubmit = true
	if _, err := f.r.Execute(context.Background(), f.a); err == nil {
		t.Fatal("expected lost receipt")
	}
	f.r.Peer = receiptOnlyPeer{f.p}
	f.r.Now = func() int64 { return f.a.ExpiresAt + 1 }
	before := len(f.p.deltas)
	r := f.run(t)
	if r != f.p.receipt || f.calls != 1 || len(f.p.updates) != 1 || len(f.p.deltas) != before {
		t.Fatal("authoritative receipt recovery reran work or uploaded again")
	}
}
