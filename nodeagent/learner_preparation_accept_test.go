package nodeagent

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/workerprobe"
)

func TestLearnerPreparationHistoricalAcceptance(t *testing.T) {
	for _, stage := range []string{"unaccepted", "intent", "prepared_journal"} {
		t.Run(stage, func(t *testing.T) {
			f := newLearnerPreparationFixture(t)
			f.manifest.Plan.ExpiresAt = f.now.Add(30 * time.Minute).Unix()
			var err error
			f.manifest.Plan.SHA256, err = f.manifest.Plan.Digest()
			preparationCheck(t, err)
			f.accepted = f.accept(t, f.manifest)
			f.lease, err = f.s.RenewLease(context.Background(), f.actor, LeaseCommand{Lease: f.lease.ID, ExpectedVersion: f.lease.Version, IdempotencyKey: strings.Repeat("a", 32), TTLSeconds: 1800}, f.now)
			preparationCheck(t, err)
			signed, original, err := f.accepted.Value()
			preparationCheck(t, err)
			// Initial acceptance alone is not durable. Only Checked's intent or
			// the older prepared journal may authorize historical continuation.
			_, err = f.s.AcceptLearnerPreparation(context.Background(), f.anchor, f.view, f.actor, signed, f.now)
			preparationCheck(t, err)
			switch stage {
			case "intent":
				p := f.ports()
				interrupted := errors.New("interrupted before first successful stage")
				p.Probe = func(context.Context) (workerprobe.Report, error) {
					if f.cache(t).ProofDigest != original.Digest {
						t.Fatal("probe preceded durable acceptance")
					}
					return workerprobe.Report{}, interrupted
				}
				_, err = f.s.PrepareLearnerChecked(context.Background(), f.actor, f.accepted, f.lease.Version, p)
				if !errors.Is(err, interrupted) {
					t.Fatal(err)
				}
			case "prepared_journal":
				_, err = f.s.PrepareLearner(context.Background(), f.actor, f.accepted, f.lease.Version, f.now)
				preparationCheck(t, err)
			}
			changed := f.manifest
			changed.Attempt = strings.Repeat("9", 32)
			changedAccepted := f.accept(t, changed)
			changedSigned, _, err := changedAccepted.Value()
			preparationCheck(t, err)
			f.now = f.now.Add(trust.SnapshotLifetime + time.Second)
			current := f.freshView(t, nil)
			f.actor, err = trust.AuthenticatePrincipal(f.anchor, current, f.chain, f.now)
			preparationCheck(t, err)
			if _, err := nodejob.AcceptLearner(f.anchor, current, f.actor, f.s.node, signed, f.now); err == nil {
				t.Fatal("fixture did not expire embedded snapshot")
			}
			continued, err := f.s.AcceptLearnerPreparation(context.Background(), f.anchor, current, f.actor, signed, f.now)
			if stage == "unaccepted" {
				if err == nil {
					t.Fatal("unaccepted old evidence created historical admission")
				}
				return
			}
			preparationCheck(t, err)
			_, receipt, err := continued.Value()
			preparationCheck(t, err)
			if receipt != original {
				t.Fatalf("historical receipt changed: %+v != %+v", receipt, original)
			}
			_, err = f.s.PrepareLearnerChecked(context.Background(), f.actor, continued, f.lease.Version, f.ports())
			preparationCheck(t, err)
			if _, err := f.s.AcceptLearnerPreparation(context.Background(), f.anchor, current, f.actor, changedSigned, f.now); err == nil {
				t.Fatal("accepted replacement manifest under historical receipt")
			}
			if _, err := f.s.AcceptLearnerPreparation(context.Background(), f.anchor, f.view, f.actor, signed, f.now); err == nil {
				t.Fatal("historical acceptance used stale current trust")
			}
			expiredActor := f.actor
			expiredActor.ExpiresAt = f.now
			if _, err := f.s.AcceptLearnerPreparation(context.Background(), f.anchor, current, expiredActor, signed, f.now); err == nil {
				t.Fatal("historical receipt bypassed current actor expiry")
			}
			wrongActor := f.actor
			wrongActor.Principal = strings.Repeat("d", 32)
			if _, err := f.s.AcceptLearnerPreparation(context.Background(), f.anchor, current, wrongActor, signed, f.now); err == nil {
				t.Fatal("historical receipt bypassed current caller identity")
			}
			compromised := f.freshView(t, []trust.Revocation{{Kind: "principal", ID: f.actor.Principal, Mode: "compromise", Reason: "test", FirstGeneration: current.Generation()}})
			if _, err := f.s.AcceptLearnerPreparation(context.Background(), f.anchor, compromised, f.actor, signed, f.now); err == nil {
				t.Fatal("compromise revocation preserved historical acceptance")
			}
			_, err = f.s.AbortReservation(context.Background(), f.actor, f.reserve, f.now)
			preparationCheck(t, err)
			if _, err := f.s.AcceptLearnerPreparation(context.Background(), f.anchor, current, f.actor, signed, f.now); err == nil {
				t.Fatal("historical receipt bypassed reservation abort")
			}
		})
	}
}

func TestLearnerPreparationHistoricalRejectsCorruptCache(t *testing.T) {
	f := newLearnerPreparationFixture(t)
	_, err := f.s.PrepareLearnerChecked(context.Background(), f.actor, f.accepted, f.lease.Version, f.ports())
	preparationCheck(t, err)
	signed, _, err := f.accepted.Value()
	preparationCheck(t, err)
	name, err := learnerPreparationFile(f.manifest.Slot.Allocation.JobID)
	preparationCheck(t, err)
	b, err := f.s.path.ReadFileLimit(name, 64<<10)
	preparationCheck(t, err)
	preparationCheck(t, f.s.path.WriteFile(name, append(b, '\n')))
	if _, err := f.s.AcceptLearnerPreparation(context.Background(), f.anchor, f.view, f.actor, signed, f.now); err == nil {
		t.Fatal("corrupt sidecar fell back to learner journal or fresh acceptance")
	}
}
