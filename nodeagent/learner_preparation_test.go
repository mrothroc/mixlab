package nodeagent

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/workerprobe"
)

func (f *learnerPreparationFixture) cache(t *testing.T) learnerPreparation {
	t.Helper()
	name, err := learnerPreparationFile(f.manifest.Slot.Allocation.JobID)
	preparationCheck(t, err)
	b, err := f.s.path.ReadFileLimit(name, 64<<10)
	preparationCheck(t, err)
	var c learnerPreparation
	preparationCheck(t, json.Unmarshal(b, &c))
	return c
}

func TestLearnerPreparationDurableStagesAndPreparedReplay(t *testing.T) {
	f := newLearnerPreparationFixture(t)
	f.manifest.Validation = &run.ValidationDataset{Hash: f.profile.Datasets[1].ID, Split: f.profile.Datasets[1].Selector, BatchTokens: 16, SequenceLength: 4, Batches: 1}
	f.accepted = f.accept(t, f.manifest)
	p := f.ports()
	probeCount, trainCount, validationCount := 0, 0, 0
	p.Probe = func(context.Context) (workerprobe.Report, error) {
		probeCount++
		return f.profile.Probe, nil
	}
	wantErr := errors.New("interrupted validation hash")
	p.DatasetID = func(_ context.Context, path string) (string, error) {
		if path == f.profile.Datasets[0].TrainPattern {
			trainCount++
			return f.profile.Datasets[0].ID, nil
		}
		validationCount++
		if validationCount == 1 {
			return "", wantErr
		}
		return f.profile.Datasets[1].ID, nil
	}
	_, err := f.s.PrepareLearnerChecked(context.Background(), f.actor, f.accepted, f.lease.Version, p)
	if !errors.Is(err, wantErr) {
		t.Fatalf("first attempt: %v", err)
	}
	c := f.cache(t)
	if c.Probe == nil || c.DatasetHash == "" || c.ValidationHash != "" {
		t.Fatalf("unexpected completed stages: %+v", c)
	}
	// Reopening the store represents owner recreation, not service startup:
	// the service's InterruptActive policy still fences real restarts.
	f.s, err = Open(f.s.path, f.s.cluster, f.s.node)
	preparationCheck(t, err)
	j, err := f.s.PrepareLearnerChecked(context.Background(), f.actor, f.accepted, f.lease.Version, p)
	preparationCheck(t, err)
	if probeCount != 1 || trainCount != 1 || validationCount != 2 {
		t.Fatalf("repeated completed work: %d/%d/%d", probeCount, trainCount, validationCount)
	}
	old, j, err := f.s.loadLearner(j.Signed.Manifest.Slot.Allocation.JobID)
	preparationCheck(t, err)
	j.Version = 3
	j.Workload = &workloadIntent{Principal: strings.Repeat("5", 32), PreparedVersion: 1}
	preparationCheck(t, f.s.saveLearner(old, j))
	p.Probe = func(context.Context) (workerprobe.Report, error) {
		t.Fatal("replayed probe")
		return workerprobe.Report{}, nil
	}
	p.DatasetID = func(context.Context, string) (string, error) { t.Fatal("replayed hash"); return "", nil }
	replay, err := f.s.PrepareLearnerChecked(context.Background(), f.actor, f.accepted, 999, p)
	preparationCheck(t, err)
	if digest(replay) != digest(j) {
		t.Fatal("did not return the existing partially prepared workload journal")
	}
}

func TestLearnerPreparationRenewalsAndAdmissionDeadline(t *testing.T) {
	f := newLearnerPreparationFixture(t)
	p := f.ports()
	renew := func(key string) {
		f.now = f.now.Add(time.Second)
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		defer cancel()
		var err error
		f.lease, err = f.s.RenewLease(ctx, f.actor, LeaseCommand{IdempotencyKey: strings.Repeat(key, 32), Lease: f.lease.ID, ExpectedVersion: f.lease.Version, TTLSeconds: 600}, f.now)
		preparationCheck(t, err)
	}
	p.Probe = func(ctx context.Context) (workerprobe.Report, error) {
		deadline, ok := ctx.Deadline()
		if !ok || time.Until(deadline) < 9*time.Minute || time.Until(deadline) > 10*time.Minute {
			t.Fatalf("probe has hidden cap or unbounded deadline: %v", deadline)
		}
		f.now = f.now.Add(40 * time.Second)
		renew("6")
		return f.profile.Probe, nil
	}
	p.DatasetID = func(context.Context, string) (string, error) {
		renew("7")
		return f.profile.Datasets[0].ID, nil
	}
	initialVersion := f.lease.Version
	_, err := f.s.PrepareLearnerChecked(context.Background(), f.actor, f.accepted, initialVersion, p)
	preparationCheck(t, err)
	l, err := f.s.LeaseStatus(context.Background(), f.actor, f.lease.ID, f.now)
	preparationCheck(t, err)
	if l.State != Prepared || l.Version != initialVersion+3 || f.cache(t).ExpectedVersion != initialVersion {
		t.Fatalf("renewal-safe commit failed: %+v", l)
	}
}

func TestLearnerPreparationExpectedVersionOnlyAtIntentCreation(t *testing.T) {
	f := newLearnerPreparationFixture(t)
	p := f.ports()
	p.Probe = func(context.Context) (workerprobe.Report, error) {
		t.Fatal("probe before version check")
		return workerprobe.Report{}, nil
	}
	_, err := f.s.PrepareLearnerChecked(context.Background(), f.actor, f.accepted, f.lease.Version+1, p)
	if err == nil {
		t.Fatal("accepted wrong initial version")
	}
	name, err := learnerPreparationFile(f.manifest.Slot.Allocation.JobID)
	preparationCheck(t, err)
	_, err = f.s.path.ReadFileLimit(name, 64<<10)
	if !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("unexpected intent after rejected version: %v", err)
	}
}

func TestLearnerPreparationLiveGuardsAfterCallbacks(t *testing.T) {
	for _, kind := range []string{"abort", "profile", "lease_expiry", "admission_expiry", "actor_expiry", "cancel"} {
		t.Run(kind, func(t *testing.T) {
			f := newLearnerPreparationFixture(t)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			if kind == "actor_expiry" {
				f.actor.ExpiresAt = f.now.Add(time.Minute)
			}
			p := f.ports()
			p.Probe = func(context.Context) (workerprobe.Report, error) {
				switch kind {
				case "abort":
					ctx, done := context.WithTimeout(context.Background(), time.Second)
					defer done()
					_, err := f.s.AbortReservation(ctx, f.actor, f.reserve, f.now)
					preparationCheck(t, err)
				case "profile":
					changed := f.profile
					changed.DisplayName += " changed"
					b, err := json.Marshal(changed)
					preparationCheck(t, err)
					preparationCheck(t, f.s.path.WriteFile(profileFile, b))
				case "lease_expiry":
					f.now = time.Unix(f.lease.Expires, 0)
				case "admission_expiry":
					f.now = time.Unix(f.manifest.Plan.ExpiresAt, 0)
				case "actor_expiry":
					f.now = f.actor.ExpiresAt
				case "cancel":
					cancel()
				}
				return f.profile.Probe, nil
			}
			p.DatasetID = func(context.Context, string) (string, error) { t.Fatal("hash after fence"); return "", nil }
			_, err := f.s.PrepareLearnerChecked(ctx, f.actor, f.accepted, f.lease.Version, p)
			if err == nil || f.cache(t).Probe != nil {
				t.Fatalf("published probe after %s: %v", kind, err)
			}
		})
	}
}

func TestLearnerPreparationRejectsManifestSubstitution(t *testing.T) {
	f := newLearnerPreparationFixture(t)
	p := f.ports()
	p.DatasetID = func(context.Context, string) (string, error) { return "", errors.New("interrupted") }
	_, err := f.s.PrepareLearnerChecked(context.Background(), f.actor, f.accepted, f.lease.Version, p)
	if err == nil {
		t.Fatal("expected interrupted dataset stage")
	}
	changed := f.manifest
	changed.Attempt = strings.Repeat("9", 32)
	accepted := f.accept(t, changed)
	p.Probe = func(context.Context) (workerprobe.Report, error) {
		t.Fatal("probe for substituted manifest")
		return workerprobe.Report{}, nil
	}
	p.DatasetID = func(context.Context, string) (string, error) {
		t.Fatal("hash for substituted manifest")
		return "", nil
	}
	_, err = f.s.PrepareLearnerChecked(context.Background(), f.actor, accepted, f.lease.Version, p)
	if err == nil {
		t.Fatal("accepted substituted signed manifest")
	}
}
