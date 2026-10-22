package admission

import (
	"context"
	"errors"
	"testing"
)

func TestServiceObservationRetryDurabilityAndLegacyRead(t *testing.T) {
	i, q := inboxFixture(t)
	ctx := context.Background()
	if err := i.Accept(ctx, q); err != nil {
		t.Fatal(err)
	}
	if err := i.Observe(ctx, q.Launch.Run, 100, errors.New("legacy failure")); err != nil {
		t.Fatal(err)
	}
	old, err := i.Observation(ctx, q.Launch.Run)
	if err != nil || old.RetryAt != 0 || old.Error != "legacy failure" {
		t.Fatal(old, err)
	}
	want := Observation{At: 101, Error: "offline", RetryAt: 150, RetryFailures: 8, RetryPhase: "aborting", EventAt: 100, FailureClass: "timeout"}
	if err := i.ObserveState(ctx, q.Launch.Run, want); err != nil {
		t.Fatal(err)
	}
	reopened, err := OpenInbox(i.path)
	if err != nil {
		t.Fatal(err)
	}
	got, err := reopened.Observation(ctx, q.Launch.Run)
	if err != nil || got != want {
		t.Fatal(got, err)
	}
	// Even within one minute, changed retry scheduling must be persisted.
	want.RetryAt++
	if err := reopened.ObserveState(ctx, q.Launch.Run, want); err != nil {
		t.Fatal(err)
	}
	got, err = i.Observation(ctx, q.Launch.Run)
	if err != nil || got != want {
		t.Fatal(got, err)
	}
	for _, mutate := range []func(*Observation){
		func(v *Observation) { v.RetryAt = v.At + 61 },
		func(v *Observation) { v.RetryAt = v.At },
		func(v *Observation) { v.RetryPhase = "unknown" },
		func(v *Observation) { v.RetryFailures = 17 },
		func(v *Observation) { v.Error = "" },
	} {
		bad := want
		mutate(&bad)
		if err := i.ObserveState(ctx, q.Launch.Run, bad); err == nil {
			t.Fatal("accepted invalid retry record", bad)
		}
	}
}
