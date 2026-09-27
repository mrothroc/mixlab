package workerjob

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/artifact"
)

func TestAssignmentCheckpointBinding(t *testing.T) {
	a := fixture(t)
	a.CheckpointAt, a.OutputMaxBytes = 2, 8<<20
	a.Resume = &artifact.Ref{SHA256: strings.Repeat("f", 64), Bytes: 1024}
	a.ResumePath = "/private/input/checkpoint"
	binding, err := a.Binding()
	if err != nil {
		t.Fatal(err)
	}
	raw, err := json.Marshal(a)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Decode(raw, binding); err != nil {
		t.Fatal(err)
	}
	for _, mutate := range []func(*Assignment){
		func(a *Assignment) { a.CheckpointAt++ },
		func(a *Assignment) { a.ResumePath += "other" },
		func(a *Assignment) { a.Resume = &artifact.Ref{SHA256: strings.Repeat("e", 64), Bytes: 1024} },
	} {
		b := a
		mutate(&b)
		raw, _ := json.Marshal(b)
		if _, err := Decode(raw, binding); err == nil {
			t.Fatal("checkpoint fields not bound to authenticated assignment")
		}
	}
	for name, mutate := range map[string]func(*Assignment){
		"missing output":   func(a *Assignment) { a.OutputMaxBytes = 0 },
		"attempt overflow": func(a *Assignment) { a.CheckpointAt = 1 << 31 },
		"relative input":   func(a *Assignment) { a.ResumePath = "checkpoint" },
		"missing input":    func(a *Assignment) { a.ResumePath = "" },
		"unapproved input": func(a *Assignment) { a.Resume = nil },
		"bad digest":       func(a *Assignment) { a.Resume = &artifact.Ref{SHA256: "bad", Bytes: 1024} },
	} {
		t.Run(name, func(t *testing.T) {
			b := a
			mutate(&b)
			if b.Validate() == nil {
				t.Fatal("invalid checkpoint assignment accepted")
			}
		})
	}
}

func TestManagedOutputBudget(t *testing.T) {
	for _, disk := range []uint64{0, 1 << 20, 16 << 20, 4 << 30} {
		if got := ManagedOutputBudget(disk, 1<<20, 0, false); got != OutputBudget(disk, 1<<20) {
			t.Fatalf("default budget changed: %d", disk)
		}
	}
	const input, logs, control = 4 << 20, 1 << 20, 1 << 20
	const disk = 32 << 20
	for _, checkpoint := range []bool{false, true} {
		copies := uint64(2)
		if checkpoint {
			copies = 3
		}
		if got := ManagedOutputBudget(disk, logs, input, checkpoint); got != (disk-logs-control-3*input)/copies {
			t.Fatalf("unaccounted checkpoint copies: %d", got)
		}
	}
	if ManagedOutputBudget(3*input+logs+control, logs, input, true) != 0 || ManagedOutputBudget(disk, logs, artifact.MaxBytes+1, true) != 0 {
		t.Fatal("accepted insufficient or oversized input budget")
	}
}
