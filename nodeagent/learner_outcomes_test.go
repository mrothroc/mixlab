package nodeagent

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/workerhost/contract"
)

func TestRetainLearnerOutcomeReceipts(t *testing.T) {
	s, _ := leaseFixture(t)
	o := contract.Outcome{JobID: strings.Repeat("c", 32), AttemptID: strings.Repeat("d", 32), ManifestHash: strings.Repeat("a", 64), ApprovalHash: strings.Repeat("b", 64), Version: 2, Kind: contract.Failed, PID: 42, NoChild: true, Error: "failure", Exit: contract.ProcessExit{Kind: contract.ProcessSignaled, Signal: 9}}
	for _, receipt := range []*contract.Outcome{nil, &o, &o} {
		if err := s.retainLearnerOutcome(receipt); err != nil {
			t.Fatal(err)
		}
	}
	name, err := learnerOutcomeFile(o.JobID, o.AttemptID, o.Version)
	if err != nil {
		t.Fatal(err)
	}
	b, err := s.path.ReadFileLimit(name, learnerOutcomeBytes)
	if err != nil {
		t.Fatal(err)
	}
	want, _ := json.Marshal(o)
	if !bytes.Equal(b, want) {
		t.Fatal("receipt lost exit evidence")
	}
	changed := o
	changed.Exit.Signal = 15
	if s.retainLearnerOutcome(&changed) == nil {
		t.Fatal("conflicting immutable receipt accepted")
	}
	if _, err := learnerOutcomeFile("../escape", o.AttemptID, 1); err == nil {
		t.Fatal("unsafe diagnostic path accepted")
	}
}

func TestLearnerOutcomeHistoryDoesNotGrowJournal(t *testing.T) {
	s, _ := leaseFixture(t)
	job := strings.Repeat("c", 32)
	name, err := learnerFile(job)
	if err != nil {
		t.Fatal(err)
	}
	// Retention must not read or grow the lifecycle journal, even at its limit.
	original := bytes.Repeat([]byte("x"), (2<<20)-1)
	if err := s.path.CompareAndSwap(name, nil, original); err != nil {
		t.Fatal(err)
	}
	total := 0
	for i := range 100 {
		o := contract.Outcome{JobID: job, AttemptID: fmt.Sprintf("%032x", i+1), ManifestHash: strings.Repeat("a", 64), ApprovalHash: strings.Repeat("b", 64), Version: 1, Kind: contract.Failed, NoChild: true, Error: strings.Repeat("\x00", 4096)}
		if err := s.retainLearnerOutcome(&o); err != nil {
			t.Fatal(err)
		}
		b, _ := json.Marshal(o)
		total += len(b)
	}
	if total <= 2<<20 {
		t.Fatal("fixture did not cross the old aggregate journal limit")
	}
	after, err := s.path.ReadFileLimit(name, 2<<20)
	if err != nil || !bytes.Equal(after, original) {
		t.Fatal("diagnostic history changed the lifecycle journal", err)
	}
}
