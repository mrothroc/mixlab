package contract

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestProcessExitValidation(t *testing.T) {
	for _, e := range []ProcessExit{{}, {Kind: ProcessNotStarted}, {Kind: ProcessExited}, {Kind: ProcessExited, Code: 17}, {Kind: ProcessSignaled, Signal: 9}} {
		if err := e.Validate(); err != nil {
			t.Fatalf("valid exit %+v: %v", e, err)
		}
	}
	for _, e := range []ProcessExit{{Code: 1}, {Signal: 9}, {Kind: "unknown"}, {Kind: ProcessNotStarted, Code: 1}, {Kind: ProcessExited, Code: -1}, {Kind: ProcessExited, Code: 256}, {Kind: ProcessExited, Signal: 9}, {Kind: ProcessSignaled}, {Kind: ProcessSignaled, Signal: -1}, {Kind: ProcessSignaled, Signal: 256}, {Kind: ProcessSignaled, Signal: 9, Code: 1}} {
		if e.Validate() == nil {
			t.Fatalf("accepted invalid exit %+v", e)
		}
	}
}

func TestOutcomeExitCompatibilityAndEquality(t *testing.T) {
	o := Outcome{JobID: "job", AttemptID: "attempt", ManifestHash: strings.Repeat("a", 64), ApprovalHash: strings.Repeat("b", 64), Version: 2, Kind: Failed, PID: 123, NoChild: true, Error: "protocol failed"}
	legacy, err := json.Marshal(o)
	if err != nil || bytes.Contains(legacy, []byte(`"exit"`)) {
		t.Fatalf("zero exit changed legacy JSON: %s %v", legacy, err)
	}
	var decoded Outcome
	if err := json.Unmarshal(legacy, &decoded); err != nil || decoded != o || decoded.Validate() != nil {
		t.Fatalf("legacy outcome round trip: %+v %v", decoded, err)
	}
	o.Exit = ProcessExit{Kind: ProcessExited}
	if err := o.Validate(); err != nil {
		t.Fatal("protocol failure may accompany OS exit zero", err)
	}
	b, err := json.Marshal(o)
	if err != nil || !bytes.Contains(b, []byte(`"exit":{"kind":"exited"}`)) {
		t.Fatalf("known exit zero lost: %s %v", b, err)
	}
	if err := json.Unmarshal(b, &decoded); err != nil || decoded != o {
		t.Fatalf("exit changed comparable outcome: %+v %v", decoded, err)
	}
	o.Exit = ProcessExit{Kind: ProcessNotStarted}
	if o.Validate() == nil {
		t.Fatal("not-started exit with PID accepted")
	}
	o.Kind, o.Version, o.NoChild, o.Error = Started, 1, false, ""
	o.Exit = ProcessExit{Kind: ProcessExited}
	if o.Validate() == nil {
		t.Fatal("started outcome carries OS exit evidence")
	}
}
