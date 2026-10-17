package clusterapp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/workerhost/contract"
	"github.com/mrothroc/mixlab/workerjob"
)

func TestLearnerExitDiagnostic(t *testing.T) {
	a := contract.Approved{Learner: &workerjob.LearnerAssignment{JobID: "job", AttemptID: "attempt"}}
	o := contract.Outcome{JobID: "job", AttemptID: "attempt", ManifestHash: strings.Repeat("a", 64), ApprovalHash: strings.Repeat("b", 64), Version: 2, Kind: contract.Failed, PID: 42, NoChild: true, Error: "EOF\nnot a new log record"}
	for _, tc := range []struct {
		name string
		out  contract.Outcome
		err  error
	}{
		{"terminal", o, nil},
		{"hosting_failure", contract.Outcome{}, errors.New("guardian reconcile failed\nsecond line")},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var events []string
			l := nodeExecutionLoop{ports: NodeExecutionPorts{Clock: func() time.Time { return time.Unix(123, 0) }, Event: func(e error) { events = append(events, e.Error()) }}}
			l.logLearnerExit(a, tc.out, tc.err)
			if len(events) != 1 || strings.Contains(events[0], "\n") {
				t.Fatalf("exit diagnostic missing or multiline: %q", events)
			}
			var record struct {
				Time    string            `json:"time"`
				Job     string            `json:"job"`
				Attempt string            `json:"attempt"`
				Outcome *contract.Outcome `json:"outcome"`
				Error   string            `json:"hosting_error"`
			}
			if err := json.Unmarshal([]byte(strings.TrimPrefix(events[0], "learner exit: ")), &record); err != nil {
				t.Fatal(err)
			}
			if record.Job != "job" || record.Attempt != "attempt" || record.Time != "1970-01-01T00:02:03Z" {
				t.Fatalf("lost diagnostic identity: %+v", record)
			}
			if tc.err == nil {
				if record.Outcome == nil || *record.Outcome != o || record.Error != "" {
					t.Fatal("lost terminal outcome")
				}
			} else if record.Error != tc.err.Error() || record.Outcome != nil {
				t.Fatal("invented outcome or lost hosting error")
			}
		})
	}
}

func TestNodeResponseRequiresActualRequestExpiry(t *testing.T) {
	for _, expired := range []bool{false, true} {
		ctx := context.Background()
		if expired {
			var cancel context.CancelFunc
			ctx, cancel = context.WithDeadline(ctx, time.Now().Add(-time.Second))
			defer cancel()
		}
		for _, timeout := range []bool{false, true} {
			err := errors.New("invalid evidence")
			if timeout {
				err = fmt.Errorf("private fetch: %w", context.DeadlineExceeded)
			}
			w := httptest.NewRecorder()
			nodeResponse(&nodeResponseContext{ResponseWriter: w, ctx: ctx}, nil, err)
			want := http.StatusConflict
			if expired && timeout {
				want = http.StatusServiceUnavailable
			}
			if w.Code != want || strings.Contains(w.Body.String(), "private") {
				t.Fatalf("expired=%v timeout=%v status=%d", expired, timeout, w.Code)
			}
		}
	}
}

func TestNodeTrustBehindHTTPStatus(t *testing.T) {
	for _, tc := range []struct {
		name   string
		err    error
		status int
	}{
		{"behind", trust.ErrTrustBehind, http.StatusServiceUnavailable},
		{"wrapped", fmt.Errorf("private refresh endpoint: %w", trust.ErrTrustBehind), http.StatusServiceUnavailable},
		{"invalid", errors.New("private invalid proof"), http.StatusConflict},
		{"unclassified", errors.New(trust.ErrTrustBehind.Error()), http.StatusConflict},
		{"success", nil, http.StatusOK},
	} {
		t.Run(tc.name, func(t *testing.T) {
			w := httptest.NewRecorder()
			nodeResponse(w, NodeLearnerAck{"job"}, tc.err)
			if w.Code != tc.status || strings.Contains(w.Body.String(), "private") {
				t.Fatalf("status=%d body=%s", w.Code, w.Body.String())
			}
		})
	}
}

func TestLearnerRecoveryDiagnosticStaysLocal(t *testing.T) {
	var events []string
	h := nodeHTTP{event: func(err error) { events = append(events, err.Error()) }}
	w := httptest.NewRecorder()
	h.recoveryResponse(w, "job", "retry-start", nil, errors.New("private recovery fence mismatch\nextra line"))
	if w.Code != http.StatusConflict || strings.Contains(w.Body.String(), "private recovery") {
		t.Fatalf("private diagnostic escaped to HTTP: %d %s", w.Code, w.Body.String())
	}
	if len(events) != 1 || !strings.Contains(events[0], "operation=retry-start") || !strings.Contains(events[0], "private recovery fence mismatch") || strings.Contains(events[0], "\n") {
		t.Fatalf("missing or unsafe local diagnostic: %q", events)
	}
	w = httptest.NewRecorder()
	h.recoveryResponse(w, "job", "retry-start", NodeLearnerAck{"job"}, nil)
	if len(events) != 1 || w.Code != http.StatusOK {
		t.Fatal("successful recovery emitted an error")
	}
}
