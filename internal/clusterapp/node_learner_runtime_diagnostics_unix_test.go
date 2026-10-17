//go:build darwin || linux

package clusterapp

import (
	"context"
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/learneripc"
	wc "github.com/mrothroc/mixlab/workercontrol"
)

func TestNodeLearnerRuntimeLogsLocalRelayFailure(t *testing.T) {
	r := learnerRuntimeRemote(t)
	var events []string
	r.runtime.Event = func(err error) { events = append(events, err.Error()) }
	s := r.runtime.relayServer(r.binding, 1024)
	defer s.Close()
	if s.OnError == nil {
		t.Fatal("production relay omitted local diagnostic hook")
	}
	ctx := context.Background()
	q := wc.Envelope{Kind: wc.KindRoundRequest, PayloadKind: learneripc.PayloadKind, PayloadVersion: 1, JobID: r.binding.JobID, AttemptID: r.binding.AttemptID, Payload: []byte(`{"operation":"poll"}`)}
	_, err := s.Handle(ctx, q)
	check(t, err)
	if len(events) != 0 {
		t.Fatal("successful precommit poll generated an error event", events)
	}
	check(t, r.runtime.Store.ConfirmLearnerStopped(ctx, r.binding))
	before, err := r.runtime.Store.ActiveExecution(ctx)
	check(t, err)
	reply, err := s.Handle(ctx, q)
	check(t, err)
	want := fmt.Sprintf("learner relay error: time=%s operation=%q job=%q attempt=%q error=%q", testNow.UTC().Format(time.RFC3339Nano), "poll", r.binding.JobID, r.binding.AttemptID, "learner attempt no longer live")
	if len(events) != 1 || events[0] != want {
		t.Fatalf("missing trusted attempt context or original local failure: %v", events)
	}
	if !strings.Contains(string(reply.Payload), "operation rejected") {
		t.Fatal("attempt fence did not remain a sanitized rejection", string(reply.Payload))
	}
	for _, forbidden := range []string{"no longer live", r.binding.JobID, r.binding.AttemptID} {
		if strings.Contains(string(reply.Payload), forbidden) {
			t.Fatal("agent diagnostic crossed IPC", string(reply.Payload))
		}
	}
	after, err := r.runtime.Store.ActiveExecution(ctx)
	check(t, err)
	if !reflect.DeepEqual(before, after) {
		t.Fatal("logging changed stopped-attempt state")
	}
}

func TestNodeLearnerRuntimeDiagnosticContextIsBoundAndQuoted(t *testing.T) {
	binding := wc.Binding{JobID: "admitted-job", AttemptID: "admitted-attempt"}
	var events []string
	now := time.Date(2026, 10, 17, 8, 36, 47, 123456789, time.FixedZone("test", -7*60*60))
	r := NodeLearnerRuntime{Clock: func() time.Time { return now }, Event: func(err error) { events = append(events, err.Error()) }}
	s := r.relayServer(binding, 1024)
	defer s.Close()
	binding.JobID, binding.AttemptID = "changed-job", "changed-attempt"
	s.OnError("write_end", fmt.Errorf("artifact transfer failed: %s", "unexpected EOF\nsecond diagnostic"))
	if len(events) != 1 || strings.Contains(events[0], "\n") || !strings.Contains(events[0], `job="admitted-job" attempt="admitted-attempt"`) || !strings.Contains(events[0], `unexpected EOF\nsecond diagnostic`) {
		t.Fatal("diagnostic lost admitted context or injected a log line", events)
	}
	if !strings.Contains(events[0], "time=2026-10-17T15:36:47.123456789Z") {
		t.Fatal("diagnostic did not use injected clock in UTC with nanoseconds", events)
	}
	quiet := (NodeLearnerRuntime{}).relayServer(binding, 1024)
	defer quiet.Close()
	if quiet.OnError != nil {
		t.Fatal("optional event sink installed an unexpected logger")
	}
}

func TestNodeLearnerRuntimeDiagnosticBoundsError(t *testing.T) {
	for _, size := range []int{2048, 2049, 1 << 20} {
		t.Run(fmt.Sprint(size), func(t *testing.T) {
			var events []string
			r := NodeLearnerRuntime{Clock: func() time.Time { return testNow }, Event: func(err error) { events = append(events, err.Error()) }}
			s := r.relayServer(wc.Binding{JobID: "job", AttemptID: "attempt"}, 1024)
			defer s.Close()
			s.OnError("read_begin", fmt.Errorf("%s", strings.Repeat("x", size)))
			want := strings.Repeat("x", min(size, 2048))
			if size > 2048 {
				want = strings.Repeat("x", 2048-len(" [truncated]")) + " [truncated]"
			}
			if len(events) != 1 || !strings.HasSuffix(events[0], fmt.Sprintf("error=%q", want)) || len(events[0]) > 2300 {
				t.Fatal("local error text was not bounded or marked truncated")
			}
		})
	}
}
