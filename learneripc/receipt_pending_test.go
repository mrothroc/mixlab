//go:build darwin || linux

package learneripc

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"testing"
	"time"

	private "github.com/mrothroc/mixlab/diloco/learner"
	wc "github.com/mrothroc/mixlab/workercontrol"
)

func pendingReceiptRequest(t *testing.T, operation string) request {
	t.Helper()
	var update private.Update
	b, err := os.ReadFile("../diloco/testdata/update.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, &update); err != nil {
		t.Fatal(err)
	}
	q := request{Operation: "call", Call: &private.RelayRequest{Format: "mixlab_learner_relay_v1", Binding: update.Binding, WorkerID: update.WorkerID, Operation: operation, Update: &update}}
	if err := q.Validate(); err != nil {
		t.Fatal(err)
	}
	return q
}

func TestLearnerIPCReceiptPendingIsQuietAndScoped(t *testing.T) {
	for _, tc := range []struct {
		name, operation, code string
		err                   error
		logs                  int
	}{
		{"pending", "receipt", receiptPendingCode, private.ErrReceiptPending, 0},
		{"wrapped-pending", "receipt", receiptPendingCode, fmt.Errorf("local detail: %w", private.ErrReceiptPending), 0},
		{"submit-not-pending", "submit", "rejected", private.ErrReceiptPending, 1},
		{"conflict", "receipt", "rejected", errors.New("learner operation rejected (HTTP 409)"), 1},
		{"authorization", "receipt", "rejected", errors.New("learner access rejected (HTTP 403)"), 1},
		{"text-is-not-a-code", "receipt", "rejected", errors.New("receipt_pending"), 1},
		{"unavailable", "receipt", "unavailable", private.ErrUnavailable, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			remote := &diagnosticRemote{err: tc.err}
			logs := 0
			s := &Server{Remote: remote, MaxBytes: 1024, OnError: func(operation string, err error) {
				logs++
				if operation != "call/"+tc.operation || err != tc.err {
					t.Fatal("diagnostic changed error or operation", operation, err)
				}
			}}
			defer s.Close()
			e, err := envelope(wc.KindRoundRequest, pendingReceiptRequest(t, tc.operation))
			if err != nil {
				t.Fatal(err)
			}
			reply, err := s.Handle(context.Background(), e)
			if err != nil {
				t.Fatal(err)
			}
			var out response
			if err := private.Decode(reply.Payload, &out); err != nil {
				t.Fatal(err)
			}
			if out.ErrorCode != tc.code || logs != tc.logs || remote.calls != 1 {
				t.Fatalf("receipt result changed logging or replay: code=%q logs=%d calls=%d", out.ErrorCode, logs, remote.calls)
			}
			if tc.code == receiptPendingCode && out.Error != receiptPendingMessage {
				t.Fatal("pending result leaked diagnostic text", out.Error)
			}
		})
	}
}

func TestLearnerIPCReceiptPendingClientMapping(t *testing.T) {
	for _, tc := range []struct {
		name, operation string
		out             response
		pending         bool
	}{
		{"pending", "receipt", response{Error: receiptPendingMessage, ErrorCode: receiptPendingCode}, true},
		{"wrong-operation", "submit", response{Error: receiptPendingMessage, ErrorCode: receiptPendingCode}, false},
		{"generic-rejection", "receipt", response{Error: receiptPendingMessage, ErrorCode: "rejected"}, false},
		{"malformed-pending", "receipt", response{Error: "private detail", ErrorCode: receiptPendingCode}, false},
		{"pending-with-data", "receipt", response{Error: receiptPendingMessage, ErrorCode: receiptPendingCode, Data: []byte("secret")}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			agent, worker := ipcPair(t)
			ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
			defer cancel()
			done := make(chan error, 1)
			go func() {
				select {
				case e := <-agent.Events():
					reply, err := envelope(wc.KindRoundResponse, tc.out)
					if err == nil {
						err = agent.Reply(ctx, e, reply)
					}
					done <- err
				case <-ctx.Done():
					done <- ctx.Err()
				}
			}()
			c := &Client{Session: worker}
			_, err := c.call(ctx, pendingReceiptRequest(t, tc.operation))
			if err == nil || errors.Is(err, private.ErrReceiptPending) != tc.pending || errors.Is(err, private.ErrUnavailable) || ctx.Err() != nil {
				t.Fatalf("pending mapping retried or hid rejection: %v", err)
			}
			if err := <-done; err != nil {
				t.Fatal(err)
			}
			select {
			case <-agent.Events():
				t.Fatal("receipt pending replayed the semantic request")
			default:
			}
		})
	}
}
