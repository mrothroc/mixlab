//go:build darwin || linux

package learneripc

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	private "github.com/mrothroc/mixlab/diloco/learner"
	wc "github.com/mrothroc/mixlab/workercontrol"
)

type failingRemote struct {
	fakeRemote
	err error
}

func (f failingRemote) Poll(context.Context) (private.RelayResponse, error) {
	return private.RelayResponse{}, f.err
}

func TestLearnerIPCErrorClassificationAndSanitization(t *testing.T) {
	for _, unavailable := range []bool{false, true} {
		t.Run(fmt.Sprint(unavailable), func(t *testing.T) {
			err := errors.New("secret credential diagnostic")
			if unavailable {
				err = errors.Join(private.ErrUnavailable, err)
			}
			s := &Server{Remote: &failingRemote{err: err}, MaxBytes: 1024}
			defer s.Close()
			e, _ := envelope(wc.KindRoundRequest, request{Operation: "poll"})
			reply, err := s.Handle(context.Background(), e)
			if err != nil {
				t.Fatal(err)
			}
			if strings.Contains(string(reply.Payload), "secret") {
				t.Fatal("remote diagnostic crossed IPC")
			}
			var out response
			if err := private.Decode(reply.Payload, &out); err != nil {
				t.Fatal(err)
			}
			want := "rejected"
			if unavailable {
				want = "unavailable"
			}
			if out.ErrorCode != want || out.Error == "" {
				t.Fatalf("wrong classification: %+v", out)
			}
		})
	}
}

func TestLearnerIPCErrorResponseValidation(t *testing.T) {
	for _, r := range []response{
		{Error: "bad", ErrorCode: "retry_forever"},
		{ErrorCode: "unavailable"},
		{Error: "bad", Transfer: "stream"},
		{Error: "bad", Data: []byte{1}},
		{Error: "bad", Reply: &private.RelayResponse{Format: "mixlab_learner_reply_v1"}},
	} {
		if r.Validate() == nil {
			t.Fatalf("accepted ambiguous response: %+v", r)
		}
	}
	if err := (response{Error: "legacy rejection"}).Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestLearnerIPCUnavailableDoesNotCloseDispatcher(t *testing.T) {
	agent, worker := ipcPair(t)
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	done := make(chan error, 1)
	go func() {
		for i := range 2 {
			select {
			case e := <-agent.Events():
				r := response{Reply: &private.RelayResponse{Format: "mixlab_learner_reply_v1", Done: true}}
				if i == 0 {
					r = response{Error: "unavailable", ErrorCode: "unavailable"}
				}
				reply, err := envelope(wc.KindRoundResponse, r)
				if err == nil {
					err = agent.Reply(ctx, e, reply)
				}
				if err != nil {
					done <- err
					return
				}
			case <-ctx.Done():
				done <- ctx.Err()
				return
			}
		}
		done <- nil
	}()
	c := &Client{Session: worker}
	if r, err := c.Poll(ctx); err != nil || !r.Done {
		t.Fatalf("semantic error closed local dispatcher: %+v %v", r, err)
	}
	if err := <-done; err != nil {
		t.Fatal(err)
	}
}
