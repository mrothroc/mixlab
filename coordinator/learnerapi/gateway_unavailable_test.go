package learnerapi

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestLearnerErrorRequiresCanceledContext(t *testing.T) {
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	expired, stop := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
	defer stop()
	for _, status := range []int{400, 403, 409, 500} {
		for _, ctx := range []context.Context{context.Background(), canceled, expired} {
			w := httptest.NewRecorder()
			learnerError(w, ctx, "policy rejected", status)
			wantStatus, wantBody := status, "policy rejected\n"
			if ctx.Err() != nil {
				wantStatus, wantBody = http.StatusServiceUnavailable, "learner service unavailable\n"
			}
			if w.Code != wantStatus || w.Body.String() != wantBody {
				t.Fatalf("ctx=%v: %d %q, want %d %q", ctx.Err(), w.Code, w.Body.String(), wantStatus, wantBody)
			}
		}
	}
}

type boundedLearnerRecorder struct{ *httptest.ResponseRecorder }

func (boundedLearnerRecorder) SetReadDeadline(time.Time) error  { return nil }
func (boundedLearnerRecorder) SetWriteDeadline(time.Time) error { return nil }

type cancelLearnerBody struct{ cancel context.CancelFunc }

func (b cancelLearnerBody) Read([]byte) (int, error) { b.cancel(); return 0, io.ErrUnexpectedEOF }
func (cancelLearnerBody) Close() error               { return nil }

func TestLearnerGatewayCanceledRequestBody(t *testing.T) {
	for _, phase := range []string{"before-read", "during-read", "live-malformed"} {
		t.Run(phase, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			r := httptest.NewRequest(http.MethodPost, Route, strings.NewReader("invalid")).WithContext(ctx)
			switch phase {
			case "before-read":
				cancel()
			case "during-read":
				r.Body = cancelLearnerBody{cancel}
			}
			w := httptest.NewRecorder()
			(Gateway{Timeout: time.Second}).serve(boundedLearnerRecorder{w}, r)
			want := http.StatusServiceUnavailable
			if phase == "live-malformed" {
				want = http.StatusBadRequest
			}
			if w.Code != want {
				t.Fatalf("status %d, want %d: %s", w.Code, want, w.Body)
			}
		})
	}
}
