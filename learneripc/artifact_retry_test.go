//go:build darwin || linux

package learneripc

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"os"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/artifact"
	private "github.com/mrothroc/mixlab/diloco/learner"
	"github.com/mrothroc/mixlab/diloco/run"
	wc "github.com/mrothroc/mixlab/workercontrol"
)

func artifactRetryFixture(t *testing.T) *fakeRemote {
	t.Helper()
	f := &fakeRemote{data: bytes.Repeat([]byte("synthetic"), ChunkBytes)}
	b, err := os.ReadFile("../diloco/testdata/assignment.json")
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(b, &f.a); err != nil {
		t.Fatal(err)
	}
	h := sha256.Sum256(f.data)
	f.a.BaseModel = run.Blob{Kind: "model_f32", Ref: artifact.Ref{SHA256: hex.EncodeToString(h[:]), Bytes: uint64(len(f.data))}}
	return f
}

func artifactRetryClient(t *testing.T, handle func(context.Context, wc.Envelope) (wc.Envelope, error)) (*Client, func()) {
	t.Helper()
	agent, worker := ipcPair(t)
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		for {
			select {
			case e, ok := <-agent.Events():
				if !ok {
					done <- io.EOF
					return
				}
				r, err := handle(ctx, e)
				if err == nil {
					err = agent.Reply(ctx, e, r)
				}
				if err != nil {
					done <- err
					return
				}
			case <-ctx.Done():
				done <- nil
				return
			}
		}
	}()
	stop := sync.OnceFunc(func() {
		cancel()
		if err := <-done; err != nil && !errors.Is(err, context.Canceled) {
			t.Error(err)
		}
	})
	t.Cleanup(stop)
	return &Client{Session: worker}, stop
}

// A failed remote read may already have written to the private spool, but none
// of those bytes reached the learner. Retrying must discard the entire spool.
type interruptedDownload struct {
	*fakeRemote
	calls int
}

func (f *interruptedDownload) Read(ctx context.Context, a run.Assignment, b run.Blob, w io.Writer) error {
	f.calls++
	if f.calls <= 2 {
		if _, err := w.Write([]byte("discard this partial download")); err != nil {
			return err
		}
		return private.ErrUnavailable
	}
	return f.fakeRemote.Read(ctx, a, b, w)
}

func (f *interruptedDownload) ReadEvaluation(ctx context.Context, a run.EvaluationAssignment, w io.Writer) error {
	return f.Read(ctx, run.Assignment{}, a.Model, w)
}

func TestArtifactBeginRetryDiscardsPartialRemoteDownload(t *testing.T) {
	for _, evaluation := range []bool{false, true} {
		t.Run(map[bool]string{false: "learner", true: "evaluation"}[evaluation], func(t *testing.T) {
			testArtifactBeginRetryDownload(t, evaluation)
		})
	}
}

func testArtifactBeginRetryDownload(t *testing.T, evaluation bool) {
	t.Helper()
	f := &interruptedDownload{fakeRemote: artifactRetryFixture(t)}
	s := &Server{Remote: f, MaxBytes: 4 << 20}
	defer s.Close()
	c, stop := artifactRetryClient(t, s.Handle)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	var out bytes.Buffer
	var err error
	if evaluation {
		var a run.EvaluationAssignment
		b, readErr := os.ReadFile("../diloco/testdata/evaluation-assignment.json")
		if readErr != nil {
			t.Fatal(readErr)
		}
		if err = json.Unmarshal(b, &a); err != nil {
			t.Fatal(err)
		}
		a.Model = f.a.BaseModel
		err = c.ReadEvaluation(ctx, a, &out)
	} else {
		err = c.Read(ctx, f.a, f.a.BaseModel, &out)
	}
	stop()
	if err != nil || f.calls != 3 || !bytes.Equal(out.Bytes(), f.data) || s.stream != nil {
		t.Fatalf("download retry lost data or leaked spool: err=%v calls=%d bytes=%d", err, f.calls, out.Len())
	}
}

func TestArtifactBeginRetriesOnlyExplicitUnavailable(t *testing.T) {
	for _, operation := range []string{"read_begin", "write_begin"} {
		for _, failure := range []string{"unavailable", "rejected", "canceled"} {
			t.Run(operation+"/"+failure, func(t *testing.T) {
				f := artifactRetryFixture(t)
				s := &Server{Remote: f, MaxBytes: 4 << 20}
				defer s.Close()
				calls := 0
				var first request
				ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
				defer cancel()
				c, stop := artifactRetryClient(t, func(ctx context.Context, e wc.Envelope) (wc.Envelope, error) {
					var q request
					if err := private.Decode(e.Payload, &q); err != nil {
						return wc.Envelope{}, err
					}
					if q.Operation == operation {
						calls++
						if calls == 1 {
							first = q
						} else if !reflect.DeepEqual(q, first) {
							t.Error("retry changed artifact identity or assignment")
						}
						if calls == 1 || failure == "canceled" {
							code := failure
							if failure == "canceled" {
								cancel()
								code = "unavailable"
							}
							return envelope(wc.KindRoundResponse, response{Error: "injected", ErrorCode: code})
						}
					}
					return s.Handle(ctx, e)
				})
				var out bytes.Buffer
				reader := bytes.NewReader(f.data)
				var err error
				if operation == "read_begin" {
					err = c.Read(ctx, f.a, f.a.BaseModel, &out)
				} else {
					blob := f.a.BaseModel
					blob.Kind = "delta_f32"
					err = c.Write(ctx, f.a, blob, reader)
				}
				stop()
				if failure == "unavailable" {
					if err != nil || calls != 2 {
						t.Fatalf("begin did not retry: calls=%d err=%v", calls, err)
					}
					if operation == "read_begin" && !bytes.Equal(out.Bytes(), f.data) || operation == "write_begin" && !bytes.Equal(f.uploaded, f.data) {
						t.Fatal("retry changed transferred bytes")
					}
				} else if err == nil || calls != 1 || out.Len() != 0 || reader.Len() != len(f.data) {
					t.Fatalf("terminal begin consumed/replayed stream: calls=%d err=%v", calls, err)
				}
			})
		}
	}
}

func TestArtifactConsumedStreamNeverRetries(t *testing.T) {
	for _, operation := range []string{"read_chunk", "read_end", "write_chunk", "write_end"} {
		t.Run(operation, func(t *testing.T) {
			f := artifactRetryFixture(t)
			s := &Server{Remote: f, MaxBytes: 4 << 20}
			defer s.Close()
			calls := 0
			c, stop := artifactRetryClient(t, func(ctx context.Context, e wc.Envelope) (wc.Envelope, error) {
				var q request
				if err := private.Decode(e.Payload, &q); err != nil {
					return wc.Envelope{}, err
				}
				if q.Operation == operation {
					calls++
					return envelope(wc.KindRoundResponse, response{Error: "injected", ErrorCode: "unavailable"})
				}
				return s.Handle(ctx, e)
			})
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()
			var err error
			if strings.HasPrefix(operation, "read") {
				err = c.Read(ctx, f.a, f.a.BaseModel, io.Discard)
			} else {
				blob := f.a.BaseModel
				blob.Kind = "delta_f32"
				err = c.Write(ctx, f.a, blob, bytes.NewReader(f.data))
			}
			stop()
			if !errors.Is(err, private.ErrUnavailable) || calls != 1 {
				t.Fatalf("partially consumed stream retried: calls=%d err=%v", calls, err)
			}
		})
	}
}
