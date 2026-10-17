//go:build darwin || linux

package learneripc

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/artifact"
	private "github.com/mrothroc/mixlab/diloco/learner"
	"github.com/mrothroc/mixlab/diloco/run"
	wc "github.com/mrothroc/mixlab/workercontrol"
)

type diagnosticRemote struct {
	err   error
	calls int
}

func (r *diagnosticRemote) Poll(context.Context) (private.RelayResponse, error) {
	r.calls++
	return private.RelayResponse{Format: "mixlab_learner_reply_v1"}, r.err
}
func (r *diagnosticRemote) Call(context.Context, private.RelayRequest) (private.RelayResponse, error) {
	r.calls++
	return private.RelayResponse{Format: "mixlab_learner_reply_v1"}, r.err
}
func (r *diagnosticRemote) Read(context.Context, run.Assignment, run.Blob, io.Writer) error {
	r.calls++
	return r.err
}
func (r *diagnosticRemote) Write(context.Context, run.Assignment, run.Blob, io.Reader) error {
	r.calls++
	return r.err
}
func (r *diagnosticRemote) ReadEvaluation(context.Context, run.EvaluationAssignment, io.Writer) error {
	r.calls++
	return r.err
}

func TestLearnerIPCLocalErrorDiagnostics(t *testing.T) {
	var assignment run.Assignment
	encoded, err := os.ReadFile("../diloco/testdata/assignment.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(encoded, &assignment); err != nil {
		t.Fatal(err)
	}
	var evaluation run.EvaluationAssignment
	encoded, err = os.ReadFile("../diloco/testdata/evaluation-assignment.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(encoded, &evaluation); err != nil {
		t.Fatal(err)
	}
	data := []byte("private artifact payload must not enter diagnostics")
	hash := sha256.Sum256(data)
	blob := run.Blob{Kind: "model_f32", Ref: artifact.Ref{SHA256: hex.EncodeToString(hash[:]), Bytes: uint64(len(data))}}
	assignment.BaseModel, evaluation.Model = blob, blob
	for _, operation := range []string{"poll", "call/start", "read_begin", "read_begin/evaluation", "write_end"} {
		for _, unavailable := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/unavailable=%v", operation, unavailable), func(t *testing.T) {
				cause := errors.New("protected credential diagnostic")
				remoteErr := errors.Join(cause, io.ErrUnexpectedEOF)
				if unavailable {
					remoteErr = errors.Join(private.ErrUnavailable, remoteErr)
				}
				remote := &diagnosticRemote{err: remoteErr}
				notifications := 0
				s := &Server{Remote: remote, MaxBytes: 1024, OnError: func(op string, err error) {
					notifications++
					if op != operation || err != remoteErr || !errors.Is(err, cause) || !errors.Is(err, io.ErrUnexpectedEOF) || errors.Is(err, private.ErrUnavailable) != unavailable {
						t.Fatalf("local diagnostic lost operation/original error: operation=%q err=%v", op, err)
					}
				}}
				defer s.Close()
				send := func(q request) response {
					t.Helper()
					e, err := envelope(wc.KindRoundRequest, q)
					if err != nil {
						t.Fatal(err)
					}
					reply, err := s.Handle(context.Background(), e)
					if err != nil {
						t.Fatal(err)
					}
					for _, forbidden := range []string{cause.Error(), io.ErrUnexpectedEOF.Error(), string(data)} {
						if strings.Contains(string(reply.Payload), forbidden) {
							t.Fatal("local diagnostic or artifact bytes crossed IPC")
						}
					}
					var out response
					if err := private.Decode(reply.Payload, &out); err != nil {
						t.Fatal(err)
					}
					return out
				}
				q := request{Operation: operation}
				switch operation {
				case "call/start":
					hash, err := run.Digest(assignment)
					if err != nil {
						t.Fatal(err)
					}
					start := private.Start{Format: "mixlab_learner_start_v1", Binding: assignment.Binding, WorkerID: assignment.WorkerID, AssignmentHash: hash, PrivateHash: strings.Repeat("c", 64)}
					q.Operation = "call"
					q.Call = &private.RelayRequest{Format: "mixlab_learner_relay_v1", Binding: assignment.Binding, WorkerID: assignment.WorkerID, Operation: "start", Start: &start}
				case "read_begin":
					q.Artifact = artifactRequest(assignment, blob, "read")
				case "read_begin/evaluation":
					q.Operation = "read_begin"
					q.Artifact = &private.ArtifactRequest{Evaluation: &evaluation, Blob: blob, Operation: "read"}
				case "write_end":
					upload := blob
					upload.Kind = "delta_f32"
					begin := send(request{Operation: "write_begin", Artifact: artifactRequest(assignment, upload, "create")})
					if begin.Error != "" || begin.Transfer == "" {
						t.Fatalf("upload did not begin: %+v", begin)
					}
					if chunk := send(request{Operation: "write_chunk", Transfer: begin.Transfer, Data: data}); chunk.Error != "" {
						t.Fatalf("upload did not spool: %+v", chunk)
					}
					if notifications != 0 || remote.calls != 0 {
						t.Fatal("successful local spooling logged or replayed a remote call")
					}
					q.Transfer, q.Offset = begin.Transfer, uint64(len(data))
				}
				out := send(q)
				want := response{Error: "operation rejected", ErrorCode: "rejected"}
				if unavailable {
					want = response{Error: "coordinator unavailable after bounded retry", ErrorCode: "unavailable"}
				}
				if out.Error != want.Error || out.ErrorCode != want.ErrorCode || out.Reply != nil || out.Transfer != "" || len(out.Data) != 0 {
					t.Fatalf("diagnostic changed sanitized response: %+v", out)
				}
				if notifications != 1 || remote.calls != 1 || s.stream != nil {
					t.Fatalf("diagnostic duplicated work or retained failed stream: notifications=%d calls=%d stream=%v", notifications, remote.calls, s.stream != nil)
				}
			})
		}
	}
}

func TestLearnerIPCDiagnosticsExcludeSuccessAndInvalidPayload(t *testing.T) {
	remote := &diagnosticRemote{}
	notifications := 0
	s := &Server{Remote: remote, MaxBytes: 1024, OnError: func(string, error) { notifications++ }}
	defer s.Close()
	e, err := envelope(wc.KindRoundRequest, request{Operation: "poll"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.Handle(context.Background(), e); err != nil {
		t.Fatal(err)
	}
	e, err = envelope(wc.KindRoundRequest, request{Operation: "unvalidated private payload"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.Handle(context.Background(), e); err == nil {
		t.Fatal("invalid operation accepted")
	}
	if notifications != 0 || remote.calls != 1 {
		t.Fatalf("success or unvalidated request entered error diagnostics: notifications=%d calls=%d", notifications, remote.calls)
	}
}
