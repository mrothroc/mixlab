//go:build darwin || linux

package learneripc

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"syscall"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/artifact"
	private "github.com/mrothroc/mixlab/diloco/learner"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/statehome"
	wc "github.com/mrothroc/mixlab/workercontrol"
	"github.com/mrothroc/mixlab/workercontrol/local"
)

type fakeRemote struct {
	data, uploaded []byte
	a              run.Assignment
}

func (f *fakeRemote) Poll(context.Context) (private.RelayResponse, error) {
	return private.RelayResponse{Format: "mixlab_learner_reply_v1", Assignment: &f.a}, nil
}
func (f *fakeRemote) Call(context.Context, private.RelayRequest) (private.RelayResponse, error) {
	return private.RelayResponse{Format: "mixlab_learner_reply_v1"}, nil
}
func (f *fakeRemote) Read(_ context.Context, _ run.Assignment, _ run.Blob, w io.Writer) error {
	_, err := w.Write(f.data)
	return err
}
func (f *fakeRemote) Write(_ context.Context, _ run.Assignment, _ run.Blob, r io.Reader) error {
	var err error
	f.uploaded, err = io.ReadAll(r)
	return err
}

func ipcPair(t *testing.T) (*local.Dispatcher, *local.Dispatcher) {
	t.Helper()
	root, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	p, err := statehome.Resolve(statehome.Options{ExactDir: filepath.Join(root, "private")}, statehome.Context{Kind: statehome.Worker})
	if err != nil {
		t.Fatal(err)
	}
	if err = p.Ensure(); err != nil {
		t.Fatal(err)
	}
	limits := local.Limits{FrameBytes: wc.MaxFrameBytes, Bytes: 32 << 20, Messages: 10000}
	s, credential, err := local.Listen(p, limits)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = s.Close() })
	fd, err := syscall.Dup(int(credential.Fd()))
	if err != nil {
		t.Fatal(err)
	}
	childCredential := os.NewFile(uintptr(fd), "test-session")
	t.Cleanup(func() { _ = childCredential.Close() })
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	type result struct {
		c   *local.Conn
		err error
	}
	ch := make(chan result, 1)
	go func() { c, e := local.Connect(ctx, s.Path(), childCredential, limits); ch <- result{c, e} }()
	c, err := s.Accept(ctx, wc.Binding{JobID: "job", AttemptID: "attempt", BinaryID: "mixlab", BuildID: "build", AssignmentDigest: [32]byte{1}}, wc.PeerIdentity{PID: os.Getpid(), UID: uint32(os.Getuid())})
	if err != nil {
		t.Fatal(err)
	}
	r := <-ch
	if r.err != nil {
		t.Fatal(r.err)
	}
	a, err := local.NewDispatcher(context.Background(), c, local.DispatcherOptions{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = a.Close() })
	b, err := local.NewDispatcher(context.Background(), r.c, local.DispatcherOptions{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = b.Close() })
	return a, b
}
func TestLearnerIPCArtifactsAndCancellation(t *testing.T) {
	b, err := os.ReadFile("../diloco/testdata/assignment.json")
	if err != nil {
		t.Fatal(err)
	}
	remote := &fakeRemote{data: bytes.Repeat([]byte("synthetic"), 3*ChunkBytes)}
	if err = json.Unmarshal(b, &remote.a); err != nil {
		t.Fatal(err)
	}
	h := sha256.Sum256(remote.data)
	blob := run.Blob{Kind: "model_f32", Ref: artifact.Ref{SHA256: hex.EncodeToString(h[:]), Bytes: uint64(len(remote.data))}}
	remote.a.BaseModel = blob
	agent, worker := ipcPair(t)
	server := &Server{Remote: remote, MaxBytes: 4 << 20}
	defer server.Close()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() {
		for {
			select {
			case e, ok := <-agent.Events():
				if !ok {
					done <- nil
					return
				}
				reply, err := server.Handle(ctx, e)
				if err == nil {
					err = agent.Reply(ctx, e, reply)
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
	defer func() {
		cancel()
		if err := <-done; err != nil {
			t.Error(err)
		}
	}()
	c := &Client{Session: worker}
	if v, err := c.Poll(ctx); err != nil || v.Assignment == nil {
		t.Fatal(v, err)
	}
	var out bytes.Buffer
	if err = c.Read(ctx, remote.a, blob, &out); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(out.Bytes(), remote.data) {
		t.Fatal("read bytes changed")
	}
	blob.Kind = "delta_f32"
	if err = c.Write(ctx, remote.a, blob, bytes.NewReader(remote.data)); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(remote.uploaded, remote.data) {
		t.Fatal("write bytes changed")
	}
	e := wc.Envelope{Kind: wc.KindCancellation, PayloadKind: "test", PayloadVersion: 1, Payload: []byte(`{}`)}
	if err = agent.Send(ctx, e); err != nil {
		t.Fatal(err)
	}
	select {
	case <-worker.Controls():
	case <-time.After(time.Second):
		t.Fatal("cancellation blocked")
	}
}

func TestLearnerIPCRejectsCorruptAndOutOfOrderStreams(t *testing.T) {
	b, _ := os.ReadFile("../diloco/testdata/assignment.json")
	remote := &fakeRemote{data: []byte("wrong")}
	_ = json.Unmarshal(b, &remote.a)
	s := &Server{Remote: remote, MaxBytes: 1 << 20}
	defer s.Close()
	q := request{Operation: "read_begin", Artifact: artifactRequest(remote.a, remote.a.BaseModel, "read")}
	e, _ := envelope(wc.KindRoundRequest, q)
	rejected, err := s.Handle(context.Background(), e)
	var rejection response
	if err != nil {
		t.Fatal(err)
	}
	if err = private.Decode(rejected.Payload, &rejection); err != nil {
		t.Fatal(err)
	}
	if rejection.Error == "" {
		t.Fatal("corrupt download accepted")
	}
	if s.stream != nil {
		t.Fatal("failed transfer leaked spool")
	}
	q.Operation = "write_begin"
	q.Artifact.Operation = "create"
	q.Artifact.Blob.Kind = "delta_f32"
	e, _ = envelope(wc.KindRoundRequest, q)
	reply, err := s.Handle(context.Background(), e)
	if err != nil {
		t.Fatal(err)
	}
	var r response
	if err = private.Decode(reply.Payload, &r); err != nil {
		t.Fatal(err)
	}
	e, _ = envelope(wc.KindRoundRequest, request{Operation: "write_chunk", Transfer: r.Transfer, Offset: 1, Data: []byte("x")})
	rejected, err = s.Handle(context.Background(), e)
	if err != nil {
		t.Fatal(err)
	}
	if err = private.Decode(rejected.Payload, &rejection); err != nil {
		t.Fatal(err)
	}
	if rejection.Error == "" {
		t.Fatal("out-of-order chunk accepted")
	}
	if s.stream != nil {
		t.Fatal("rejected transfer leaked spool")
	}
}
