//go:build darwin || linux

package workerhost

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/artifact"
	"github.com/mrothroc/mixlab/artifact/tensors"
	private "github.com/mrothroc/mixlab/diloco/learner"
	"github.com/mrothroc/mixlab/diloco/optimizer"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/distributed"
	"github.com/mrothroc/mixlab/learner"
	"github.com/mrothroc/mixlab/precision"
	"github.com/mrothroc/mixlab/statehome"
	wc "github.com/mrothroc/mixlab/workercontrol"
	"github.com/mrothroc/mixlab/workercontrol/local"
	"github.com/mrothroc/mixlab/workerhost/contract"
	"github.com/mrothroc/mixlab/workerjob"
)

// This child uses real learner storage/runtime under the real guardian. Only
// numerical work and the coordinator peer are synthetic; neither owns storage.
func helperManagedLearnerRuntime(ctx context.Context, conn *local.Conn, a workerjob.LearnerAssignment, path statehome.Path) int {
	send := func(seq uint64, kind wc.Kind, event workerjob.Event) error {
		e, err := workerjob.Envelope(conn.Binding(), seq, kind, event)
		if err != nil {
			return err
		}
		return conn.Send(ctx, e)
	}
	if send(1, wc.KindReadiness, workerjob.Event{}) != nil {
		return 80
	}
	err := executeManagedStorage(ctx, a, path)
	event := workerjob.Event{}
	if err != nil {
		event.Error = err.Error()
	}
	if send(2, wc.KindTerminalOutcome, event) != nil {
		return 81
	}
	return 0
}

func executeManagedStorage(ctx context.Context, a workerjob.LearnerAssignment, path statehome.Path) error {
	s, err := learner.Open(path, a.PrivateMaxBytes)
	if err != nil {
		return err
	}
	model := []optimizer.Tensor{{Name: "w0_weight", Shape: []int{1}, Values: []float32{4}}}
	var base bytes.Buffer
	if err := tensors.Write(&base, model); err != nil {
		return err
	}
	h := sha256.Sum256(base.Bytes())
	ref := artifact.Ref{SHA256: hex.EncodeToString(h[:]), Bytes: uint64(base.Len())}
	policy := precision.Policy{ComputeDType: "float32", Backend: "metal"}
	ph, err := policy.Digest()
	if err != nil {
		return err
	}
	round := run.Assignment{Format: "mixlab_round_assignment_v1", ID: "storage-round",
		Binding:     run.Binding{RunID: a.RunID, PlanHash: a.PlanHash, CommitHash: a.PlanHash, Generation: a.Cohort.Generation, CohortHash: a.Cohort.CohortHash},
		Coordinator: a.Coordinator, WorkerID: a.Member.WorkerID, PartitionID: a.Member.DatasetPartitionID, Contract: a.Contract,
		BaseModel: run.Blob{Ref: ref, Kind: "model_f32"}, OuterState: run.Blob{Ref: ref, Kind: "outer_state_f32"},
		InnerSteps: 2, PrecisionHash: ph, IssuedAt: 1200, ExpiresAt: 1800}
	r := learner.Runtime{Store: s, Peer: &storagePeer{base.Bytes()}, Now: func() int64 { return 1300 },
		Scope: learner.Scope{RunID: a.RunID, PlanHash: a.PlanHash, WorkerID: a.Member.WorkerID, PartitionID: a.Member.DatasetPartitionID,
			ParticipantID: a.Member.ParticipantID, IslandID: a.Member.IslandID, Contract: a.Contract, Coordinator: a.Coordinator,
			CohortHash: a.Cohort.CohortHash, Generation: a.Cohort.Generation, ExpectedPrivateHash: a.ExpectedPrivateHash}}
	r.Factory = func(model []optimizer.Tensor) (learner.Engine, error) {
		if a.ExpectedPrivateHash != "" {
			return nil, fmt.Errorf("replay unexpectedly rebuilt numerical engine")
		}
		return &storageEngine{learner.Snapshot{Model: model, Optimizer: []byte("moment"), Sampler: []byte("cursor"), Schedule: []byte("schedule"), Metadata: []byte("metadata"), Precision: policy,
			Replay: private.Replay{Version: "mixlab_learner_rng_v1", RunID: a.RunID, WorkerID: a.Member.WorkerID, PartitionID: a.Member.DatasetPartitionID, Seed: 7}}}, nil
	}
	_, err = r.Execute(ctx, round)
	return err
}

type storageEngine struct{ v learner.Snapshot }

func (e *storageEngine) Snapshot() (learner.Snapshot, error) { return e.v, nil }
func (e *storageEngine) Restore(v learner.Snapshot) error    { e.v = v; return nil }
func (e *storageEngine) Install(v []optimizer.Tensor) error  { e.v.Model = v; return nil }
func (e *storageEngine) Close()                              {}
func (e *storageEngine) Train(_ context.Context, n uint64) (learner.Counts, error) {
	e.v.Replay.AttemptStep += n
	e.v.Model = []optimizer.Tensor{{Name: "w0_weight", Shape: []int{1}, Values: []float32{3}}}
	e.v.Optimizer, e.v.Sampler = []byte("moment-after"), []byte("cursor-after")
	return learner.Counts{Attempted: n, Committed: n, Tokens: n * 8, Examples: n * 2, Loss: 2}, nil
}

type storagePeer struct{ base []byte }

func (p *storagePeer) Read(_ context.Context, _ run.Assignment, _ run.Blob, w io.Writer) error {
	_, err := w.Write(p.base)
	return err
}
func (p *storagePeer) Write(ctx context.Context, _ run.Assignment, blob run.Blob, r io.Reader) error {
	return artifact.Copy(ctx, io.Discard, r, blob.Ref)
}
func (p *storagePeer) Start(context.Context, private.Start) error { return nil }
func (p *storagePeer) Submit(_ context.Context, u private.Update) (private.Receipt, error) {
	h, err := run.Digest(u)
	return private.Receipt{Format: "mixlab_update_receipt_v1", Binding: u.Binding, UpdateID: u.ID, UpdateHash: h, PrivateHash: u.PrivateHash}, err
}

func TestManagedLearnerRuntimeStoreStartupAndReplay(t *testing.T) {
	supervisor, p := learnerPlan(t, "runtime-store-startup")
	// Keep Unix socket paths short, as in the production runtime root.
	root, err := statehome.Resolve(statehome.Options{ExactDir: filepath.Join(privateDirectory(t).Dir(), "runtime")}, statehome.Context{Kind: statehome.Worker})
	if err != nil {
		t.Fatal(err)
	}
	store, err := InitializeRuntimeStore(root)
	if err != nil {
		t.Fatal(err)
	}
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	runtime, err := NewGuardianRuntime(store, exe, supervisor.build, exe, supervisor.build, 5*time.Second, 2*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	a := p.Learner
	a.JobID, a.RunID, a.Member.WorkerID = strings.Repeat("a", 32), strings.Repeat("b", 32), strings.Repeat("c", 32)
	cohort, err := distributed.NewLearnerCohort(a.RunID, 0, []distributed.LearnerMember{a.Member})
	if err != nil {
		t.Fatal(err)
	}
	a.Cohort = &cohort
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	path, err := runtime.LearnerPrivate(ctx, a.RunID, a.Member.WorkerID)
	if err != nil {
		t.Fatal(err)
	}
	identityPath := filepath.Join(filepath.Dir(path.Dir()), "runtime-identity")
	identity, err := os.ReadFile(identityPath)
	if err != nil {
		t.Fatal(err)
	}
	var head []byte
	for i := 0; i < 3; i++ {
		a.AttemptID = strings.Repeat(fmt.Sprintf("%x", i+1), 32)
		if i == 2 {
			// A marker inside learner storage is still an ownership violation.
			if err := path.WriteFile("runtime-identity", []byte("foreign")); err != nil {
				t.Fatal(err)
			}
		}
		host, runner, err := runtime.OpenLearner(ctx, a.JobID, a.AttemptID, p.Relay, path)
		if err != nil {
			t.Fatal(err)
		}
		approved := contract.Approved{Version: contract.LearnerVersion, ManifestHash: a.PlanHash, TransportHash: a.PlanHash, Learner: a, Limits: p.Limits}
		out, err := host.Run(ctx, approved, runner)
		if err != nil || !out.NoChild || out.PID == 0 {
			t.Fatalf("managed startup: %+v %v", out, err)
		}
		if i == 2 {
			if out.Kind != contract.Failed || !strings.Contains(out.Error, `unexpected learner private file "runtime-identity"`) {
				t.Fatalf("expected ownership rejection, got %+v", out)
			}
			b, err := path.ReadFileLimit("runtime-identity", 128)
			if err != nil || string(b) != "foreign" {
				t.Fatal("foreign marker removed", err)
			}
			break
		}
		if out.Kind != contract.Exited {
			t.Fatalf("learner startup failed: %+v", out)
		}
		b, err := path.ReadFileLimit("head.json", 4<<20)
		if err != nil {
			t.Fatal(err)
		}
		if i == 0 {
			head = b
			var j struct {
				Receipt private.Receipt `json:"receipt"`
			}
			if err := json.Unmarshal(b, &j); err != nil {
				t.Fatal(err)
			}
			if err := j.Receipt.Validate(); err != nil {
				t.Fatal(err)
			}
			a.ExpectedPrivateHash = j.Receipt.PrivateHash
		} else if !bytes.Equal(head, b) {
			t.Fatal("replay changed private journal")
		}
		// Reopen the production allocator, not a manually constructed path.
		reopened, err := OpenRuntimeStore(root)
		if err != nil {
			t.Fatal(err)
		}
		path, err = reopened.LearnerPrivate(ctx, a.RunID, a.Member.WorkerID)
		if err != nil {
			t.Fatal(err)
		}
		privateBytes, err := attemptDiskBytes(ctx, path.Dir())
		if err != nil {
			t.Fatal(err)
		}
		p.LearnerPrivateDirectory, p.Learner.PrivateMaxBytes = path, uint64(privateBytes)
		total, err := p.withPrivateDisk(ctx, 123)
		if err != nil || total != privateBytes+123 {
			t.Fatalf("private budget accounting: %d %v", total, err)
		}
		p.Learner.PrivateMaxBytes--
		if _, err := p.withPrivateDisk(ctx, 123); err == nil {
			t.Fatal("private budget bypassed")
		}
		p.Learner.PrivateMaxBytes = 1 << 20
	}
	if b, err := os.ReadFile(identityPath); err != nil || !bytes.Equal(b, identity) {
		t.Fatal("host identity changed", err)
	}
}
