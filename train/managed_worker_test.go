package train

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/data"
	"github.com/mrothroc/mixlab/distributed"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/workerhost"
	"github.com/mrothroc/mixlab/workerhost/contract"
	"github.com/mrothroc/mixlab/workerjob"
)

const managedTinyConfig = `{"name":"managed_tiny","model_dim":16,"seq_len":4,"vocab_size":32,"blocks":[{"type":"plain","heads":2},{"type":"swiglu"}],"training":{"optimizer":"adamw","steps":4,"batch_tokens":8,"seed":73,"lr":0.001,"warmup_steps":0,"weight_decay":0,"distributed":{"mode":"ddp","backend":"ring","gradient_accumulation_steps":2,"gradient_bucket_bytes":1024}}}`

func managedFixture(t *testing.T) workerjob.Assignment {
	t.Helper()
	dir := t.TempDir()
	tokens := make([]uint16, 257)
	for i := range tokens {
		tokens[i] = uint16(1 + i%30)
	}
	writeTestShard(t, dir, "train.bin", tokens)
	pattern := filepath.Join(dir, "train.bin")
	dataset, err := data.DistributedDatasetIdentity(pattern)
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := ParseArchConfig([]byte(managedTinyConfig), "managed")
	if err != nil {
		t.Fatal(err)
	}
	program, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	digest, err := workerjob.Digest(program)
	if err != nil {
		t.Fatal(err)
	}
	m, err := distributed.NewDDPGroupMembership("managed-run", "ddp", 0, "ring", []distributed.DDPGroupMember{{MemberID: "node-a/rank/0", Rank: 0}, {MemberID: "node-a/rank/1", Rank: 1}})
	if err != nil {
		t.Fatal(err)
	}
	v, err := distributed.NewLocalGroupView(m, m.OrderedMembers[0].MemberID, 0, "attempt")
	if err != nil {
		t.Fatal(err)
	}
	plan, err := inspectWorkerConfig([]byte(managedTinyConfig), strings.Repeat("a", 64))
	if err != nil {
		t.Fatal(err)
	}
	return workerjob.Assignment{Version: workerjob.Version, JobID: "job", AttemptID: "attempt", BuildID: strings.Repeat("a", 64), View: v, WeightLayoutSHA256: plan.WeightLayoutHash, OptimizerSHA256: plan.OptimizerHash,
		Config: json.RawMessage(managedTinyConfig), DatasetSelector: "synthetic", TrainPattern: pattern, DatasetSHA256: dataset, ProgramSHA256: digest,
		RingAddresses: [][]string{{"127.0.0.1:30200"}, {"127.0.0.1:30201"}}, RuntimeSeconds: 90}
}

func TestManagedAssignmentPreflight(t *testing.T) {
	a := managedFixture(t)
	ring, err := json.Marshal(a.RingAddresses)
	if err != nil {
		t.Fatal(err)
	}
	ringPath := filepath.Join(t.TempDir(), "ring.json")
	if err = os.WriteFile(ringPath, ring, 0600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("MLX_HOSTFILE", ringPath)
	t.Setenv("MLX_RANK", "0")
	if _, err := prepareManagedAssignment(a); err != nil {
		t.Fatal(err)
	}
	for _, which := range []string{"dataset", "program", "rank", "ring", "config", "single", "layout", "optimizer"} {
		t.Run(which, func(t *testing.T) {
			b := a
			switch which {
			case "layout":
				b.WeightLayoutSHA256 = strings.Repeat("f", 64)
			case "optimizer":
				b.OptimizerSHA256 = strings.Repeat("f", 64)
			case "dataset":
				b.DatasetSHA256 = strings.Repeat("f", 64)
			case "program":
				b.ProgramSHA256 = strings.Repeat("f", 64)
			case "rank":
				t.Setenv("MLX_RANK", "1")
			case "ring":
				t.Setenv("MLX_HOSTFILE", "/nonexistent")
			case "config":
				b.Config = json.RawMessage(`{"unexpected":true}`)
			case "single":
				b.Config = json.RawMessage(strings.Replace(managedTinyConfig, `,"distributed":{"mode":"ddp","backend":"ring","gradient_accumulation_steps":2,"gradient_bucket_bytes":1024}`, "", 1))
			}
			if _, err := prepareManagedAssignment(b); err == nil {
				t.Fatal("invalid assignment passed preflight")
			}
		})
	}
}

func managedPrivateDir(t *testing.T) statehome.Path {
	t.Helper()
	base := os.TempDir()
	if runtime.GOOS == "darwin" {
		base = "/private/tmp"
	}
	dir, err := os.MkdirTemp(base, "mw-")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(dir) })
	dir, err = filepath.EvalSymlinks(dir)
	if err != nil {
		t.Fatal(err)
	}
	p, err := statehome.Resolve(statehome.Options{ExactDir: dir}, statehome.Context{Kind: statehome.Worker})
	if err != nil {
		t.Fatal(err)
	}
	return p
}

// This uses the actual compiled CLI, kernel-authenticated child, strict MLX
// ring, production sampler/optimizer and real per-rank progress, not a fake.
func TestManagedWorkerCLITraining(t *testing.T) {
	cli := os.Getenv("MIXLAB_MANAGED_CLI")
	if cli == "" {
		t.Skip("set MIXLAB_MANAGED_CLI to a freshly built MLX trainer")
	}
	a := managedFixture(t)
	id, err := workerjob.FileDigest(cli)
	if err != nil {
		t.Fatal(err)
	}
	a.BuildID = id
	s, err := workerhost.New(cli, id)
	if err != nil {
		t.Fatal(err)
	}
	// Tests choose two unused listener ports without relying on fixed CI ports.
	ports := managedRingPorts(t)
	for i, p := range ports {
		a.RingAddresses[i][0] = fmt.Sprintf("127.0.0.1:%d", p)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	type outcome struct {
		r   workerhost.Result
		err error
		dir string
	}
	results := make(chan outcome, 2)
	for rank := 0; rank < 2; rank++ {
		b := a
		b.View.LocalRank = rank
		b.View.LocalMemberID = b.View.Membership.OrderedMembers[rank].MemberID
		p := workerhost.LaunchPlan{Assignment: b, Directory: managedPrivateDir(t), StartupTimeout: 60 * time.Second, ShutdownGrace: 3 * time.Second,
			Limits: contract.Limits{CPUSeconds: 3600, MemoryBytes: 8 << 30, DiskBytes: 1 << 30, LogBytes: 1 << 20}}
		go func() { r, e := s.Run(ctx, p); results <- outcome{r, e, p.Directory.Dir()} }()
	}
	var losses []float64
	for range 2 {
		out := <-results
		if out.err != nil {
			cancel()
		}
		if out.err != nil || !out.r.Ready || out.r.Progress.Step != 4 || out.r.Progress.Committed != 4 || out.r.Progress.Loss <= 0 {
			log, _ := os.ReadFile(filepath.Join(out.dir, "worker.log"))
			t.Errorf("managed worker: %+v err=%v\n%s", out.r, out.err, log)
		}
		losses = append(losses, out.r.Progress.Loss)
	}
	if losses[0] != losses[1] {
		t.Fatalf("ranks disagree: %v", losses)
	}
}

func TestManagedWorkerCLIRejectsAssignmentMismatch(t *testing.T) {
	cli := os.Getenv("MIXLAB_MANAGED_CLI")
	if cli == "" {
		t.Skip("set MIXLAB_MANAGED_CLI to a freshly built trainer")
	}
	id, err := workerjob.FileDigest(cli)
	if err != nil {
		t.Fatal(err)
	}
	s, err := workerhost.New(cli, id)
	if err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{"dataset", "program"} {
		t.Run(field, func(t *testing.T) {
			a := managedFixture(t)
			a.BuildID = id
			if field == "dataset" {
				a.DatasetSHA256 = strings.Repeat("f", 64)
			} else {
				a.ProgramSHA256 = strings.Repeat("f", 64)
			}
			p := workerhost.LaunchPlan{Assignment: a, Directory: managedPrivateDir(t), StartupTimeout: 10 * time.Second, ShutdownGrace: 3 * time.Second,
				Limits: contract.Limits{CPUSeconds: 3600, MemoryBytes: 8 << 30, DiskBytes: 1 << 30, LogBytes: 1 << 20}}
			r, err := s.Run(context.Background(), p)
			if err == nil || r.Ready || !strings.Contains(err.Error(), field+" digest mismatch") {
				log, _ := os.ReadFile(filepath.Join(p.Directory.Dir(), "worker.log"))
				t.Fatalf("assignment did not fail before training: %+v %v\n%s", r, err, log)
			}
		})
	}
}
