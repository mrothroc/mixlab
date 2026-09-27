package train

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/artifact"
	"github.com/mrothroc/mixlab/artifact/checkpoint"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/workerhost"
	"github.com/mrothroc/mixlab/workerhost/contract"
	"github.com/mrothroc/mixlab/workerjob"
)

func TestManagedCheckpointRejectsCorruptedInput(t *testing.T) {
	var bundle bytes.Buffer
	err := checkpoint.Write(context.Background(), &bundle, [3]checkpoint.Member{
		{Size: 2, Reader: bytes.NewBufferString("{}")},
		{Size: 5, Reader: bytes.NewBufferString("model")},
		{Size: 5, Reader: bytes.NewBufferString("state")},
	})
	if err != nil {
		t.Fatal(err)
	}
	original := bytes.Clone(bundle.Bytes())
	h := sha256.Sum256(original)
	for _, kind := range []string{"digest", "truncated", "invalid-container"} {
		t.Run(kind, func(t *testing.T) {
			source, work := managedPrivateDir(t), managedPrivateDir(t)
			t.Chdir(work.Dir())
			a := managedFixture(t)
			a.ResumePath = filepath.Join(source.Dir(), checkpoint.File)
			a.Resume = &artifact.Ref{SHA256: hex.EncodeToString(h[:]), Bytes: uint64(len(original))}
			b := bytes.Clone(original)
			switch kind {
			case "digest":
				b[len(b)-1] ^= 1
			case "truncated":
				b = b[:len(b)-1]
			case "invalid-container":
				b[0] ^= 1
				digest := sha256.Sum256(b)
				a.Resume.SHA256 = hex.EncodeToString(digest[:])
			}
			if err := source.WriteFile(checkpoint.File, b); err != nil {
				t.Fatal(err)
			}
			if _, err := prepareManagedResume(context.Background(), a); err == nil {
				t.Fatal("corrupt checkpoint accepted")
			}
			if _, err := os.Stat(filepath.Join(work.Dir(), "managed-resume")); !os.IsNotExist(err) {
				t.Fatal("failed extraction published state", err)
			}
		})
	}
}

func TestManagedCheckpointStopValidation(t *testing.T) {
	for _, tc := range []struct {
		at           uint64
		start, steps int
		ok           bool
	}{{0, 0, 4, true}, {2, 0, 4, true}, {4, 2, 4, true}, {2, 2, 4, false}, {1, 2, 4, false}, {5, 0, 4, false}} {
		opts := TrainOptions{managed: &managedTraining{assignment: workerjob.Assignment{CheckpointAt: tc.at}}}
		_, err := managedCheckpointStop(opts, tc.start, tc.steps)
		if (err == nil) != tc.ok {
			t.Fatal(tc, err)
		}
	}
}

func unpackManagedTestBundle(t *testing.T, path string) string {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = f.Close() }()
	p := managedPrivateDir(t)
	err = checkpoint.Read(context.Background(), f, func(name string, n uint64, r io.Reader) error {
		return p.PublishStream(name, int64(n), func(w io.Writer) error { _, err := io.Copy(w, r); return err })
	})
	if err != nil {
		t.Fatal(err)
	}
	return filepath.Join(p.Dir(), checkpoint.Manifest)
}

// Real kernel-authenticated child processes, two native MLX ranks, model and
// optimizer-state files, direct sampler restore, and a new execution attempt.
func TestManagedCheckpointResumeCLIParity(t *testing.T) {
	cli := os.Getenv("MIXLAB_MANAGED_CLI")
	if cli == "" {
		t.Skip("set MIXLAB_MANAGED_CLI to the matched MLX binary")
	}
	a := managedFixture(t)
	id, err := workerjob.FileDigest(cli)
	if err != nil {
		t.Fatal(err)
	}
	a.BuildID = id
	a.OutputMaxBytes = 8 << 20
	s, err := workerhost.New(cli, id)
	if err != nil {
		t.Fatal(err)
	}
	type result struct {
		out workerhost.Result
		dir statehome.Path
		err error
	}
	run := func(base workerjob.Assignment, name string, stop uint64) string {
		t.Helper()
		base.JobID, base.AttemptID, base.View.LaunchAttemptID = name, name, name
		base.CheckpointAt = stop
		base.RingAddresses = [][]string{{""}, {""}}
		for i, port := range managedRingPorts(t) {
			base.RingAddresses[i][0] = fmt.Sprintf("127.0.0.1:%d", port)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
		defer cancel()
		results := make(chan result, 2)
		for rank := range 2 {
			b := base
			b.View.LocalRank = rank
			b.View.LocalMemberID = b.View.Membership.OrderedMembers[rank].MemberID
			dir := managedPrivateDir(t)
			p := workerhost.LaunchPlan{Assignment: b, Directory: dir, StartupTimeout: 60 * time.Second, ShutdownGrace: 3 * time.Second, Limits: contract.Limits{CPUSeconds: 3600, MemoryBytes: 8 << 30, DiskBytes: 1 << 30, LogBytes: 1 << 20}}
			go func() { out, err := s.Run(ctx, p); results <- result{out, dir, err} }()
		}
		var source, manifest string
		for range 2 {
			r := <-results
			if r.err != nil {
				cancel()
				log, _ := os.ReadFile(filepath.Join(r.dir.Dir(), "worker.log"))
				t.Errorf("managed %s failed: %v\n%s", name, r.err, log)
				continue
			}
			if r.out.Output.Bytes > 0 {
				source = filepath.Join(r.dir.Dir(), r.out.Output.SHA256)
				manifest = unpackManagedTestBundle(t, source)
				if name == "split" {
					a.Resume = &r.out.Output
					a.ResumePath = source
				}
			}
		}
		if t.Failed() {
			t.FailNow()
		}
		if source == "" {
			t.Fatal("missing checkpoint output")
		}
		return manifest
	}
	baseline := run(a, "baseline", 4)
	first := run(a, "split", 2)
	initial, err := readDistributedResumeManifest(first)
	if err != nil {
		t.Fatal(err)
	}
	if initial.GlobalOptimizerAttempt != 2 || initial.Schedule.OriginalTotalSteps != 4 {
		t.Fatal("checkpoint changed counters or schedule", initial.GlobalOptimizerAttempt, initial.Schedule)
	}
	resumed := run(a, "resumed", 4)
	x, err := readDistributedResumeManifest(baseline)
	if err != nil {
		t.Fatal(err)
	}
	y, err := readDistributedResumeManifest(resumed)
	if err != nil {
		t.Fatal(err)
	}
	if x.GlobalOptimizerAttempt != y.GlobalOptimizerAttempt || x.GlobalCommittedStep != y.GlobalCommittedStep || !reflect.DeepEqual(x.Sampler, y.Sampler) || !reflect.DeepEqual(x.Schedule, y.Schedule) {
		t.Fatal("resume counters/sampler/schedule differ")
	}
	if x.Topology.LaunchAttemptID == y.Topology.LaunchAttemptID || !reflect.DeepEqual(x.Topology.OrderedMembers, y.Topology.OrderedMembers) {
		t.Fatal("resume did not use a new attempt with the same members")
	}
	cfg, err := ParseArchConfig(a.Config, "managed")
	if err != nil {
		t.Fatal(err)
	}
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatal(err)
	}
	wx, err := loadSafetensorsWeights(filepath.Join(filepath.Dir(baseline), x.ModelFile), shapes)
	if err != nil {
		t.Fatal(err)
	}
	wy, err := loadSafetensorsWeights(filepath.Join(filepath.Dir(resumed), y.ModelFile), shapes)
	if err != nil {
		t.Fatal(err)
	}
	compare := func(x, y []float32) {
		t.Helper()
		if len(x) != len(y) {
			t.Fatal("tensor size mismatch")
		}
		for i, v := range x {
			if math.IsNaN(float64(y[i])) || math.Abs(float64(v-y[i])) > 1e-6 {
				t.Fatalf("resume tensor differs at %d: %g vs %g", i, v, y[i])
			}
		}
	}
	for i := range wx {
		compare(wx[i], wy[i])
	}
	sx, err := loadDistributedResumeState(x)
	if err != nil {
		t.Fatal(err)
	}
	sy, err := loadDistributedResumeState(y)
	if err != nil {
		t.Fatal(err)
	}
	if len(sx.Trainer.Tensors) != len(sy.Trainer.Tensors) {
		t.Fatal("optimizer state count mismatch")
	}
	for i := range sx.Trainer.Tensors {
		compare(sx.Trainer.Tensors[i].Data, sy.Trainer.Tensors[i].Data)
	}
}
