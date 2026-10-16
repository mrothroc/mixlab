//go:build darwin || linux

package workerhost

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/distributed"
	"github.com/mrothroc/mixlab/statehome"
	wc "github.com/mrothroc/mixlab/workercontrol"
	"github.com/mrothroc/mixlab/workercontrol/local"
	"github.com/mrothroc/mixlab/workerhost/contract"
	"github.com/mrothroc/mixlab/workerjob"
)

func TestMain(m *testing.M) {
	if len(os.Args) == 3 && os.Args[1] == "internal-test-agent" {
		if err := helperGuardianAgent(os.Args[2]); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		os.Exit(0)
	}
	if len(os.Args) == 2 && os.Args[1] == guardianCommand {
		if err := ServeGuardian(os.NewFile(3, "guardian-control")); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		os.Exit(0)
	}
	if len(os.Args) > 1 && os.Args[1] == "-mode" {
		if len(os.Args) == 3 && os.Args[2] == "worker-probe" {
			os.Exit(helperProbe())
		}
		os.Exit(helperWorker())
	}
	os.Exit(m.Run())
}

func helperWorker() int {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if len(os.Args) != 7 || os.Args[1] != "-mode" || os.Args[2] != "managed-worker" || os.Args[3] != "-worker-control-socket" || os.Args[4] == "" || os.Args[5] != "-worker-session-fd" || os.Args[6] != "3" {
		return 20
	}
	credential := os.NewFile(3, "session")
	conn, err := local.Connect(ctx, os.Args[4], credential, local.Limits{FrameBytes: wc.MaxFrameBytes, Bytes: workerjob.ControlByteBudget, Messages: workerjob.ControlMessageBudget})
	if err != nil {
		return 21
	}
	defer func() { _ = conn.Close() }()
	if _, err := credential.Stat(); !errors.Is(err, os.ErrClosed) {
		return 38
	}
	e, err := conn.Receive(ctx)
	if err != nil {
		return 22
	}
	if err := checkChildLaunchBoundary(e); err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 39
	}
	if e.PayloadKind == workerjob.LearnerVersion {
		return helperLearner(ctx, conn, e)
	}
	a, err := workerjob.Decode(e.Payload, conn.Binding())
	if err != nil {
		return 23
	}
	// The launcher must not copy credentials/debugger/library injection from
	// its ambient environment. The capability itself is descriptor-only.
	if os.Getenv("WORKERHOST_SECRET") != "" || os.Getenv("DYLD_INSERT_LIBRARIES") != "" {
		return 24
	}
	var seq uint64
	send := func(kind wc.Kind, event workerjob.Event) bool {
		seq++
		e, err := workerjob.Envelope(conn.Binding(), seq, kind, event)
		return err == nil && conn.Send(ctx, e) == nil
	}
	switch a.DatasetSelector {
	case "exit":
		return 25
	case "stall":
		time.Sleep(10 * time.Second)
		return 26
	case "logs":
		for i := 0; i < 1000; i++ {
			fmt.Println(strings.Repeat("x", 100))
		}
		time.Sleep(10 * time.Second)
		return 27
	}
	if !send(wc.KindReadiness, workerjob.Event{}) {
		return 28
	}
	if a.DatasetSelector == "duplicate" {
		send(wc.KindReadiness, workerjob.Event{})
		time.Sleep(time.Second)
		return 29
	}
	if !send(wc.KindProgress, workerjob.Event{Step: 1, Committed: 1, Loss: 2}) {
		return 31
	}
	if a.DatasetSelector == "runtime-stall" {
		if os.WriteFile("worker-ready", []byte("ready"), 0600) != nil {
			return 37
		}
		time.Sleep(10 * time.Second)
		return 30
	}
	switch a.DatasetSelector {
	case "resource-cpu":
		var n uint64
		until := time.Now().Add(10 * time.Second)
		for time.Now().Before(until) {
			for i := 0; i < 10000; i++ {
				n = n*1664525 + 1013904223
			}
		}
		runtime.KeepAlive(n)
	case "resource-memory":
		b := make([]byte, 64<<20)
		for i := range b {
			b[i] = 1
		}
		time.Sleep(10 * time.Second)
		runtime.KeepAlive(b)
	case "resource-disk", "resource-final-disk":
		if os.WriteFile("large-output.bin", make([]byte, 2<<20), 0600) != nil {
			return 36
		}
		if a.DatasetSelector == "resource-disk" {
			time.Sleep(10 * time.Second)
		}
	}
	if a.DatasetSelector == "descendant" {
		child := exec.Command("/bin/sleep", "30")
		if child.Start() != nil {
			return 34
		}
		if os.WriteFile("descendant.pid", []byte(strconv.Itoa(child.Process.Pid)), 0600) != nil {
			return 35
		}
		_ = child.Process.Release()
	}
	event := workerjob.Event{}
	if a.DatasetSelector == "error" {
		event.Error = "intentional training error"
	}
	if !send(wc.KindTerminalOutcome, event) {
		return 32
	}
	if a.DatasetSelector == "terminal-hang" {
		time.Sleep(10 * time.Second)
	}
	if a.DatasetSelector == "exit-after-success" {
		return 33
	}
	return 0
}

const launchCredentialCanary = "test-only-parent-credential-canary"

func checkChildLaunchBoundary(initial wc.Envelope) error {
	cwd, err := os.Getwd()
	if err != nil {
		return err
	}
	expected := map[string]string{"PATH": "/usr/bin:/bin", "HOME": cwd, "TMPDIR": cwd}
	if initial.PayloadKind == workerjob.LearnerVersion {
		private := os.Getenv("MIXLAB_LEARNER_PRIVATE_DIR")
		if private == "" || !filepath.IsAbs(private) || pathsOverlap(private, cwd) {
			return fmt.Errorf("invalid private directory environment")
		}
		expected["MIXLAB_LEARNER_PRIVATE_DIR"] = private
	} else {
		var assignment workerjob.Assignment
		if err := json.Unmarshal(initial.Payload, &assignment); err != nil {
			return err
		}
		expected["MLX_HOSTFILE"] = filepath.Join(cwd, "ring.json")
		expected["MLX_RANK"] = strconv.Itoa(assignment.View.LocalRank)
	}
	for _, entry := range os.Environ() {
		key, value, ok := strings.Cut(entry, "=")
		want, allowed := expected[key]
		if !ok || !allowed || value != want {
			return fmt.Errorf("unexpected child environment key %q", key)
		}
		delete(expected, key)
	}
	if len(expected) != 0 {
		return fmt.Errorf("missing child environment keys")
	}
	if bytes.Contains(initial.Payload, []byte(launchCredentialCanary)) || strings.Contains(strings.Join(os.Args, "\x00"), launchCredentialCanary) {
		return fmt.Errorf("parent credential leaked into assignment or argv")
	}
	return nil
}

func TestSupervisedWorkerCredentialIsolation(t *testing.T) {
	for _, key := range []string{"WORKERHOST_SECRET", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "MIXLAB_WORKLOAD_TOKEN", "SSL_CERT_FILE", "SSL_KEY_FILE", "DYLD_INSERT_LIBRARIES", "LD_PRELOAD", "MIXLAB_LEARNER_PRIVATE_DIR"} {
		t.Setenv(key, launchCredentialCanary)
	}
	for _, mode := range []string{"ddp", "learner"} {
		t.Run(mode, func(t *testing.T) {
			s, p := plan(t, "success")
			if mode == "learner" {
				s, p = learnerPlan(t, "success")
			}
			result, err := s.Run(context.Background(), p)
			if err != nil || !result.Ready {
				t.Fatalf("credential isolation launch: %+v %v", result, err)
			}
			// The fake credentials are parent-only. No logs, assignment sidecars,
			// or retained child output may contain them after either launch path.
			roots := []string{p.Directory.Dir()}
			if p.Learner != nil {
				roots = append(roots, p.LearnerPrivateDirectory.Dir())
			}
			for _, root := range roots {
				err := filepath.WalkDir(root, func(path string, entry os.DirEntry, err error) error {
					if err != nil {
						return err
					}
					if !entry.Type().IsRegular() {
						return nil
					}
					body, err := os.ReadFile(path)
					if err != nil {
						return err
					}
					if bytes.Contains(body, []byte(launchCredentialCanary)) {
						return fmt.Errorf("parent credential leaked into %s", entry.Name())
					}
					return nil
				})
				if err != nil {
					t.Fatal(err)
				}
			}
		})
	}
}

func privateDirectory(t *testing.T) statehome.Path {
	t.Helper()
	base := os.TempDir()
	if runtime.GOOS == "darwin" {
		base = "/private/tmp"
	}
	dir, err := os.MkdirTemp(base, "wh-")
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

func plan(t *testing.T, selector string) (*Supervisor, LaunchPlan) {
	t.Helper()
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	id, err := workerjob.FileDigest(exe)
	if err != nil {
		t.Fatal(err)
	}
	s, err := New(exe, id)
	if err != nil {
		t.Fatal(err)
	}
	m, err := distributed.NewDDPGroupMembership("run", "group", 0, "ring", []distributed.DDPGroupMember{{MemberID: "a", Rank: 0}, {MemberID: "b", Rank: 1}})
	if err != nil {
		t.Fatal(err)
	}
	v, err := distributed.NewLocalGroupView(m, "a", 0, "attempt")
	if err != nil {
		t.Fatal(err)
	}
	p := LaunchPlan{Directory: privateDirectory(t), StartupTimeout: time.Second, ShutdownGrace: 1500 * time.Millisecond,
		Limits: contract.Limits{CPUSeconds: 60, MemoryBytes: 1 << 30, DiskBytes: 1 << 30, LogBytes: 4096},
		Assignment: workerjob.Assignment{Version: workerjob.Version, JobID: "job", AttemptID: "attempt", BuildID: id, View: v,
			Config: json.RawMessage(`{"model_dim":16}`), DatasetSelector: selector, TrainPattern: "/unused/train.bin",
			DatasetSHA256: strings.Repeat("a", 64), ProgramSHA256: strings.Repeat("b", 64), WeightLayoutSHA256: strings.Repeat("c", 64), OptimizerSHA256: strings.Repeat("d", 64), RuntimeSeconds: 5,
			RingAddresses: [][]string{{"127.0.0.1:30000"}, {"127.0.0.1:30001"}}}}
	return s, p
}

func TestSupervisedWorkerLifecycle(t *testing.T) {
	t.Setenv("WORKERHOST_SECRET", "must-not-reach-child")
	for _, mode := range []string{"success", "descendant", "error", "exit", "stall", "logs", "duplicate", "runtime-stall", "terminal-hang", "exit-after-success"} {
		t.Run(mode, func(t *testing.T) {
			s, p := plan(t, mode)
			if mode == "runtime-stall" {
				p.Assignment.RuntimeSeconds = 1
			}
			ctx, cancel := context.WithTimeout(context.Background(), 8*time.Second)
			defer cancel()
			before := time.Now()
			r, err := s.Run(ctx, p)
			if (err == nil) != (mode == "success" || mode == "descendant") {
				t.Fatalf("mode=%s result=%+v err=%v", mode, r, err)
			}
			if mode == "success" && (!r.Ready || r.Progress.Step != 1) {
				t.Fatalf("lost events %+v", r)
			}
			if r.PID == 0 {
				t.Fatalf("child not launched: %v", err)
			}
			if err := syscall.Kill(r.PID, 0); err != syscall.ESRCH {
				t.Fatalf("child not reaped: %v", err)
			}
			if mode == "descendant" {
				pid, err := os.ReadFile(filepath.Join(p.Directory.Dir(), "descendant.pid"))
				if err != nil {
					t.Fatal(err)
				}
				// An orphan may briefly be a zombie until init reaps it. It must
				// never remain a live sleeper after process-group cleanup.
				deadline := time.Now().Add(time.Second)
				for {
					state, err := exec.Command("/bin/ps", "-o", "stat=", "-p", string(pid)).Output()
					if err != nil || strings.TrimSpace(string(state)) == "" || strings.HasPrefix(strings.TrimSpace(string(state)), "Z") {
						break
					}
					if time.Now().After(deadline) {
						t.Fatalf("descendant survived: %s", state)
					}
					time.Sleep(10 * time.Millisecond)
				}
			}
			if _, err := os.Lstat(filepath.Join(p.Directory.Dir(), "control.sock")); !os.IsNotExist(err) {
				t.Fatalf("socket not cleaned: %v", err)
			}
			info, err := os.Stat(filepath.Join(p.Directory.Dir(), "worker.log"))
			if err != nil || info.Size() > p.Limits.LogBytes {
				t.Fatalf("unbounded log: %v", err)
			}
			if time.Since(before) > 6*time.Second {
				t.Fatal("supervision not bounded")
			}
			if _, err := s.Run(ctx, p); err == nil {
				t.Fatal("reused attempt directory")
			}
		})
	}
}

func TestWorkerExitBeforeAdmission(t *testing.T) {
	_, p := plan(t, "unused")
	id, err := workerjob.FileDigest("/usr/bin/true")
	if err != nil {
		t.Fatal(err)
	}
	s, err := New("/usr/bin/true", id)
	if err != nil {
		t.Fatal(err)
	}
	p.Assignment.BuildID = id
	p.StartupTimeout = 10 * time.Second
	start := time.Now()
	r, err := s.Run(context.Background(), p)
	if err == nil || r.PID == 0 || time.Since(start) > 2*time.Second {
		t.Fatalf("dead child admission: %+v %v", r, err)
	}
}

func TestSupervisedWorkerCancellationAndBuildMismatch(t *testing.T) {
	s, p := plan(t, "stall")
	ctx, cancel := context.WithCancel(context.Background())
	timer := time.AfterFunc(300*time.Millisecond, cancel)
	defer timer.Stop()
	r, err := s.Run(ctx, p)
	if err == nil || r.PID == 0 || syscall.Kill(r.PID, 0) != syscall.ESRCH {
		t.Fatalf("cancel: %+v %v", r, err)
	}
	s, p = plan(t, "success")
	p.Assignment.BuildID = strings.Repeat("f", 64)
	r, err = s.Run(context.Background(), p)
	if err == nil || r.PID != 0 {
		t.Fatalf("bad build launched: %+v %v", r, err)
	}
}
