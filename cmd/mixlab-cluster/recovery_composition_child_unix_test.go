//go:build darwin || linux

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"runtime"
	"strings"
	"testing"
	"time"

	private "github.com/mrothroc/mixlab/diloco/learner"
	runcontract "github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/learneripc"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/precision"
	wc "github.com/mrothroc/mixlab/workercontrol"
	"github.com/mrothroc/mixlab/workercontrol/local"
	"github.com/mrothroc/mixlab/workerhost"
	"github.com/mrothroc/mixlab/workerjob"
	"github.com/mrothroc/mixlab/workerprobe"
)

// Only the numerical executable is a fixture. Agent, guardian, IPC, credentials,
// admission and installed coordinator composition all run their production code.
func TestMain(m *testing.M) {
	var err error
	switch {
	case len(os.Args) == 2 && os.Args[1] == "internal-worker-host":
		err = workerhost.ServeGuardian(os.NewFile(3, "guardian-control"))
	case len(os.Args) > 2 && os.Args[1] == "-mode":
		err = compositionWorkerMode(os.Args[2])
	default:
		os.Exit(m.Run())
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	os.Exit(0)
}

func compositionProbe(build string) workerprobe.Report {
	return workerprobe.Report{Version: workerprobe.Version, BuildID: build, BuildVersion: "composition-fixture", WorkerProtocol: wc.Version, OS: runtime.GOOS, Arch: runtime.GOARCH, Available: true, MLXVersion: "0.32.3", MLXSupported: true, DeviceKind: "metal", DeviceName: "fixture", Backends: []string{"ring"}, DTypes: []string{"fp32"}, CustomOps: []string{}}
}

func compositionPlan(build string, config []byte) workerprobe.Plan {
	h := strings.Repeat("a", 64)
	return workerprobe.Plan{Version: workerprobe.PlanVersion, BuildID: build, ConfigHash: nodejob.Hash(config), ProgramHash: h, WeightLayoutHash: h, OptimizerHash: h, DType: "fp32", Learner: &workerprobe.LearnerPlan{Outer: runcontract.Outer{Kind: "nesterov", LR: 1, Momentum: 0, Aggregation: "uniform", InnerSteps: 2, TotalSteps: 4, Accumulation: 1}, BatchTokens: 8, SequenceLength: 4, Precision: &precision.Policy{ComputeDType: "float32", Backend: "metal"}}}
}

func compositionWorkerMode(mode string) error {
	if mode == "managed-worker" {
		return compositionLearnerChild()
	}
	exe, err := os.Executable()
	if err != nil {
		return err
	}
	build, err := workerjob.FileDigest(exe)
	if err != nil {
		return err
	}
	switch mode {
	case "worker-probe":
		return json.NewEncoder(os.Stdout).Encode(compositionProbe(build))
	case "worker-learner-probe":
		config, err := io.ReadAll(io.LimitReader(os.Stdin, 256<<10))
		if err != nil {
			return err
		}
		return json.NewEncoder(os.Stdout).Encode(compositionPlan(build, config))
	default:
		return fmt.Errorf("unsupported composition fixture mode %q", mode)
	}
}

func compositionLearnerChild() error {
	if len(os.Args) != 7 || os.Args[3] != "-worker-control-socket" || os.Args[5] != "-worker-session-fd" || os.Args[6] != "3" {
		return fmt.Errorf("unexpected learner launch arguments")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	c, err := local.Connect(ctx, os.Args[4], os.NewFile(3, "session"), local.Limits{FrameBytes: wc.MaxFrameBytes, Bytes: 1 << 30, Messages: 100000})
	if err != nil {
		return err
	}
	defer func() { _ = c.Close() }()
	e, err := c.Receive(ctx)
	if err != nil {
		return err
	}
	a, err := workerjob.DecodeLearner(e.Payload, c.Binding())
	if err != nil {
		return err
	}
	d, err := local.NewDispatcher(ctx, c, local.LifetimeOptions(time.Duration(a.RuntimeSeconds)*time.Second))
	if err != nil {
		return err
	}
	defer func() { _ = d.Close() }()
	send := func(kind wc.Kind, event workerjob.Event) error {
		e, err := workerjob.Envelope(d.Binding(), 1, kind, event)
		if err != nil {
			return err
		}
		return d.Send(ctx, e)
	}
	if err := send(wc.KindReadiness, workerjob.Event{}); err != nil {
		return err
	}
	heartbeatDone := make(chan struct{})
	go func() {
		defer close(heartbeatDone)
		timer := time.NewTicker(5 * time.Second)
		defer timer.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-timer.C:
				if send(wc.KindHeartbeat, workerjob.Event{}) != nil {
					cancel()
					return
				}
			}
		}
	}()
	defer func() { cancel(); <-heartbeatDone }()
	peer := &learneripc.Client{Session: d}
	started := false
	admitted := false
	_, loadFixture := os.Stat(a.TrainPattern + ".composition-no-exit")
	poll := time.NewTimer(0)
	defer poll.Stop()
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-d.Controls():
			return send(wc.KindTerminalOutcome, workerjob.Event{})
		case <-d.Done():
			return nil
		case <-poll.C:
		}
		if started {
			if loadFixture == nil {
				if _, err := peer.Poll(ctx); err != nil {
					return err
				}
				if _, err := os.Stat(a.TrainPattern + ".composition-resume"); err == nil {
					if err := os.WriteFile(a.TrainPattern+".composition-after-"+a.Member.WorkerID, []byte(a.AttemptID), 0600); err != nil {
						return err
					}
				}
				poll.Reset(200 * time.Millisecond)
				continue
			}
			// Numerical work is deliberately idle in this composition fixture.
			// The independent heartbeat keeps hosting live without more polls.
			continue
		}
		if !admitted {
			// This synthetic numerical child waits until the test observes Open.
			// Readiness and heartbeats remain live through real admission, without
			// idle learner RPCs competing with admission on the race-instrumented host.
			if _, err := os.Stat(a.TrainPattern + ".composition-open"); os.IsNotExist(err) {
				poll.Reset(200 * time.Millisecond)
				continue
			} else if err != nil {
				return err
			}
			admitted = true
		}
		next, err := peer.Poll(ctx)
		if err != nil {
			return err
		}
		if next.Assignment == nil {
			// Match the managed learner's delay after each completed poll, rather
			// than letting slow calls accumulate a continuously ready ticker.
			poll.Reset(200 * time.Millisecond)
			continue
		}
		privateHash := strings.Repeat("c", 64)
		if a.ExpectedPrivateHash != "" && a.ExpectedPrivateHash != privateHash {
			return fmt.Errorf("recovery changed fixture private hash")
		}
		h, err := runcontract.Digest(*next.Assignment)
		if err != nil {
			return err
		}
		if err := peer.Start(ctx, private.Start{Format: "mixlab_learner_start_v1", Binding: next.Assignment.Binding, WorkerID: a.Member.WorkerID, AssignmentHash: h, PrivateHash: privateHash}); err != nil {
			return err
		}
		if a.ExpectedPrivateHash != "" {
			if _, err := fmt.Fprintf(os.Stdout, "composition-start-ack attempt=%s pid=%d\n", a.AttemptID, os.Getpid()); err != nil {
				return err
			}
		}
		_, keepOriginal := os.Stat(a.TrainPattern + ".composition-no-exit")
		if a.Member.DatasetPartitionID == "partition-0" && a.ExpectedPrivateHash == "" && os.IsNotExist(keepOriginal) {
			_ = send(wc.KindTerminalOutcome, workerjob.Event{Error: "composition fixture stopped learner"})
			return fmt.Errorf("composition fixture stopped learner")
		}
		started = true
		if loadFixture == nil {
			poll.Reset(200 * time.Millisecond)
		}
	}
}
