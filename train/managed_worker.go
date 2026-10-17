package train

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/artifact/checkpoint"
	"github.com/mrothroc/mixlab/data"
	wc "github.com/mrothroc/mixlab/workercontrol"
	"github.com/mrothroc/mixlab/workercontrol/local"
	"github.com/mrothroc/mixlab/workerjob"
)

type managedTraining struct {
	ctx        context.Context
	assignment workerjob.Assignment
	ready      func() error
	progress   func(workerjob.Event) error
}

// RunManagedWorker accepts only an inherited capability descriptor and private
// socket. Authentication and assignment checks precede config/data/GPU use.
// The hosting process owns hard deadlines and terminates stuck native calls.
func RunManagedWorker(ctx context.Context, socket string, fd int) error {
	return RunManagedWorkerWithLearner(ctx, socket, fd, nil)
}

// RunManagedWorkerWithLearner lets the executable compose a learner without
// linking it into train. The callback receives the authenticated assignment and
// sole dispatcher after strict envelope/schema/binding validation, owns learner
// lifecycle events, and must honor ctx. A nil
// callback permits only DDP. All connection resources close when it returns.
func RunManagedWorkerWithLearner(ctx context.Context, socket string, fd int, learner func(context.Context, *local.Dispatcher, wc.Envelope) error) error {
	if socket == "" || fd < 3 {
		return fmt.Errorf("managed-worker requires a control socket and inherited session FD >= 3")
	}
	credential := os.NewFile(uintptr(fd), "worker-session")
	if credential == nil {
		return fmt.Errorf("invalid worker session descriptor")
	}
	authCtx, cancel := context.WithTimeout(ctx, managedBootstrapTimeout())
	limits := local.Limits{FrameBytes: wc.MaxFrameBytes, Bytes: workerjob.ControlByteBudget, Messages: workerjob.ControlMessageBudget}
	if learner != nil {
		limits.Bytes = 1 << 40
		limits.Messages = 16 << 20
	}
	conn, err := local.Connect(authCtx, socket, credential, limits)
	cancel()
	if err != nil {
		return err
	}
	defer func() { _ = conn.Close() }()
	return serveManagedWorkerWithLearner(ctx, conn, runManagedAssignment, learner)
}

type managedRun func(context.Context, workerjob.Assignment, func() error, func(workerjob.Event) error) error

func serveManagedWorker(parent context.Context, conn *local.Conn, run managedRun) error {
	return serveManagedWorkerWithLearner(parent, conn, run, nil)
}

func serveManagedWorkerWithLearner(parent context.Context, conn *local.Conn, run managedRun, learner func(context.Context, *local.Dispatcher, wc.Envelope) error) error {
	// Read the authenticated assignment before choosing session lifetimes.
	// Only learner sessions may intentionally suspend while awaiting resources.
	readCtx, stopRead := context.WithTimeout(parent, managedBootstrapTimeout())
	e, err := conn.Receive(readCtx)
	stopRead()
	if err != nil {
		return err
	}
	options := local.DispatcherOptions{ReadTimeout: 7 * 24 * time.Hour, RequestTimeout: 2 * time.Minute}
	if e.PayloadKind == workerjob.LearnerVersion {
		if e.Kind != wc.KindAssignment || e.PayloadVersion != 1 || e.CorrelationID != conn.Binding().AttemptID {
			return fmt.Errorf("invalid managed learner assignment envelope")
		}
		a, err := workerjob.DecodeLearner(e.Payload, conn.Binding())
		if err != nil {
			return fmt.Errorf("invalid managed learner assignment: %w", err)
		}
		if learner == nil {
			return fmt.Errorf("managed learner schema is not composed")
		}
		options = local.LifetimeOptions(time.Duration(a.RuntimeSeconds) * time.Second)
	}
	session, err := local.NewDispatcher(parent, conn, options)
	if err != nil {
		return err
	}
	defer func() { _ = session.Close() }()
	if e.PayloadKind == workerjob.LearnerVersion {
		return learner(parent, session, e)
	}
	return serveManagedDDP(parent, session, e, run)
}

func serveManagedDDP(parent context.Context, session *local.Dispatcher, e wc.Envelope, run managedRun) error {
	defer func() { _ = session.Close() }()
	var err error
	if err = workerjob.CheckPayload(e, wc.KindAssignment); err != nil {
		return err
	}
	a, err := workerjob.Decode(e.Payload, session.Binding())
	if err != nil {
		return err
	}
	exe, err := os.Executable()
	if err != nil {
		return err
	}
	id, err := workerjob.FileDigest(exe)
	if err != nil || id != a.BuildID {
		return fmt.Errorf("managed worker executable does not match approved build")
	}
	ctx, cancel := context.WithTimeout(parent, time.Duration(a.RuntimeSeconds)*time.Second)
	defer cancel()
	send := func(kind wc.Kind, event any) error {
		msg, err := workerjob.Envelope(session.Binding(), 1, kind, event)
		if err != nil {
			return err
		}
		outCtx, done := context.WithTimeout(ctx, 5*time.Second)
		defer done()
		return session.Send(outCtx, msg)
	}
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		// Only cancellation is permitted after the immutable assignment. Any
		// disconnect or protocol violation revokes this running attempt as well.
		select {
		case msg := <-session.Controls():
			var event workerjob.Event
			if workerjob.CheckPayload(msg, wc.KindCancellation) == nil {
				_ = workerjob.DecodePayload(msg.Payload, &event)
			}
		case <-session.Events():
		case <-session.Done():
		case <-ctx.Done():
		}
		cancel()
	}()
	go func() {
		defer wg.Done()
		ticker := time.NewTicker(5 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				if send(wc.KindHeartbeat, workerjob.Event{}) != nil {
					cancel()
					return
				}
			}
		}
	}()
	defer func() {
		cancel()
		wg.Wait()
	}()
	err = run(ctx, a, func() error { return send(wc.KindReadiness, workerjob.Event{}) }, func(event workerjob.Event) error {
		return send(wc.KindProgress, event)
	})
	if err == nil {
		err = ctx.Err()
	}
	if err == nil && a.OutputMaxBytes > 0 && a.View.LocalRank == 0 {
		name := workerjob.FinalWeightsFile
		if a.CheckpointAt > 0 {
			name = checkpoint.File
		}
		err = sendManagedArtifactFile(ctx, name, a.OutputMaxBytes, func(message workerjob.ArtifactMessage) error { return send(wc.KindArtifactWrite, message) })
	}
	terminal := workerjob.Event{}
	if err != nil {
		terminal.Error = err.Error()
		if len(terminal.Error) > 4096 {
			terminal.Error = "managed training failed (see bounded worker log)"
		}
	}
	if sendErr := send(wc.KindTerminalOutcome, terminal); err == nil {
		err = sendErr
	}
	return err
}

func prepareManagedAssignment(a workerjob.Assignment) (*ArchConfig, error) {
	if err := a.Validate(); err != nil {
		return nil, err
	}
	cfg, err := arch.ParseArchConfig(a.Config, "managed assignment")
	if err != nil {
		return nil, err
	}
	if err = arch.ValidateDistributedConfig(cfg); err != nil {
		return nil, err
	}
	if cfg.Training.Distributed == nil || cfg.Training.Distributed.Backend != "ring" || cfg.CharVocabSize != 0 {
		return nil, fmt.Errorf("managed foundation requires explicit ring DDP and no external char artifacts")
	}
	// Dataset content is read only after local authentication. The loader checks
	// the identity again after actual MLX rank/world admission, before batches.
	dataset, err := data.DistributedDatasetIdentity(a.TrainPattern)
	if err != nil {
		return nil, err
	}
	if dataset != a.DatasetSHA256 {
		return nil, fmt.Errorf("managed assignment dataset digest mismatch")
	}
	if err := configureDatasetForTraining(cfg, a.TrainPattern, cfg.Name); err != nil {
		return nil, err
	}
	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		return nil, err
	}
	digest, err := workerjob.Digest(prog)
	if err != nil || digest != a.ProgramSHA256 {
		return nil, fmt.Errorf("managed assignment program digest mismatch")
	}
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		return nil, err
	}
	spec, err := buildTrainerOptimizerSpec(cfg, shapes)
	if err != nil {
		return nil, err
	}
	spec.ComputeDType, err = gpuComputeDTypeForTraining(cfg)
	if err != nil {
		return nil, err
	}
	if err := checkManagedLayout(a, shapes, spec); err != nil {
		return nil, err
	}
	// The launcher writes this bounded, exact ring file. Reject inherited or
	// substituted launch environment rather than interpreting it as authority.
	ring, err := os.ReadFile(os.Getenv("MLX_HOSTFILE"))
	want, marshalErr := json.Marshal(a.RingAddresses)
	if err != nil || marshalErr != nil || string(ring) != string(want) || os.Getenv("MLX_RANK") != fmt.Sprint(a.View.LocalRank) {
		return nil, fmt.Errorf("managed ring launch inputs differ from assignment")
	}
	return cfg, nil
}

func runManagedAssignment(ctx context.Context, a workerjob.Assignment, ready func() error, progress func(workerjob.Event) error) error {
	cfg, err := prepareManagedAssignment(a)
	if err != nil {
		return err
	}
	if err = ctx.Err(); err != nil {
		return err
	}
	opts := TrainOptions{LogEvery: 100, ValEvery: 100, managed: &managedTraining{ctx: ctx, assignment: a, ready: ready, progress: progress}}
	if a.Resume != nil {
		opts.Resume, err = prepareManagedResume(ctx, a)
		if err != nil {
			return err
		}
	}
	if a.CheckpointAt > 0 {
		opts.CheckpointDir = "managed-checkpoint"
		opts.CheckpointEvery = int(a.CheckpointAt)
	} else if a.OutputMaxBytes > 0 {
		opts.SafetensorsPath = workerjob.FinalWeightsFile
	}
	_, err = runTrain(cfg, a.TrainPattern, opts)
	if err == nil && a.CheckpointAt > 0 && a.View.LocalRank == 0 {
		err = writeManagedCheckpointBundle(ctx, opts.CheckpointDir, checkpoint.File, a.OutputMaxBytes)
	}
	return err
}
