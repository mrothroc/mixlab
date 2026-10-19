//go:build darwin || linux

package clusterapp

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/precision"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/workload"
	"github.com/mrothroc/mixlab/workerprobe"
)

func preparationApp(t *testing.T, f learnerRuntimeFixture, life context.Context, numerical func(context.Context, []byte) (workerprobe.Plan, error)) *NodeJobs {
	t.Helper()
	r := f.remote.runtime
	app, err := NewNodeJobs(life, NodeJobOptions{Store: r.Store, Node: f.prepare.Signed.Manifest.Slot.Allocation.NodeID, Anchor: r.Anchor, CredentialRoot: f.root, Workload: f.ports, CurrentTrust: NodeCurrentTrust(r.Principal), RelayAddress: "127.0.0.1:7445", Clock: r.Clock, LearnerProbe: numerical, Preparation: nodeagent.PreparationPorts{
		Clock: r.Clock,
		Probe: func(context.Context) (workerprobe.Report, error) { return f.probe, nil },
		DatasetID: func(context.Context, string) (string, error) {
			return f.prepare.Signed.Manifest.Plan.Contract.DatasetHash, nil
		},
		ArtifactPresent: func(context.Context, nodejob.ArtifactRef) error { return nil },
	}})
	check(t, err)
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		check(t, app.StopLearnerPreparation(ctx, f.prepare.Signed.Manifest.Slot.Allocation.JobID))
	})
	return app
}

func preparationNumerical(f learnerRuntimeFixture) workerprobe.Plan {
	p := f.prepare.Signed.Manifest.Plan
	c := p.Contract
	return workerprobe.Plan{Version: workerprobe.PlanVersion, BuildID: c.BuildHash, ConfigHash: c.ConfigHash, ProgramHash: c.ProgramHash, WeightLayoutHash: c.WeightLayoutHash, OptimizerHash: c.InnerOptimizerHash, DType: "fp32", Learner: &workerprobe.LearnerPlan{Outer: p.Outer, BatchTokens: 4, SequenceLength: 4, Precision: &precision.Policy{Backend: "metal", ComputeDType: "float32"}}}
}

func TestLearnerPreparationOutlivesCallerAndRejoins(t *testing.T) {
	f := newLearnerRuntimeFixtureMode(t, false)
	entered, release := make(chan struct{}), make(chan struct{})
	var probes atomic.Int32
	app := preparationApp(t, f, context.Background(), func(ctx context.Context, _ []byte) (workerprobe.Plan, error) {
		probes.Add(1)
		return preparationNumerical(f), ctx.Err()
	})
	app.options.Preparation.Probe = func(ctx context.Context) (workerprobe.Report, error) {
		close(entered)
		select {
		case <-release:
			return f.probe, nil
		case <-ctx.Done():
			return workerprobe.Report{}, ctx.Err()
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	first := make(chan error, 1)
	go func() { _, err := app.PrepareLearner(ctx, f.actor, f.prepare); first <- err }()
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("probe did not start")
	}
	cancel()
	if err := <-first; !errors.Is(err, context.Canceled) {
		t.Fatalf("caller: %v", err)
	}
	// Lease operations must remain available while the numerical process runs.
	lease, err := app.options.Store.LeaseStatus(context.Background(), f.actor, f.prepare.Signed.Manifest.Slot.Allocation.LeaseID, testNow)
	check(t, err)
	_, err = app.options.Store.RenewLease(context.Background(), f.actor, nodeagent.LeaseCommand{Lease: lease.ID, ExpectedVersion: lease.Version, IdempotencyKey: "11111111111111111111111111111111", TTLSeconds: 600}, testNow)
	check(t, err)
	x, err := app.options.Store.ActiveExecution(context.Background())
	check(t, err)
	authorized := make(chan error, 1)
	go func() { authorized <- app.Authorize(context.Background(), *x) }()
	select {
	case err := <-authorized:
		check(t, err)
	case <-time.After(time.Second):
		t.Fatal("preparation held the node-wide authorization mutex")
	}
	joined := make(chan NodeLearnerPrepared, 1)
	secondErr := make(chan error, 1)
	go func() {
		out, err := app.PrepareLearner(context.Background(), f.actor, f.prepare)
		joined <- out
		secondErr <- err
	}()
	close(release)
	var got NodeLearnerPrepared
	select {
	case got = <-joined:
	case <-time.After(5 * time.Second):
		t.Fatal("retry did not finish")
	}
	check(t, <-secondErr)
	if probes.Load() != 1 {
		t.Fatalf("numerical probes=%d", probes.Load())
	}
	check(t, got.Registration.ValidateFor(f.prepare.Signed.Manifest.Plan))
	// Recreating the application owner must reuse durable numerical and dataset
	// progress, not just the in-process completed response.
	restarted := preparationApp(t, f, context.Background(), func(context.Context, []byte) (workerprobe.Plan, error) {
		t.Error("replayed numerical probe")
		return workerprobe.Plan{}, errors.New("replayed")
	})
	restarted.options.Preparation.Probe = func(context.Context) (workerprobe.Report, error) {
		t.Error("replayed device probe")
		return workerprobe.Report{}, errors.New("replayed")
	}
	restarted.options.Preparation.DatasetID = func(context.Context, string) (string, error) {
		t.Error("replayed dataset hash")
		return "", errors.New("replayed")
	}
	again, err := restarted.PrepareLearner(context.Background(), f.actor, f.prepare)
	check(t, err)
	if !reflect.DeepEqual(got, again) {
		t.Fatal("retry changed registration or workload request")
	}
}

func TestLearnerPreparationPartialWorkloadRetry(t *testing.T) {
	f := newLearnerRuntimeFixtureMode(t, false)
	var numerical, signing atomic.Int32
	app := preparationApp(t, f, context.Background(), func(context.Context, []byte) (workerprobe.Plan, error) {
		numerical.Add(1)
		return preparationNumerical(f), nil
	})
	request := app.options.Workload.Request
	app.options.Workload.Request = func(ctx context.Context, scope workload.Scope) (workload.SignedRequest, error) {
		if signing.Add(1) == 1 {
			return workload.SignedRequest{}, context.DeadlineExceeded
		}
		return request(ctx, scope)
	}
	_, err := app.PrepareLearner(context.Background(), f.actor, f.prepare)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("first prepare: %v", err)
	}
	out, err := app.PrepareLearner(context.Background(), f.actor, f.prepare)
	check(t, err)
	check(t, out.Registration.ValidateFor(f.prepare.Signed.Manifest.Plan))
	if numerical.Load() != 1 || signing.Load() != 2 {
		t.Fatalf("numerical=%d signing=%d", numerical.Load(), signing.Load())
	}
}

func TestLearnerPreparationAbortAndServiceStop(t *testing.T) {
	for _, stop := range []string{"abort", "service", "admission_deadline"} {
		t.Run(stop, func(t *testing.T) {
			f := newLearnerRuntimeFixtureMode(t, false)
			life, cancel := context.WithCancel(context.Background())
			defer cancel()
			entered := make(chan struct{})
			app := preparationApp(t, f, life, func(ctx context.Context, _ []byte) (workerprobe.Plan, error) {
				close(entered)
				<-ctx.Done()
				return workerprobe.Plan{}, ctx.Err()
			})
			if stop == "admission_deadline" {
				clock := func() time.Time { return testNow.Add(59 * time.Second) }
				app.options.Clock, app.options.Preparation.Clock, app.options.Workload.Clock = clock, clock, clock
			}
			done := make(chan error, 1)
			go func() { _, err := app.PrepareLearner(context.Background(), f.actor, f.prepare); done <- err }()
			select {
			case <-entered:
			case <-time.After(5 * time.Second):
				t.Fatal("probe did not start")
			}
			switch stop {
			case "service":
				cancel()
			case "abort":
				l, err := app.options.Store.LeaseStatus(context.Background(), f.actor, f.prepare.Signed.Manifest.Slot.Allocation.LeaseID, testNow)
				check(t, err)
				_, err = app.options.Store.ReleaseLease(context.Background(), f.actor, nodeagent.LeaseCommand{Lease: l.ID, ExpectedVersion: l.Version, IdempotencyKey: "22222222222222222222222222222222"}, testNow)
				check(t, err)
			}
			select {
			case err := <-done:
				if err == nil {
					t.Fatal("canceled preparation succeeded")
				}
			case <-time.After(5 * time.Second):
				t.Fatal("preparation did not stop")
			}
			x, err := app.options.Store.ActiveExecution(context.Background())
			check(t, err)
			if x != nil && x.Learner != nil && (x.Learner.Credential != nil || x.Learner.Assignment != nil) {
				t.Fatal("canceled preparation installed credential or started learner")
			}
		})
	}
}

func TestLearnerPreparationResponseClassification(t *testing.T) {
	denial := errors.New("private invalid evidence")
	for _, tc := range []struct {
		name   string
		expiry error
		err    error
		status int
	}{
		{"pending", nil, errLearnerPreparationPending, http.StatusServiceUnavailable},
		{"internal_deadline", nil, context.DeadlineExceeded, http.StatusConflict},
		{"internal_cancel", nil, context.Canceled, http.StatusConflict},
		{"request_deadline", context.DeadlineExceeded, fmt.Errorf("private fetch: %w", context.DeadlineExceeded), http.StatusServiceUnavailable},
		{"request_cancel", context.Canceled, context.Canceled, http.StatusServiceUnavailable},
		{"joined_request_deadline", context.DeadlineExceeded, errors.Join(context.DeadlineExceeded, fmt.Errorf("fetch: %w", context.DeadlineExceeded)), http.StatusServiceUnavailable},
		{"joined_context_causes", context.DeadlineExceeded, errors.Join(context.DeadlineExceeded, context.Canceled), http.StatusServiceUnavailable},
		{"denial", nil, denial, http.StatusConflict},
		{"denial_after_expiry", context.DeadlineExceeded, denial, http.StatusConflict},
		{"denial_with_cancel", nil, errors.Join(context.Canceled, denial), http.StatusConflict},
		{"denial_with_request_cancel", context.Canceled, errors.Join(context.Canceled, denial), http.StatusConflict},
		{"wrapped_denial_with_deadline", context.DeadlineExceeded, fmt.Errorf("prepare: %w", errors.Join(context.DeadlineExceeded, denial)), http.StatusConflict},
		{"output_limit", nil, errors.Join(context.Canceled, io.ErrShortWrite), http.StatusConflict},
		{"output_limit_with_request_cancel", context.Canceled, errors.Join(context.Canceled, io.ErrShortWrite), http.StatusConflict},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			switch tc.expiry {
			case context.DeadlineExceeded:
				var cancel context.CancelFunc
				ctx, cancel = context.WithDeadline(ctx, time.Now().Add(-time.Second))
				defer cancel()
			case context.Canceled:
				var cancel context.CancelFunc
				ctx, cancel = context.WithCancel(ctx)
				cancel()
			}
			w := httptest.NewRecorder()
			nodeResponse(&nodeResponseContext{ResponseWriter: w, ctx: ctx}, nil, tc.err)
			if w.Code != tc.status || strings.Contains(w.Body.String(), "private") {
				t.Fatalf("status=%d want=%d body=%q", w.Code, tc.status, w.Body.String())
			}
		})
	}
}

func TestLearnerPreparationWatcherPreservesCallerDenial(t *testing.T) {
	f := newLearnerRuntimeFixtureMode(t, false)
	var deny atomic.Bool
	denial := errors.New("current caller revoked")
	app := preparationApp(t, f, context.Background(), func(ctx context.Context, _ []byte) (workerprobe.Plan, error) {
		deny.Store(true)
		<-ctx.Done()
		return workerprobe.Plan{}, ctx.Err()
	})
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	ctx = context.WithValue(ctx, nodeCallerCheckKey{}, func() error {
		if deny.Load() {
			return denial
		}
		return nil
	})
	_, err := app.PrepareLearner(ctx, f.actor, f.prepare)
	if !errors.Is(err, denial) || !errors.Is(err, context.Canceled) || ctx.Err() != nil {
		t.Fatalf("watcher lost permanent cause: err=%v request=%v", err, ctx.Err())
	}
	w := httptest.NewRecorder()
	nodeResponse(&nodeResponseContext{ResponseWriter: w, ctx: ctx}, nil, err)
	if w.Code != http.StatusConflict {
		t.Fatalf("watcher denial became retryable: status=%d", w.Code)
	}
}

func TestLearnerPreparationWatcherUsesOperationDeadline(t *testing.T) {
	f := newLearnerRuntimeFixtureMode(t, false)
	life, stop := context.WithCancel(context.Background())
	defer stop()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	numericalContext := make(chan context.Context, 1)
	watcherContext := make(chan context.Context, 1)
	watcherStopped := make(chan struct{})
	release := make(chan struct{})
	var numericalEntered, intercepted atomic.Bool
	app := preparationApp(t, f, life, func(ctx context.Context, _ []byte) (workerprobe.Plan, error) {
		numericalContext <- ctx
		numericalEntered.Store(true)
		select {
		case <-release:
			return preparationNumerical(f), nil
		case <-ctx.Done():
			return workerprobe.Plan{}, ctx.Err()
		}
	})
	current := app.options.CurrentTrust
	app.options.CurrentTrust = func(ctx context.Context, now time.Time) (trust.Anchor, trust.VerifiedSnapshot, error) {
		// Numerical work is blocked, so the first concurrent trust read belongs
		// to the watcher. Later publication checks must remain independent.
		if numericalEntered.Load() && intercepted.CompareAndSwap(false, true) {
			watcherContext <- ctx
			<-ctx.Done()
			close(watcherStopped)
			return trust.Anchor{}, trust.VerifiedSnapshot{}, ctx.Err()
		}
		return current(ctx, now)
	}
	type result struct {
		out NodeLearnerPrepared
		err error
	}
	done := make(chan result, 1)
	go func() {
		out, err := app.PrepareLearner(ctx, f.actor, f.prepare)
		done <- result{out, err}
	}()
	var operation, watcher context.Context
	select {
	case operation = <-numericalContext:
	case <-ctx.Done():
		t.Fatal("numerical preparation did not start", ctx.Err())
	}
	select {
	case watcher = <-watcherContext:
	case <-ctx.Done():
		t.Fatal("watcher did not start its trust check", ctx.Err())
	}
	operationDeadline, operationBounded := operation.Deadline()
	watcherDeadline, watcherBounded := watcher.Deadline()
	if !operationBounded || !watcherBounded || !watcherDeadline.Equal(operationDeadline) {
		t.Fatalf("watcher shortened admission budget: operation=%v bounded=%v watcher=%v bounded=%v", operationDeadline, operationBounded, watcherDeadline, watcherBounded)
	}
	close(release)
	select {
	case got := <-done:
		check(t, got.err)
		check(t, got.out.Registration.ValidateFor(f.prepare.Signed.Manifest.Plan))
	case <-ctx.Done():
		t.Fatal("completed preparation did not cancel and join its watcher", ctx.Err())
	}
	select {
	case <-watcherStopped:
	default:
		t.Fatal("preparation published before its watcher stopped")
	}
}

func TestLearnerPreparationDiagnosticStaysBoundedAndLocal(t *testing.T) {
	var events []string
	h := nodeHTTP{event: func(err error) { events = append(events, err.Error()) }}
	diagnostic := "private preparation failure\nforged record\r\n" + strings.Repeat("x", 8192)
	w := httptest.NewRecorder()
	h.preparationResponse(w, "job", nil, errors.New(diagnostic))
	if w.Code != http.StatusConflict || strings.Contains(w.Body.String(), "private preparation") {
		t.Fatalf("private diagnostic escaped: status=%d body=%q", w.Code, w.Body.String())
	}
	want := fmt.Sprintf("learner preparation job=%q operation=prepare error=%q", "job", diagnostic[:4096])
	if len(events) != 1 || events[0] != want || strings.ContainsAny(events[0], "\r\n") {
		t.Fatalf("diagnostic not bounded and quoted: %q", events)
	}
	w = httptest.NewRecorder()
	h.preparationResponse(w, "job", NodeLearnerAck{"job"}, nil)
	if w.Code != http.StatusOK || len(events) != 1 {
		t.Fatal("successful preparation emitted an error")
	}
}

func TestLearnerPreparationCleanupJoinsCredentialWriter(t *testing.T) {
	for _, stage := range []string{"before_directory", "after_directory", "sign_request"} {
		t.Run(stage, func(t *testing.T) {
			f := newLearnerRuntimeFixtureMode(t, false)
			app := preparationApp(t, f, context.Background(), func(context.Context, []byte) (workerprobe.Plan, error) { return preparationNumerical(f), nil })
			entered := make(chan struct{})
			cancelled := make(chan struct{})
			release := make(chan struct{})
			var releaseOnce sync.Once
			releaseWriter := func() { releaseOnce.Do(func() { close(release) }) }
			defer releaseWriter()
			block := func(ctx context.Context) error {
				close(entered)
				<-ctx.Done()
				close(cancelled)
				<-release
				return ctx.Err()
			}
			if stage == "sign_request" {
				app.options.Workload.Request = func(ctx context.Context, _ workload.Scope) (workload.SignedRequest, error) {
					return workload.SignedRequest{}, block(ctx)
				}
			} else {
				initialize := app.options.Workload.Initialize
				app.options.Workload.Initialize = func(ctx context.Context, scope workload.Scope) error {
					if stage == "after_directory" {
						if err := initialize(ctx, scope); err != nil {
							return err
						}
					}
					return block(ctx)
				}
			}
			done := make(chan error, 1)
			go func() { _, err := app.PrepareLearner(context.Background(), f.actor, f.prepare); done <- err }()
			select {
			case <-entered:
			case <-time.After(5 * time.Second):
				t.Fatal("credential writer did not start")
			}
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			l, err := app.options.Store.LeaseStatus(ctx, f.actor, f.prepare.Signed.Manifest.Slot.Allocation.LeaseID, testNow)
			check(t, err)
			_, err = app.options.Store.ReleaseLease(ctx, f.actor, nodeagent.LeaseCommand{Lease: l.ID, ExpectedVersion: l.Version, IdempotencyKey: "33333333333333333333333333333333"}, testNow)
			check(t, err)
			x, err := app.options.Store.ActiveExecution(ctx)
			check(t, err)
			stopped := make(chan struct{})
			loop := nodeExecutionLoop{store: app.options.Store, ports: NodeExecutionPorts{
				StopPreparation: func(ctx context.Context, job string) error {
					err := app.StopLearnerPreparation(ctx, job)
					close(stopped)
					return err
				},
				Cleanup: app.Cleanup,
			}}
			cleaned := make(chan error, 1)
			go func() { cleaned <- loop.cleanupLearner(ctx, *x) }()
			select {
			case <-cancelled:
			case <-ctx.Done():
				t.Fatal("cleanup did not cancel the credential writer", ctx.Err())
			}
			// Observe the stop boundary too: the credential lock must not hide a
			// missing join by independently blocking the subsequent cleanup.
			select {
			case <-stopped:
				t.Fatal("preparation stop returned while the cancelled writer was still running")
			case err := <-cleaned:
				t.Fatalf("cleanup returned before the credential writer finished: %v", err)
			case <-time.After(100 * time.Millisecond):
			}
			releaseWriter()
			select {
			case err := <-cleaned:
				check(t, err)
			case <-ctx.Done():
				t.Fatal("cleanup did not finish after releasing the writer", ctx.Err())
			}
			select {
			case err := <-done:
				if err == nil {
					t.Fatal("aborted credential preparation succeeded")
				}
			case <-ctx.Done():
				t.Fatal("credential preparation did not finish", ctx.Err())
			}
			x, err = app.options.Store.ActiveExecution(ctx)
			check(t, err)
			if x != nil {
				t.Fatal("cleanup left the reservation active")
			}
		})
	}
}
