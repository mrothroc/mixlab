//go:build darwin || linux

package clusterapp

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/coordinator/learnerapi"
	"github.com/mrothroc/mixlab/diloco/authenticated"
	private "github.com/mrothroc/mixlab/diloco/learner"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/transport/managedtls"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/workerhost"
	"github.com/mrothroc/mixlab/workerhost/contract"
)

func committedDrainFixture(t *testing.T) (learnerRuntimeFixture, trust.AuthenticatedPrincipal) {
	t.Helper()
	f := newLearnerRuntimeFixture(t)
	return f, commitDrainFixture(t, f)
}

func commitDrainFixture(t *testing.T, f learnerRuntimeFixture) trust.AuthenticatedPrincipal {
	t.Helper()
	ctx := context.Background()
	x, err := f.remote.runtime.Store.ActiveExecution(ctx)
	check(t, err)
	m := x.Learner.Signed.Manifest
	v, err := f.authority.Current(ctx, testNow)
	check(t, err)
	identity, signer, err := f.coordinator.Active(testNow)
	check(t, err)
	actor, err := trust.AuthenticatePrincipal(f.authority.Anchor, v, identity.Chain, testNow)
	check(t, err)
	var reg run.Registration
	b, err := os.ReadFile("../../diloco/testdata/registration.json")
	check(t, err)
	check(t, json.Unmarshal(b, &reg))
	reg.RunID, reg.PlanHash = m.Plan.Cohort.RunID, m.Plan.SHA256
	reg.Generation, reg.CohortHash = m.Plan.Cohort.Generation, m.Plan.Cohort.CohortHash
	reg.Slot, reg.Contract, reg.Coordinator = m.Slot, m.Plan.Contract, m.Plan.Coordinator
	reg.BaseModelHash, reg.AttemptID, reg.JobManifestHash = m.Plan.BaseModel.Ref.SHA256, m.Attempt, x.Approval.ManifestHash
	c := run.Commit{Format: run.CommitVersion, Binding: run.Binding{RunID: reg.RunID, PlanHash: reg.PlanHash, Generation: reg.Generation, CohortHash: reg.CohortHash}, Coordinator: reg.Coordinator, Registrations: []run.Registration{reg}, CommittedAt: testNow.Unix()}
	c.Binding.CommitHash, err = c.Digest()
	check(t, err)
	proof, err := trust.SignPrincipalProof(f.authority.Anchor, identity.Chain, signer, v, trust.SignRequest{Version: trust.ProofVersion, Purpose: trust.RunCommit, Digest: c.Binding.CommitHash, Context: reg.RunID, Audience: reg.Coordinator.Audience}, testNow)
	check(t, err)
	check(t, f.remote.runtime.Store.FinalizeLearner(ctx, actor, x.Job.ID, f.authority.Anchor, v, authenticated.Commit{Body: c, Proof: proof}, testNow))
	return actor
}

func drainCommand(t *testing.T, f learnerRuntimeFixture) nodeagent.LeaseCommand {
	t.Helper()
	x, err := f.remote.runtime.Store.ActiveExecution(context.Background())
	check(t, err)
	key, err := controllerRandomID()
	check(t, err)
	return nodeagent.LeaseCommand{IdempotencyKey: key, Lease: x.Lease.ID, ExpectedVersion: x.Lease.Version, CompletedAttempt: x.Job.Attempt}
}

func TestNodeCompletedAttemptDrainAuthority(t *testing.T) {
	if nodeagent.LearnerDrainTimeout <= learnerRetryBudget {
		t.Fatal("drain cannot accommodate an in-flight semantic retry")
	}
	f, actor := committedDrainFixture(t)
	s := f.remote.runtime.Store
	ctx := context.Background()
	q := drainCommand(t, f)
	wrong := q
	wrong.CompletedAttempt = strings.Repeat("f", 32)
	if _, err := s.ReleaseLease(ctx, actor, wrong, testNow); err == nil {
		t.Fatal("foreign attempt completed")
	}
	if _, err := s.RenewLease(ctx, actor, q, testNow); err == nil {
		t.Fatal("completion accepted as renewal")
	}
	l, err := s.ReleaseLease(ctx, actor, q, testNow)
	check(t, err)
	if l.CompletedAttempt != q.CompletedAttempt || l.DrainUntil != testNow.Add(nodeagent.LearnerDrainTimeout).Unix() {
		t.Fatal("missing bounded drain", l)
	}
	retry, err := s.ReleaseLease(ctx, actor, q, testNow.Add(10*time.Second))
	check(t, err)
	if retry != l {
		t.Fatal("retry extended drain")
	}
	x, err := f.remote.execution(ctx, true)
	check(t, err)
	if !x.LearnerDraining(testNow) {
		t.Fatal("terminal poll not allowed to drain")
	}
	for name, call := range learnerRemoteCommands(f.remote) {
		if err := call(ctx); err == nil {
			t.Fatal("drain allowed new operation", name)
		}
	}
	stale := *f.remote
	stale.binding.AttemptID = strings.Repeat("e", 32)
	if _, err := stale.execution(ctx, true); err == nil {
		t.Fatal("stale attempt retained poll")
	}
	f.remote.runtime.Clock = func() time.Time { return testNow.Add(nodeagent.LearnerDrainTimeout) }
	if _, err := f.remote.execution(ctx, true); err == nil {
		t.Fatal("expired drain retained poll")
	}
	f.remote.runtime.Clock = func() time.Time { return testNow }
	check(t, s.InterruptActive(ctx, "authority_unavailable"))
	if _, err := f.remote.execution(ctx, true); err == nil {
		t.Fatal("revocation fence retained drain")
	}
}

func TestNodeCompletionCannotOverrideFences(t *testing.T) {
	for _, mode := range []string{"cancel-before", "cancel-after", "restart", "expired"} {
		t.Run(mode, func(t *testing.T) {
			f, actor := committedDrainFixture(t)
			s, ctx := f.remote.runtime.Store, context.Background()
			if mode == "cancel-before" {
				check(t, s.InterruptActive(ctx, "agent_shutdown"))
			}
			q := drainCommand(t, f)
			now := testNow
			if mode == "expired" {
				x, err := s.ActiveExecution(ctx)
				check(t, err)
				now = time.Unix(x.Lease.Expires, 0)
			}
			_, err := s.ReleaseLease(ctx, actor, q, now)
			check(t, err)
			switch mode {
			case "cancel-after":
				cancel := drainCommand(t, f)
				cancel.CompletedAttempt = ""
				_, err = s.ReleaseLease(ctx, actor, cancel, testNow)
				check(t, err)
			case "restart":
				check(t, s.InterruptActive(ctx, "agent_restart"))
			}
			x, err := s.ActiveExecution(ctx)
			check(t, err)
			if x.LearnerDraining(testNow) {
				t.Fatal("terminal drain revived a fenced execution", mode)
			}
			if _, err := f.remote.execution(ctx, true); err == nil {
				t.Fatal("fenced terminal poll accepted", mode)
			}
		})
	}
}

// This child exits successfully only upon receiving an actual terminal reply.
// The hosting adapter deliberately pauses delivery so the real node execution
// loop runs its release path while the poll is pending. No sleeps select the race.
func TestNodeDrainChild(t *testing.T) {
	if os.Getenv("MIXLAB_TEST_DRAIN_CHILD") != "1" {
		return
	}
	var reply private.RelayResponse
	if json.NewDecoder(os.Stdin).Decode(&reply) != nil || !reply.Done {
		os.Exit(1)
	}
	os.Exit(0)
}

type drainRunner struct {
	workerhost.Runner
	remote          *nodeLearnerRemote
	entered         chan struct{}
	exit            chan int
	emptyBeforeDone bool
}

func (r *drainRunner) Run(ctx context.Context, a contract.Approved, started func(int) error) error {
	exe, err := os.Executable()
	if err != nil {
		return err
	}
	cmd := exec.CommandContext(ctx, exe, "-test.run=^TestNodeDrainChild$")
	cmd.Env = append(os.Environ(), "MIXLAB_TEST_DRAIN_CHILD=1")
	in, err := cmd.StdinPipe()
	if err != nil {
		return err
	}
	if err = cmd.Start(); err != nil {
		return err
	}
	if err = started(cmd.Process.Pid); err != nil {
		_ = in.Close()
		_ = cmd.Wait()
		return err
	}
	close(r.entered)
	reply, pollErr := r.remote.Poll(ctx)
	if pollErr == nil && r.emptyBeforeDone {
		if reply != (private.RelayResponse{Format: "mixlab_learner_reply_v1"}) {
			pollErr = fmt.Errorf("draining poll exposed stale work: %+v", reply)
		} else {
			reply, pollErr = r.remote.Poll(ctx)
		}
	}
	if pollErr == nil {
		pollErr = json.NewEncoder(in).Encode(reply)
	}
	_ = in.Close()
	err = cmd.Wait()
	r.exit <- cmd.ProcessState.ExitCode()
	if pollErr != nil {
		return pollErr
	}
	return err
}

func TestNodeLearnerCompleteReleasePendingPoll(t *testing.T) {
	for _, mode := range []string{"complete", "stale-assignment", "cancel", "timeout", "authority-unavailable"} {
		t.Run(mode, func(t *testing.T) {
			server := httptest.NewUnstartedServer(nil)
			f := newLearnerRuntimeFixtureEndpoint(t, true, "https://"+server.Listener.Addr().String())
			actor := commitDrainFixture(t, f)
			x, err := f.remote.execution(context.Background(), false)
			check(t, err)
			plan, commit := x.Learner.Signed.Manifest.Plan, *x.Learner.Commit
			reg := commit.Registrations[0]
			outerState := plan.BaseModel
			outerState.Kind = "outer_state_f32"
			assignment := run.Assignment{Format: run.AssignmentVersion, ID: "stale-round", Binding: commit.Binding, Coordinator: plan.Coordinator, WorkerID: reg.Slot.Member.WorkerID, BaseModel: plan.BaseModel, OuterState: outerState, Contract: plan.Contract, PartitionID: reg.Slot.Member.DatasetPartitionID, InnerSteps: plan.Outer.InnerSteps, PrecisionHash: reg.PrecisionHash, IssuedAt: testNow.Unix(), ExpiresAt: testNow.Add(time.Minute).Unix()}
			check(t, assignment.ValidateFor(plan, commit))
			pollEntered, deliver := make(chan struct{}), make(chan struct{})
			var polls, workCalls atomic.Int32
			identity, signer, err := f.coordinator.Active(testNow)
			check(t, err)
			view, err := f.authority.Current(context.Background(), testNow)
			check(t, err)
			policy, err := managedtls.New(managedtls.Identity{Chain: identity.Chain, Key: signer}, func(chain [][]byte, now time.Time) (trust.AuthenticatedPrincipal, error) {
				return trust.AuthenticatePrincipal(f.authority.Anchor, view, chain, now)
			}, func() time.Time { return testNow })
			check(t, err)
			server.Config.Handler = policy.AuthenticateHTTP(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var request learnerapi.Request
				if json.NewDecoder(r.Body).Decode(&request) != nil || request.Call.Operation != "poll" {
					workCalls.Add(1)
					http.Error(w, "unexpected work/artifact call", http.StatusBadRequest)
					return
				}
				n := polls.Add(1)
				if n == 1 {
					close(pollEntered)
					select {
					case <-deliver:
					case <-r.Context().Done():
						return
					}
				}
				reply := private.RelayResponse{Format: "mixlab_learner_reply_v1", Done: true}
				if mode == "stale-assignment" && n == 1 {
					reply.Done, reply.Assignment = false, &assignment
				}
				body, _ := json.Marshal(learnerapi.Response{Reply: reply})
				_, _ = w.Write(body)
			}))
			server.TLS = policy.ServerConfig()
			server.StartTLS()
			defer server.Close()
			s := f.remote.runtime.Store
			ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
			defer cancel()
			root, err := filepath.EvalSymlinks(t.TempDir())
			check(t, err)
			p, err := statehome.Resolve(statehome.Options{ExactDir: filepath.Join(root, "host")}, statehome.Context{Kind: statehome.Worker})
			check(t, err)
			check(t, p.Ensure())
			host, err := workerhost.NewAttemptStore(p)
			check(t, err)
			r := &drainRunner{remote: f.remote, entered: make(chan struct{}), exit: make(chan int, 1), emptyBeforeDone: mode == "stale-assignment"}
			now := testNow
			loop := nodeExecutionLoop{store: s, ports: NodeExecutionPorts{Clock: func() time.Time { return now }, Authorize: func(context.Context, nodeagent.LocalExecution) error { return nil }, OpenLearner: func(context.Context, nodeagent.LocalExecution) (*workerhost.AttemptStore, workerhost.Runner, error) {
				return host, r, nil
			}}}
			check(t, loop.poll(ctx))
			select {
			case <-r.entered:
			case <-ctx.Done():
				t.Fatal("child did not start")
			}
			select {
			case <-pollEntered:
			case <-ctx.Done():
				t.Fatal("poll did not reach coordinator")
			}
			q := drainCommand(t, f)
			if mode == "cancel" {
				q.CompletedAttempt = ""
			}
			_, err = s.ReleaseLease(ctx, actor, q, now)
			check(t, err)
			if mode == "timeout" {
				now = now.Add(nodeagent.LearnerDrainTimeout)
			}
			if mode == "authority-unavailable" {
				loop.ports.Authorize = func(context.Context, nodeagent.LocalExecution) error { return fmt.Errorf("revoked") }
			}
			check(t, loop.poll(ctx))
			close(deliver)
			var code int
			select {
			case code = <-r.exit:
			case <-ctx.Done():
				t.Fatal("child did not exit")
			}
			select {
			case err = <-loop.running.done:
				check(t, err)
			case <-ctx.Done():
				t.Fatal("outcome not published")
			}
			loop.running.cancel()
			x, err = s.ActiveExecution(ctx)
			check(t, err)
			if x.Outcome == nil || !x.Outcome.Terminal() || !x.Outcome.NoChild {
				t.Fatal("missing reaped outcome")
			}
			if mode == "stale-assignment" && (polls.Load() != 2 || workCalls.Load() != 0) {
				t.Fatalf("stale reply triggered work or skipped terminal poll: polls=%d work=%d", polls.Load(), workCalls.Load())
			}
			if mode == "complete" || mode == "stale-assignment" {
				if code != 0 || x.Outcome.Kind != contract.Exited || x.Outcome.Error != "" {
					t.Fatalf("successful drain failed: exit=%d outcome=%+v", code, x.Outcome)
				}
			} else if code == 0 || x.Outcome.Kind != contract.Canceled {
				t.Fatalf("%s became successful: exit=%d outcome=%+v", mode, code, x.Outcome)
			}
		})
	}
}
