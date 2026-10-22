package clusterapp

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/admission"
	"github.com/mrothroc/mixlab/coordinator"
)

type terminalReadRepository struct {
	coordinator.Repository
	state coordinator.State
	err   error
	reads []string
}

func (r *terminalReadRepository) Load(ctx context.Context, id string) (coordinator.State, error) {
	r.reads = append(r.reads, id)
	if err := ctx.Err(); err != nil {
		return coordinator.State{}, err
	}
	return r.state, r.err
}

func TestInstalledDiLoCoServiceUsesDurableTerminalRead(t *testing.T) {
	repo := &terminalReadRepository{}
	models := &DiLoCoModels{Inbox: &admission.Inbox{}, Repository: repo, Clock: func() time.Time { return testNow }}
	service := installedDiLoCoService(models)
	if service.RunTerminal == nil {
		t.Fatal("installed coordinator omitted durable terminal-state wiring")
	}
	if service.Inbox != models.Inbox || !service.Clock().Equal(testNow) || service.Interval != time.Second {
		t.Fatal("installed service changed lifecycle dependencies or normal cadence")
	}
	// Each call must reload its requested run, not capture a phase at startup.
	phases := []coordinator.Phase{coordinator.Ready, coordinator.Open, coordinator.Complete, coordinator.Failed, coordinator.Evaluating}
	for _, phase := range phases {
		repo.state.Phase = phase
		id := "run-" + string(phase)
		terminal, err := service.RunTerminal(context.Background(), id)
		check(t, err)
		if terminal != (phase == coordinator.Complete || phase == coordinator.Failed) || repo.reads[len(repo.reads)-1] != id {
			t.Fatalf("installed callback did not reload %s: terminal=%v reads=%v", id, terminal, repo.reads)
		}
	}
	repo.err = errors.New("durable read unavailable")
	repo.state.Phase = coordinator.Complete
	terminal, err := service.RunTerminal(context.Background(), "unavailable")
	if terminal || !errors.Is(err, repo.err) {
		t.Fatalf("read failure authorized retry delay: terminal=%v err=%v", terminal, err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := service.RunTerminal(ctx, "cancelled"); !errors.Is(err, context.Canceled) {
		t.Fatal("installed callback discarded context", err)
	}
	if len(repo.reads) != len(phases)+2 {
		t.Fatalf("expected one durable read per call: %v", repo.reads)
	}
}
