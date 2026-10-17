//go:build darwin || linux

package train

import (
	"context"
	"testing"
	"time"

	wc "github.com/mrothroc/mixlab/workercontrol"
	"github.com/mrothroc/mixlab/workercontrol/local"
)

func TestManagedLearnerDelayedAssignment(t *testing.T) {
	t.Setenv("MIXLAB_LEARNER_PRIVATE_DIR", "/private/learner")
	binding, initial := managedLearnerAssignment(t)
	child, host := managedConnectionPairWithOptions(t, binding, local.LifetimeOptions(2*time.Minute))
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	result := make(chan error, 1)
	go func() {
		result <- serveManagedWorkerWithLearner(ctx, child, nil, func(_ context.Context, d *local.Dispatcher, e wc.Envelope) error {
			if d.Binding() != binding || string(e.Payload) != string(initial.Payload) {
				t.Error("delayed assignment changed identity or payload")
			}
			return nil
		})
	}()
	// Model admission delayed by resource suspension/host scheduling. Exercise
	// the real worker receive path beyond its old 30s pre-readiness timeout.
	select {
	case err := <-result:
		t.Fatalf("learner exited before assignment: %v", err)
	case <-time.After(32 * time.Second):
	}
	if err := host.Send(ctx, initial); err != nil {
		t.Fatal(err)
	}
	select {
	case err := <-result:
		if err != nil {
			t.Fatal(err)
		}
	case <-ctx.Done():
		t.Fatal(ctx.Err())
	}
}
