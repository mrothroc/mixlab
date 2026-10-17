package train

import (
	"testing"
	"time"
)

func TestManagedBootstrapTimeoutScope(t *testing.T) {
	t.Setenv("MIXLAB_LEARNER_PRIVATE_DIR", "")
	if managedBootstrapTimeout() != 30*time.Second {
		t.Fatal("ordinary worker bootstrap changed")
	}
	t.Setenv("MIXLAB_LEARNER_PRIVATE_DIR", "/private/learner")
	if managedBootstrapTimeout() != 2*time.Minute {
		t.Fatal("learner bootstrap shorter than hosting allowance")
	}
}
