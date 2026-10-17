package train

import (
	"os"
	"time"
)

func managedBootstrapTimeout() time.Duration {
	// Hosting supplies this directory only for a learner. Match its bounded
	// two-minute startup allowance rather than aborting earlier on a loaded host.
	if os.Getenv("MIXLAB_LEARNER_PRIVATE_DIR") != "" {
		return 2 * time.Minute
	}
	return 30 * time.Second
}
