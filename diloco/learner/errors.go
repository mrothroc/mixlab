package learner

import "errors"

// ErrUnavailable marks a bounded transport outage, never an authorization or
// protocol rejection. The agent owns retries; exhaustion lets hosting recover
// the same allocation without exposing remote credential diagnostics to MLX.
var ErrUnavailable = errors.New("learner coordinator unavailable")
