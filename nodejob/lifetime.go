package nodejob

import (
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/trust"
)

const CleanupAllowance = 5 * time.Minute

// WorkloadDeadline includes the latest permitted start, declared runtime, and
// cleanup. Admission remains short-lived; an admitted execution does not inherit
// the admission expiry. Reject overlong budgets instead of truncating a run.
func (m Manifest) WorkloadDeadline() (time.Time, error) {
	if m.Created <= 0 || m.Expires <= m.Created || m.Expires-m.Created > int64(time.Hour/time.Second) || m.Limits.RuntimeSeconds <= 0 || m.Limits.RuntimeSeconds > int(trust.MaxWorkloadLifetime/time.Second) {
		return time.Time{}, fmt.Errorf("invalid job lifetime budget")
	}
	lifetime := time.Duration(m.Expires-m.Created)*time.Second + time.Duration(m.Limits.RuntimeSeconds)*time.Second + CleanupAllowance
	if lifetime > trust.MaxWorkloadLifetime {
		return time.Time{}, fmt.Errorf("job admission, runtime and cleanup must fit within seven days")
	}
	return time.Unix(m.Created, 0).Add(lifetime), nil
}
