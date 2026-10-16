//go:build darwin || linux

package workerhost

import (
	"context"
	"fmt"
	"math"
	"time"
)

// A learner's persistent state is deliberately outside its fresh attempt
// directory, but both belong to the same approved disk allowance.
func (p LaunchPlan) withPrivateDisk(ctx context.Context, attempt int64) (int64, error) {
	if p.Learner == nil {
		return attempt, nil
	}
	private, err := attemptDiskBytes(ctx, p.LearnerPrivateDirectory.Dir())
	if err != nil {
		return 0, err
	}
	if private > int64(p.Learner.PrivateMaxBytes) {
		return 0, &ResourceLimitError{"learner_private_bytes", private, int64(p.Learner.PrivateMaxBytes)}
	}
	if attempt < 0 || private > math.MaxInt64-attempt {
		return 0, fmt.Errorf("combined learner disk usage overflows")
	}
	return attempt + private, nil
}

func (p LaunchPlan) resourceSampler(base resourceSampler) resourceSampler {
	if p.Learner == nil {
		return base
	}
	if base == nil {
		base = sampleResources
	}
	return func(ctx context.Context, process *ownedProcess, dir string) (resourceUsage, error) {
		u, err := base(ctx, process, dir)
		if err == nil {
			u.Disk, err = p.withPrivateDisk(ctx, u.Disk)
		}
		return u, err
	}
}

func (p LaunchPlan) checkDisk(parent context.Context) error {
	if p.Learner == nil {
		return checkAttemptDisk(parent, p.Directory.Dir(), p.Limits.DiskBytes)
	}
	ctx, cancel := context.WithTimeout(parent, 2*time.Second)
	defer cancel()
	n, err := attemptDiskBytes(ctx, p.Directory.Dir())
	if err != nil {
		return err
	}
	n, err = p.withPrivateDisk(ctx, n)
	if err != nil {
		return err
	}
	if n > p.Limits.DiskBytes {
		return &ResourceLimitError{"disk_bytes", n, p.Limits.DiskBytes}
	}
	return nil
}
