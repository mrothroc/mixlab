//go:build darwin || linux

package workerhost

import (
	"context"
	"errors"
	"fmt"
	"math"
	"time"
)

type diskSampler func(context.Context, string) (int64, error)

// A learner's persistent state is deliberately outside its fresh attempt
// directory, but both belong to the same approved disk allowance.
func (p LaunchPlan) withPrivateDisk(ctx context.Context, attempt int64) (int64, error) {
	return p.withPrivateDiskSample(ctx, attempt, attemptDiskBytes)
}

func (p LaunchPlan) withPrivateDiskSample(ctx context.Context, attempt int64, sample diskSampler) (int64, error) {
	if p.Learner == nil {
		return attempt, nil
	}
	private, err := sample(ctx, p.LearnerPrivateDirectory.Dir())
	if attempt < 0 || private < 0 || private > math.MaxInt64-attempt {
		return 0, errors.Join(fmt.Errorf("combined learner disk usage overflows"), err)
	}
	total := attempt + private
	if private > int64(p.Learner.PrivateMaxBytes) {
		return total, errors.Join(&ResourceLimitError{"learner_private_bytes", private, int64(p.Learner.PrivateMaxBytes)}, err)
	}
	return total, err
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
	return p.checkLearnerDisk(ctx, attemptDiskBytes)
}

// The caller chooses the lifetime: prelaunch uses its short probe, while final
// verification uses the original supervision context and its remaining budget.
func (p LaunchPlan) checkLearnerDisk(ctx context.Context, sample diskSampler) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	n, err := sample(ctx, p.Directory.Dir())
	if n > p.Limits.DiskBytes {
		return errors.Join(&ResourceLimitError{"disk_bytes", n, p.Limits.DiskBytes}, err)
	}
	if err != nil {
		return err
	}
	n, err = p.withPrivateDiskSample(ctx, n, sample)
	if n > p.Limits.DiskBytes {
		return errors.Join(&ResourceLimitError{"disk_bytes", n, p.Limits.DiskBytes}, err)
	}
	if err != nil {
		return err
	}
	return ctx.Err()
}
