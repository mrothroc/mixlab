//go:build darwin || linux

package workerhost

import (
	"context"
	"errors"
	"math"
	"testing"
)

func TestLearnerDiskCountsPersistentAndAttemptDirectories(t *testing.T) {
	_, p := learnerPlan(t, "success")
	if err := p.prepareLearnerStorage(); err != nil {
		t.Fatal(err)
	}
	if err := p.Directory.WriteFile("scratch", make([]byte, 600)); err != nil {
		t.Fatal(err)
	}
	if err := p.LearnerPrivateDirectory.WriteFile("snapshot", make([]byte, 600)); err != nil {
		t.Fatal(err)
	}
	p.Limits.DiskBytes = 1000
	p.Learner.PrivateMaxBytes = 800
	var limit *ResourceLimitError
	if err := p.checkDisk(context.Background()); !errors.As(err, &limit) || limit.Resource != "disk_bytes" || limit.Observed != 1200 {
		t.Fatalf("combined pre/post-launch disk check: %v", err)
	}
	base := func(context.Context, *ownedProcess, string) (resourceUsage, error) {
		return resourceUsage{Disk: 600}, nil
	}
	u, err := p.resourceSampler(base)(context.Background(), nil, p.Directory.Dir())
	if err != nil || u.Disk != 1200 {
		t.Fatal("live sampler omitted private state", u, err)
	}
	p.Learner.PrivateMaxBytes = 500
	if _, err = p.resourceSampler(base)(context.Background(), nil, p.Directory.Dir()); !errors.As(err, &limit) || limit.Resource != "learner_private_bytes" {
		t.Fatal("private budget not enforced by host", err)
	}
	p.Learner.PrivateMaxBytes = 800
	if _, err = p.withPrivateDisk(context.Background(), math.MaxInt64); err == nil {
		t.Fatal("disk accounting overflow accepted")
	}
}

func TestLearnerPrivateBudgetCannotExceedLaunchApproval(t *testing.T) {
	_, p := learnerPlan(t, "success")
	p.Learner.PrivateMaxBytes = uint64(p.Limits.DiskBytes)
	if err := p.prepareLearnerStorage(); err == nil {
		t.Fatal("private state plus logs exceeded launch approval")
	}
}
