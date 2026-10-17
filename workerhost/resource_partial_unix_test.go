//go:build darwin || linux

package workerhost

import (
	"context"
	"errors"
	"testing"
	"testing/synctest"
	"time"

	"github.com/mrothroc/mixlab/workerhost/contract"
)

func TestLearnerResourcePartialBreachBeforeDiskTimeout(t *testing.T) {
	for _, tc := range []struct {
		name  string
		usage resourceUsage
		kind  string
	}{
		{"rss", resourceUsage{Processes: []processUsage{{Identity: "worker", RSS: 101}}}, "memory_bytes"},
		{"cpu", resourceUsage{Processes: []processUsage{{Identity: "worker", CPU: 2 * time.Second}}}, "cpu_nanoseconds"},
		{"disk", resourceUsage{Disk: 101}, "disk_bytes"},
	} {
		for _, shutdown := range []bool{false, true} {
			name := tc.name + "/probe-timeout"
			if shutdown {
				name = tc.name + "/shutdown"
			}
			t.Run(name, func(t *testing.T) {
				synctest.Test(t, func(t *testing.T) {
					ctx, cancel := context.WithCancel(context.Background())
					defer cancel()
					control := &resourceControlFixture{}
					calls := 0
					sample := func(probe context.Context, _ *ownedProcess, _ string) (resourceUsage, error) {
						calls++
						if shutdown {
							cancel()
						}
						<-probe.Done()
						return tc.usage, probe.Err()
					}
					stop := watchResourceMonitor(ctx, cancel, nil, "", contract.Limits{CPUSeconds: 1, MemoryBytes: 100, DiskBytes: 100}, sample, control)
					<-ctx.Done()
					var limit *ResourceLimitError
					if err := stop(); !errors.As(err, &limit) || limit.Resource != tc.kind || calls != 1 || control.stops != 0 || control.resumes != 0 {
						t.Fatalf("partial breach became transient: calls=%d control=%+v err=%v", calls, control, err)
					}
				})
			})
		}
	}
}

func TestLearnerResourcePartialCPUAcrossTimeoutAndResume(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		control := &resourceControlFixture{}
		calls := 0
		sample := func(probe context.Context, _ *ownedProcess, _ string) (resourceUsage, error) {
			calls++
			switch calls {
			case 1:
				return resourceUsage{Processes: []processUsage{
					{Identity: "worker", CPU: 2 * time.Second},
					{Identity: "exited-child", CPU: time.Second},
				}}, nil
			case 2:
				<-probe.Done()
				return resourceUsage{Processes: []processUsage{{Identity: "worker", CPU: 3 * time.Second}}}, probe.Err()
			case 3:
				return sampleTimeout(probe) // No process observation: retain the baseline.
			case 4:
				return resourceUsage{Processes: []processUsage{{Identity: "worker", CPU: 3 * time.Second}}}, nil
			default:
				<-probe.Done()
				return resourceUsage{Processes: []processUsage{{Identity: "worker", CPU: 4 * time.Second}}}, probe.Err()
			}
		}
		stop := watchResourceMonitor(ctx, cancel, nil, "", contract.Limits{CPUSeconds: 4}, sample, control)
		<-ctx.Done()
		var limit *ResourceLimitError
		if err := stop(); !errors.As(err, &limit) || limit.Resource != "cpu_nanoseconds" || limit.Observed != int64(5*time.Second) || calls != 5 || control.resumes != 1 {
			t.Fatalf("partial CPU evidence lost/double counted: calls=%d control=%+v err=%v", calls, control, err)
		}
	})
}
