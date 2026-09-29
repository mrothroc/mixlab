//go:build darwin || linux

package clusterapp

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/workercontrol"
	"github.com/mrothroc/mixlab/workerjob"
	"github.com/mrothroc/mixlab/workerprobe"
)

func approvalFixture(t *testing.T) (statehome.Path, NodeInstallation, workerprobe.Report) {
	t.Helper()
	dir, err := filepath.EvalSymlinks(t.TempDir())
	check(t, err)
	check(t, os.Chmod(dir, 0700))
	exe := filepath.Join(dir, "binary")
	check(t, os.WriteFile(exe, []byte("old"), 0700))
	hash, err := workerjob.FileDigest(exe)
	check(t, err)
	id := strings.Repeat("a", 32)
	i := NodeInstallation{Cluster: id, Node: id, PrincipalDirectory: filepath.Join(dir, "identity"), WorkerBinary: exe, GuardianBinary: exe, WorkerBuild: hash, GuardianBuild: hash, RelayAddress: "127.0.0.1:32499"}
	r := workerprobe.Report{Version: workerprobe.Version, BuildID: hash, BuildVersion: "test", WorkerProtocol: workercontrol.Version, OS: "darwin", Arch: "arm64", DeviceKind: "none", Backends: []string{}, DTypes: []string{}, CustomOps: []string{}}
	profile := nodeagent.Profile{Version: nodeagent.ProfileVersion, Node: id, DisplayName: "test", TransportEndpoint: i.RelayAddress, Generation: 1, Probe: r, ProbeObservedAt: testNow.Unix(), Limits: nodejob.Limits{RuntimeSeconds: 60, CPUSeconds: 120, MemoryBytes: 1 << 30, DiskBytes: 1 << 30, LogBytes: 1 << 20}, Datasets: []nodeagent.LocalDataset{}}
	p, err := statehome.Resolve(statehome.Options{ExactDir: filepath.Join(dir, "agent")}, statehome.Context{Kind: statehome.Agent})
	check(t, err)
	check(t, InitializeNodeInstallation(context.Background(), p, i, profile))
	i.Version = nodeInstallationVersion
	return p, i, r
}

func TestReapproveInstallationChangesPinsNotIdentity(t *testing.T) {
	p, old, report := approvalFixture(t)
	check(t, os.WriteFile(old.WorkerBinary, []byte("upgrade"), 0700))
	if _, err := OpenNodeInstallation(p); err == nil {
		t.Fatal("changed binary accepted")
	}
	i, err := ReapproveNodeInstallation(context.Background(), p, old.WorkerBinary, old.GuardianBinary, func(_ context.Context, binary, hash string) (workerprobe.Report, error) {
		report.BuildID = hash
		return report, nil
	})
	check(t, err)
	if i.Cluster != old.Cluster || i.Node != old.Node || i.PrincipalDirectory != old.PrincipalDirectory || i.WorkerBuild == old.WorkerBuild {
		t.Fatal(i)
	}
	got, err := OpenNodeInstallation(p)
	check(t, err)
	if got != i {
		t.Fatal(got)
	}
	s, err := nodeagent.Open(p, i.Cluster, i.Node)
	check(t, err)
	check(t, s.CheckApprovedWorker(i.WorkerBuild))
}

func TestReapproveCannotRaceRunningAgent(t *testing.T) {
	p, i, _ := approvalFixture(t)
	ready := make(chan struct{})
	release := make(chan struct{})
	done := make(chan error, 1)
	go func() {
		done <- p.WithProcessLock(context.Background(), NodeServiceLock, func() error { close(ready); <-release; return nil })
	}()
	<-ready
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()
	_, err := ReapproveNodeInstallation(ctx, p, i.WorkerBinary, i.GuardianBinary, func(context.Context, string, string) (workerprobe.Report, error) {
		t.Error("probe while agent running")
		return workerprobe.Report{}, nil
	})
	close(release)
	check(t, <-done)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatal(err)
	}
	got, err := OpenNodeInstallation(p)
	check(t, err)
	if got != i {
		t.Fatal("failed approval changed pins")
	}
}
