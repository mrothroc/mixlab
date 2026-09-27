//go:build darwin || linux

package clusterapp

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/workercontrol"
	"github.com/mrothroc/mixlab/workerjob"
	"github.com/mrothroc/mixlab/workerprobe"
)

func TestNodeInstallationAtomicSetupAndMissingState(t *testing.T) {
	dir, err := filepath.EvalSymlinks(t.TempDir())
	check(t, err)
	check(t, os.Chmod(dir, 0700))
	worker := filepath.Join(dir, "approved-worker")
	check(t, os.WriteFile(worker, []byte("synthetic binary identity; never executed"), 0700))
	hash, err := workerjob.FileDigest(worker)
	check(t, err)
	id := strings.Repeat("a", 32)
	i := NodeInstallation{Cluster: id, Node: id, PrincipalDirectory: filepath.Join(dir, "identity"), WorkerBinary: worker, WorkerBuild: hash, GuardianBinary: worker, GuardianBuild: hash, RelayAddress: "127.0.0.1:32499"}
	probe := workerprobe.Report{Version: workerprobe.Version, BuildID: hash, BuildVersion: "test", WorkerProtocol: workercontrol.Version, OS: "darwin", Arch: "arm64", DeviceKind: "none", Backends: []string{}, DTypes: []string{}, CustomOps: []string{}}
	profile := nodeagent.Profile{Version: nodeagent.ProfileVersion, Node: id, DisplayName: "test", TransportEndpoint: i.RelayAddress, Generation: 1, Probe: probe, ProbeObservedAt: testNow.Unix(), Limits: nodejob.Limits{RuntimeSeconds: 60, CPUSeconds: 120, MemoryBytes: 1 << 30, DiskBytes: 1 << 30, LogBytes: 1 << 20}, Datasets: []nodeagent.LocalDataset{}}
	p, err := statehome.Resolve(statehome.Options{ExactDir: filepath.Join(dir, "agent")}, statehome.Context{Kind: statehome.Agent})
	check(t, err)
	check(t, InitializeNodeInstallation(context.Background(), p, i, profile))
	got, err := OpenNodeInstallation(p)
	check(t, err)
	i.Version = nodeInstallationVersion
	if got != i {
		t.Fatal("installation changed approval", got)
	}
	if err := InitializeNodeInstallation(context.Background(), p, i, profile); err == nil {
		t.Fatal("setup reset existing installation")
	}
	store, err := nodeagent.Open(p, id, id)
	check(t, err)
	av, err := store.Availability(testNow)
	check(t, err)
	if !av.Available {
		t.Fatal("new node not available")
	}
	check(t, os.Remove(filepath.Join(p.Dir(), nodeInstallationFile)))
	if _, err := OpenNodeInstallation(p); err == nil {
		t.Fatal("missing installation silently repaired")
	}
	check(t, os.WriteFile(worker, []byte("changed"), 0700))
	if err := i.validate(); err == nil {
		t.Fatal("approved executable mutation accepted")
	}
}
