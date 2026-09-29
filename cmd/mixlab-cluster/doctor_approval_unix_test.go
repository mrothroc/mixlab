//go:build darwin || linux

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/workercontrol"
	"github.com/mrothroc/mixlab/workerjob"
	"github.com/mrothroc/mixlab/workerprobe"
)

func TestDoctorDetectsInterruptedApproval(t *testing.T) {
	dir, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(dir, 0700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("HOME", dir)
	exe := filepath.Join(dir, "worker")
	if err := os.WriteFile(exe, []byte("fixture"), 0700); err != nil {
		t.Fatal(err)
	}
	hash, err := workerjob.FileDigest(exe)
	if err != nil {
		t.Fatal(err)
	}
	id := strings.Repeat("a", 32)
	report := workerprobe.Report{Version: workerprobe.Version, BuildID: hash, BuildVersion: "test", WorkerProtocol: workercontrol.Version, OS: "darwin", Arch: "arm64", DeviceKind: "none", Backends: []string{}, DTypes: []string{}, CustomOps: []string{}}
	i := clusterapp.NodeInstallation{Cluster: id, Node: id, PrincipalDirectory: filepath.Join(dir, "identity"), WorkerBinary: exe, GuardianBinary: exe, WorkerBuild: hash, GuardianBuild: hash, RelayAddress: "127.0.0.1:32499"}
	profile := nodeagent.Profile{Version: nodeagent.ProfileVersion, Node: id, DisplayName: "test", TransportEndpoint: i.RelayAddress, Generation: 1, Probe: report, ProbeObservedAt: time.Now().Unix(), Limits: nodejob.Limits{RuntimeSeconds: 60, CPUSeconds: 120, MemoryBytes: 1 << 30, DiskBytes: 1 << 30, LogBytes: 1 << 20}, Datasets: []nodeagent.LocalDataset{}}
	p, err := statehome.Resolve(statehome.Options{ExactDir: filepath.Join(dir, "agent")}, statehome.Context{Kind: statehome.Agent})
	if err != nil {
		t.Fatal(err)
	}
	if err := clusterapp.InitializeNodeInstallation(context.Background(), p, i, profile); err != nil {
		t.Fatal(err)
	}
	checkApproval := func(want string) {
		t.Helper()
		var out, stderr bytes.Buffer
		run([]string{"doctor", "-agent-state-dir", p.Dir()}, &out, &stderr)
		var result struct {
			Checks []doctorCheck `json:"checks"`
		}
		if err := json.Unmarshal(out.Bytes(), &result); err != nil {
			t.Fatal(err, stderr.String())
		}
		for _, check := range result.Checks {
			if check.Check == "executable_approval" {
				if check.Status != want {
					t.Fatalf("approval=%+v, want %s", check, want)
				}
				return
			}
		}
		t.Fatal("no executable approval diagnostic")
	}
	checkApproval("ok")
	s, err := nodeagent.Open(p, id, id)
	if err != nil {
		t.Fatal(err)
	}
	err = s.ReapproveWorker(context.Background(), func(context.Context) (workerprobe.Report, error) {
		return report, nil
	}, func() error { return errors.New("simulated interrupted pin publication") }, time.Now())
	if err == nil {
		t.Fatal("expected interrupted reapproval")
	}
	checkApproval("failed")
}
