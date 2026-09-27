package nodeagent

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/workercontrol"
	"github.com/mrothroc/mixlab/workerprobe"
)

func profileFixture(s *Store) Profile {
	return Profile{Version: ProfileVersion, Node: s.node, DisplayName: "test node", Generation: 1, ProbeObservedAt: leaseNow.Unix(), Probe: workerprobe.Report{Version: workerprobe.Version, BuildID: strings.Repeat("a", 64), BuildVersion: "test", WorkerProtocol: workercontrol.Version, OS: "darwin", Arch: "arm64", Available: true, MLXVersion: "0.32.1", MLXSupported: true, DeviceKind: "metal", DeviceName: "test", Backends: []string{"ring"}, DTypes: []string{"bf16", "fp32"}, CustomOps: []string{"mixlab-ir-v1"}}, Limits: nodejob.Limits{RuntimeSeconds: 60, CPUSeconds: 120, MemoryBytes: 1 << 30, DiskBytes: 1 << 30, LogBytes: 1 << 20}, Datasets: []LocalDataset{{Dataset: Dataset{Selector: "toy.train", ID: strings.Repeat("b", 64)}, TrainPattern: "/private/local/train*.bin"}}}
}
func TestNodeProfileCapabilityIsolationAndGenerationRecovery(t *testing.T) {
	ctx := context.Background()
	s, actor := leaseFixture(t)
	p := profileFixture(s)
	if err := s.InstallProfile(ctx, p); err != nil {
		t.Fatal(err)
	}
	c, err := s.Capabilities(ctx, actor, leaseNow)
	if err != nil {
		t.Fatal(err)
	}
	b, _ := json.Marshal(c)
	if strings.Contains(string(b), "/private/local") {
		t.Fatal("capability response leaked local dataset path")
	}
	if !c.Recruitable || c.Generation != 1 || c.ProbeObservedAt != p.ProbeObservedAt {
		t.Fatal(c)
	}
	old, err := s.path.ReadFileLimit(leaseFile, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	p.Generation = 2
	p.DisplayName = "updated"
	if err := s.InstallProfile(ctx, p); err != nil {
		t.Fatal(err)
	}
	if err := s.path.WriteFile(leaseFile, old); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Capabilities(ctx, actor, leaseNow); err == nil {
		t.Fatal("unreconciled profile advertised")
	}
	if err := s.InstallProfile(ctx, p); err != nil {
		t.Fatal(err)
	}
	c, err = s.Capabilities(ctx, actor, leaseNow)
	if err != nil || c.Generation != 2 {
		t.Fatal(c, err)
	}
	q := reservation("1")
	q.CapabilityGeneration = 2
	q.ExpectedNodeVersion = c.Availability.NodeVersion
	if _, err := s.Reserve(ctx, actor, q, leaseNow); err != nil {
		t.Fatal(err)
	}
	p.Generation = 3
	if err := s.InstallProfile(ctx, p); err == nil {
		t.Fatal("busy profile replaced")
	}
	c, err = s.Capabilities(ctx, actor, leaseNow)
	if err != nil || c.Recruitable {
		t.Fatal("busy node recruitable", err)
	}
}
func TestNodeProfileRejectsMissingHistoryAndUnavailableGPU(t *testing.T) {
	s, actor := leaseFixture(t)
	p := profileFixture(s)
	p.Probe.Available = false
	p.Probe.DeviceKind = "none"
	p.Probe.DeviceName = ""
	p.Probe.DTypes = []string{}
	p.Probe.CustomOps = []string{}
	if err := s.InstallProfile(context.Background(), p); err != nil {
		t.Fatal(err)
	}
	c, err := s.Capabilities(context.Background(), actor, leaseNow)
	if err != nil || c.Recruitable {
		t.Fatal("unavailable GPU recruitable", err)
	}
	if err := os.Remove(filepath.Join(s.path.Dir(), profileFile)); err != nil {
		t.Fatal(err)
	}
	if err := s.InstallProfile(context.Background(), p); err == nil {
		t.Fatal("missing published profile regenerated")
	}
}

func TestNodeProfileInitialPublicationRecovery(t *testing.T) {
	for _, point := range []string{"claim", "profile"} {
		t.Run(point, func(t *testing.T) {
			s, actor := leaseFixture(t)
			p := profileFixture(s)
			ctx := context.Background()
			if err := s.InstallProfile(ctx, p); err != nil {
				t.Fatal(err)
			}
			if err := os.Remove(filepath.Join(s.path.Dir(), profileReady)); err != nil {
				t.Fatal(err)
			}
			if point == "claim" {
				if err := os.Remove(filepath.Join(s.path.Dir(), profileFile)); err != nil {
					t.Fatal(err)
				}
			}
			if _, err := s.Capabilities(ctx, actor, leaseNow); err == nil {
				t.Fatal("unpublished profile visible")
			}
			changed := p
			changed.DisplayName = "different"
			if err := s.InstallProfile(ctx, changed); err == nil {
				t.Fatal("changed incomplete initializer accepted")
			}
			if err := s.InstallProfile(ctx, p); err != nil {
				t.Fatal(err)
			}
			if _, err := s.Capabilities(ctx, actor, leaseNow); err != nil {
				t.Fatal(err)
			}
		})
	}
}
