package integration

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/workercontrol"
	"github.com/mrothroc/mixlab/workerprobe"
)

func TestCheckedPreparationRevalidatesLocalEvidence(t *testing.T) {
	for _, mode := range []string{"valid", "build", "data", "gpu", "artifact", "expired-during-check"} {
		t.Run(mode, func(t *testing.T) {
			ctx := context.Background()
			f := newTLSFixture(t)
			node := id(t)
			dir, e := filepath.EvalSymlinks(t.TempDir())
			check(t, e)
			check(t, os.Chmod(dir, 0700))
			p, e := statehome.Resolve(statehome.Options{ExactDir: dir}, statehome.Context{Kind: statehome.Agent})
			check(t, e)
			s, e := nodeagent.Initialize(ctx, p, f.a.Cluster(), node, 1)
			check(t, e)
			actor, e := trust.AuthenticatePrincipal(f.a, f.view, f.client.Chain, f.now)
			check(t, e)
			// Reserve identity is chosen before installing the immutable catalog,
			// but no reservation exists until the profile has been published.
			run := id(t)
			stubLease := nodeagent.Lease{ID: id(t), Run: run}
			signed := signedJobFixture(t, f, node, stubLease)
			m := signed.Manifest
			probe := workerprobe.Report{Version: workerprobe.Version, BuildID: m.BuildID, BuildVersion: "test", WorkerProtocol: workercontrol.Version, OS: "darwin", Arch: "arm64", Available: true, MLXVersion: "0.32.1", MLXSupported: true, DeviceKind: "metal", DeviceName: "test", Backends: []string{"ring"}, DTypes: []string{"bf16", "fp32"}, CustomOps: []string{"mixlab-ir-v1"}}
			profile := nodeagent.Profile{Version: nodeagent.ProfileVersion, Node: node, DisplayName: "node", Generation: 1, Probe: probe, ProbeObservedAt: f.now.Unix(), Limits: m.Limits, Datasets: []nodeagent.LocalDataset{{Dataset: nodeagent.Dataset{Selector: m.DatasetSelector, ID: m.DatasetID}, TrainPattern: "/private/owned/train.bin"}}}
			check(t, s.InstallProfile(ctx, profile))
			lease, e := s.Reserve(ctx, actor, nodeagent.Reserve{IdempotencyKey: id(t), ExpectedNodeVersion: 1, CapabilityGeneration: 1, Run: run, TTLSeconds: 300}, f.now)
			check(t, e)
			m.Lease = lease.ID
			if mode == "artifact" {
				m.Artifacts = []nodejob.ArtifactRef{{SHA256: m.BuildID, Bytes: 100, Kind: "weights"}}
			}
			q, e := m.SigningRequest()
			check(t, e)
			proof, e := trust.SignPrincipalProof(f.a, f.client.Chain, f.client.Key, f.view, q, f.now)
			check(t, e)
			accepted, e := nodejob.Accept(f.a, f.view, actor, node, nodejob.Signed{Manifest: m, Proof: proof}, f.now)
			check(t, e)
			ports := nodeagent.PreparationPorts{Clock: func() time.Time {
				if mode == "expired-during-check" {
					return f.now.Add(2 * time.Minute)
				}
				return f.now
			}, Probe: func(context.Context) (workerprobe.Report, error) {
				r := probe
				if mode == "build" {
					r.BuildID = m.ConfigHash
				}
				if mode == "gpu" {
					return r, errors.New("GPU unavailable")
				}
				return r, nil
			}, DatasetID: func(_ context.Context, path string) (string, error) {
				if path != "/private/owned/train.bin" {
					t.Fatal("untrusted dataset path", path)
				}
				if mode == "data" {
					return m.ConfigHash, nil
				}
				return m.DatasetID, nil
			}, ArtifactPresent: func(context.Context, nodejob.ArtifactRef) error { return errors.New("artifact unavailable") }}
			j, e := s.PrepareChecked(ctx, actor, accepted, lease.Version, ports, f.now)
			if (e == nil) != (mode == "valid") {
				t.Fatal(mode, j, e)
			}
			av, e := s.Availability(f.now)
			check(t, e)
			if mode != "valid" && (av.Lease.State != nodeagent.Reserved || av.Lease.Job != "") {
				t.Fatal("failed local checks committed a job")
			}
		})
	}
}
