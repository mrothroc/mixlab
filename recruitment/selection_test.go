package recruitment

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/discovery"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/workercontrol"
	"github.com/mrothroc/mixlab/workerprobe"
)

var selectionNow = time.Unix(1800000000, 0)

func fixture() (Requirements, Observation) {
	r := Requirements{Cluster: strings.Repeat("a", 32), Count: 2, BuildID: strings.Repeat("b", 64), MLXVersion: "0.32.1", DeviceKind: "metal", DType: "fp32", CustomOps: []string{"mixlab-ir-v1"}, DatasetSelector: "toy", DatasetID: strings.Repeat("c", 64), ProbeMaxAge: time.Minute,
		Limits: nodejob.Limits{RuntimeSeconds: 60, CPUSeconds: 120, MemoryBytes: 1 << 30, DiskBytes: 1 << 30, LogBytes: 1 << 20}}
	id := strings.Repeat("1", 32)
	p := trust.AuthenticatedPrincipal{Cluster: r.Cluster, Role: trust.Node, Principal: id, NotBefore: selectionNow.Add(-time.Hour), ExpiresAt: selectionNow.Add(time.Hour)}
	c := nodeagent.Capabilities{Version: nodeagent.CapabilityVersion, Node: id, DisplayName: "node", Generation: 1, ObservedAt: selectionNow.Unix(), ProbeObservedAt: selectionNow.Unix(), Limits: r.Limits, ManifestVersions: []string{nodejob.Version}, Datasets: []nodeagent.Dataset{{Selector: r.DatasetSelector, ID: r.DatasetID}}, Availability: nodeagent.Availability{NodeVersion: 1, CapabilityGeneration: 1, Available: true}, Recruitable: true,
		Probe: workerprobe.Report{Version: workerprobe.Version, BuildID: r.BuildID, BuildVersion: "test", WorkerProtocol: workercontrol.Version, OS: "darwin", Arch: "arm64", Available: true, MLXVersion: r.MLXVersion, MLXSupported: true, DeviceKind: "metal", DeviceName: "test", Backends: []string{"ring"}, DTypes: []string{"fp32"}, CustomOps: []string{"mixlab-ir-v1"}}}
	return r, Observation{p, c}
}

func TestSelectionDeterministicAuthenticatedIdentity(t *testing.T) {
	r, first := fixture()
	_, second := fixture()
	second.Peer.Principal, second.Capabilities.Node = strings.Repeat("2", 32), strings.Repeat("2", 32)
	_, third := fixture()
	third.Peer.Principal, third.Capabilities.Node = strings.Repeat("3", 32), strings.Repeat("3", 32)
	observations := map[string]Observation{"z.local:1": first, "a.local:1": second, "b.local:1": third, "z.local:2": first}
	hints := []discovery.Hint{}
	for endpoint := range observations {
		hints = append(hints, discovery.Hint{Service: discovery.Node, Endpoint: endpoint, Claims: discovery.Claims{Cluster: "forged", Node: "forged"}})
	}
	hints = append(hints, hints[0], discovery.Hint{Service: discovery.Node, Endpoint: "bad/endpoint"}, discovery.Hint{Service: discovery.Authority, Endpoint: "ignored.local:1"})
	calls := 0
	lookup := func(_ context.Context, endpoint string) (Observation, error) {
		calls++
		return observations[endpoint], nil
	}
	a, err := Select(context.Background(), hints, r, lookup, func() time.Time { return selectionNow })
	if err != nil || len(a.Selected) != 2 || calls != 4 || a.Selected[0].Endpoint != "z.local:1" || a.Selected[1].Endpoint != "a.local:1" {
		t.Fatal(a, calls, err)
	}
	slices.Reverse(hints)
	b, err := Select(context.Background(), hints, r, lookup, func() time.Time { return selectionNow })
	if err != nil || !reflect.DeepEqual(a, b) {
		t.Fatal("selection depends on advertisement order", err)
	}
	encoded, _ := json.Marshal(a)
	if strings.Contains(string(encoded), "forged") {
		t.Fatal("TXT identity reached decision evidence")
	}
}

func TestSelectionRejectsUnusableCandidates(t *testing.T) {
	cases := []struct {
		name, reason string
		change       func(*Observation)
	}{
		{"role", "wrong_identity", func(o *Observation) { o.Peer.Role = trust.Controller }},
		{"cluster", "wrong_identity", func(o *Observation) { o.Peer.Cluster = strings.Repeat("d", 32) }},
		{"node", "wrong_identity", func(o *Observation) { o.Capabilities.Node = strings.Repeat("d", 32) }},
		{"expired", "wrong_identity", func(o *Observation) { o.Peer.ExpiresAt = selectionNow }},
		{"future", "wrong_identity", func(o *Observation) { o.Peer.NotBefore = selectionNow.Add(time.Second) }},
		{"generation", "invalid_capabilities", func(o *Observation) { o.Capabilities.Generation++ }},
		{"catalog", "invalid_capabilities", func(o *Observation) {
			o.Capabilities.Datasets = append(o.Capabilities.Datasets, o.Capabilities.Datasets[0])
		}},
		{"observation", "stale_capabilities", func(o *Observation) { o.Capabilities.ObservedAt -= 31; o.Capabilities.ProbeObservedAt -= 31 }},
		{"probe", "stale_capabilities", func(o *Observation) { o.Capabilities.ProbeObservedAt -= 61 }},
		{"future observation", "stale_capabilities", func(o *Observation) { o.Capabilities.ObservedAt++ }},
		{"build", "incompatible_worker", func(o *Observation) { o.Capabilities.Probe.BuildID = strings.Repeat("e", 64) }},
		{"mlx", "incompatible_worker", func(o *Observation) { o.Capabilities.Probe.MLXVersion = "other" }},
		{"ops", "incompatible_worker", func(o *Observation) { o.Capabilities.Probe.CustomOps = []string{} }},
		{"device", "incompatible_worker", func(o *Observation) { o.Capabilities.Probe.DeviceKind = "cuda" }},
		{"dtype", "incompatible_worker", func(o *Observation) { o.Capabilities.Probe.DTypes = []string{"bf16"} }},
		{"manifest", "incompatible_worker", func(o *Observation) { o.Capabilities.ManifestVersions = []string{"future"} }},
		{"memory", "insufficient_resources", func(o *Observation) {
			total, free := uint64(1<<30), uint64(1<<20)
			o.Capabilities.Probe.MemoryBytes, o.Capabilities.Probe.FreeMemoryBytes = &total, &free
		}},
		{"policy", "insufficient_resources", func(o *Observation) { o.Capabilities.Limits.RuntimeSeconds-- }},
		{"data", "dataset_mismatch", func(o *Observation) { o.Capabilities.Datasets[0].ID = strings.Repeat("e", 64) }},
		{"busy", "not_recruitable", func(o *Observation) {
			o.Capabilities.Recruitable = false
			o.Capabilities.Availability.Available = false
			o.Capabilities.Availability.Lease = &nodeagent.Lease{Format: nodeagent.LeaseVersion, ID: strings.Repeat("a", 32), Node: o.Capabilities.Node, Controller: strings.Repeat("b", 32), Run: strings.Repeat("c", 32), CapabilityGeneration: 1, Version: 1, State: nodeagent.Reserved, Created: selectionNow.Unix(), Expires: selectionNow.Add(time.Minute).Unix(), RenewBy: selectionNow.Add(30 * time.Second).Unix()}
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r, o := fixture()
			tc.change(&o)
			result, err := Select(context.Background(), []discovery.Hint{{Service: discovery.Node, Endpoint: "node.local:1"}}, r, func(context.Context, string) (Observation, error) { return o, nil }, func() time.Time { return selectionNow })
			if err == nil || len(result.Selected) != 0 || len(result.Decisions) != 1 || result.Decisions[0].Reason != tc.reason {
				t.Fatal(result, err)
			}
		})
	}
}

func TestSelectionConflictAndCancellation(t *testing.T) {
	r, o := fixture()
	hints := []discovery.Hint{{Service: discovery.Node, Endpoint: "a.local:1"}, {Service: discovery.Node, Endpoint: "b.local:1"}}
	result, err := Select(context.Background(), hints, r, func(_ context.Context, endpoint string) (Observation, error) {
		copy := o
		if endpoint == "b.local:1" {
			copy.Capabilities.DisplayName = "conflicting profile"
		}
		return copy, nil
	}, func() time.Time { return selectionNow })
	if err == nil || len(result.Selected) != 0 || result.Decisions[1].Reason != "conflicting_node_observations" {
		t.Fatal(result, err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := Select(ctx, hints, r, func(context.Context, string) (Observation, error) {
		t.Fatal("queried after cancellation")
		return o, nil
	}, func() time.Time { return selectionNow }); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
}

func TestSelectionRejectsNegativeSiblingObservation(t *testing.T) {
	for _, reason := range []string{"incompatible_worker", "insufficient_resources", "dataset_mismatch", "not_recruitable"} {
		t.Run(reason, func(t *testing.T) {
			r, o := fixture()
			hints := []discovery.Hint{{Service: discovery.Node, Endpoint: "a.local:1"}, {Service: discovery.Node, Endpoint: "a.local:2"}, {Service: discovery.Node, Endpoint: "b.local:1"}}
			out, err := Select(context.Background(), hints, r, func(_ context.Context, endpoint string) (Observation, error) {
				_, observed := fixture()
				if endpoint == "b.local:1" {
					observed.Peer.Principal, observed.Capabilities.Node = strings.Repeat("2", 32), strings.Repeat("2", 32)
				}
				if endpoint == "a.local:2" {
					switch reason {
					case "incompatible_worker":
						observed.Capabilities.Probe.BuildID = strings.Repeat("e", 64)
					case "insufficient_resources":
						observed.Capabilities.Limits.RuntimeSeconds--
					case "dataset_mismatch":
						observed.Capabilities.Datasets = []nodeagent.Dataset{}
					case "not_recruitable":
						observed.Capabilities.Recruitable = false
						observed.Capabilities.Probe.MLXSupported = false
					}
				}
				return observed, nil
			}, func() time.Time { return selectionNow })
			if err == nil || len(out.Selected) != 0 || out.Decisions[0].Node != o.Peer.Principal || out.Decisions[0].Reason != reason {
				t.Fatal(out, err)
			}
		})
	}
}

func TestSelectionRechecksCredentialExpiryAtCommit(t *testing.T) {
	r, _ := fixture()
	hints := []discovery.Hint{{Service: discovery.Node, Endpoint: "a.local:1"}, {Service: discovery.Node, Endpoint: "b.local:1"}}
	calls := 0
	out, err := Select(context.Background(), hints, r, func(_ context.Context, endpoint string) (Observation, error) {
		_, o := fixture()
		if endpoint == "a.local:1" {
			o.Peer.ExpiresAt = selectionNow.Add(2 * time.Second)
		} else {
			o.Peer.Principal, o.Capabilities.Node = strings.Repeat("2", 32), strings.Repeat("2", 32)
		}
		return o, nil
	}, func() time.Time { now := selectionNow.Add(time.Duration(calls) * time.Second); calls++; return now })
	if err == nil || len(out.Selected) != 0 || out.Decisions[0].Reason != "wrong_identity" {
		t.Fatal(out, err)
	}
}

func TestSelectionUsesLiveMultiAddressObservation(t *testing.T) {
	for _, expireCredential := range []bool{false, true} {
		r, _ := fixture()
		now := selectionNow
		hints := []discovery.Hint{{Service: discovery.Node, Endpoint: "a.local:1"}, {Service: discovery.Node, Endpoint: "a.local:2"}, {Service: discovery.Node, Endpoint: "b.local:1"}}
		out, err := Select(context.Background(), hints, r, func(_ context.Context, endpoint string) (Observation, error) {
			_, o := fixture()
			switch endpoint {
			case "a.local:1":
				if expireCredential {
					o.Peer.ExpiresAt = selectionNow.Add(time.Second)
				} else {
					o.Capabilities.ObservedAt -= 29
					o.Capabilities.ProbeObservedAt -= 29
				}
			case "a.local:2":
				now = selectionNow.Add(time.Second)
				o.Capabilities.ObservedAt = now.Unix()
				o.Capabilities.ProbeObservedAt = now.Unix()
			case "b.local:1":
				now = selectionNow.Add(2 * time.Second)
				o.Peer.Principal, o.Capabilities.Node = strings.Repeat("2", 32), strings.Repeat("2", 32)
			}
			return o, nil
		}, func() time.Time { return now })
		if err != nil || len(out.Selected) != 2 || out.Selected[0].Endpoint != "a.local:2" || out.Decisions[1].Reason != "selected" {
			t.Fatal(expireCredential, out, err)
		}
	}
}
