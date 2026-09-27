package integration

import (
	"context"
	"crypto/ecdh"
	"crypto/rand"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/discovery"
	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/recruitment"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/transport/managedtls"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/internal/certificates"
	"github.com/mrothroc/mixlab/workercontrol"
	"github.com/mrothroc/mixlab/workerprobe"
)

func capabilityNode(t *testing.T, f *tlsFixture) (*nodeagent.Store, *managedtls.Policy, *managedtls.Policy, recruitment.Requirements) {
	t.Helper()
	node := id(t)
	k := key(t)
	envelope, err := ecdh.X25519().GenerateKey(rand.Reader)
	check(t, err)
	der, err := certificates.IssueNode(f.a, f.issuer, f.issuerKey, node, k.Public(), envelope.PublicKey().Bytes(), f.now)
	check(t, err)
	if !slices.Contains(f.snapshot.Payload.EligibleRoles, trust.Node) {
		f.snapshot.Payload.EligibleRoles = append(f.snapshot.Payload.EligibleRoles, trust.Node)
		f.signSnapshot(t)
	}
	nodePolicy, err := managedtls.New(managedtls.Identity{Chain: [][]byte{der, f.issuer, f.a.DER()}, Key: k}, f.verify(trust.Controller, f.clientID), f.clock)
	check(t, err)
	clientPolicy, err := managedtls.New(f.client, func(chain [][]byte, now time.Time) (trust.AuthenticatedPrincipal, error) {
		f.mu.RLock()
		defer f.mu.RUnlock()
		return trust.AuthenticatePrincipal(f.a, f.view, chain, now)
	}, f.clock)
	check(t, err)
	dir, err := filepath.EvalSymlinks(t.TempDir())
	check(t, err)
	check(t, os.Chmod(dir, 0700))
	p, err := statehome.Resolve(statehome.Options{ExactDir: dir}, statehome.Context{Kind: statehome.Agent})
	check(t, err)
	s, err := nodeagent.Initialize(context.Background(), p, f.a.Cluster(), node, 1)
	check(t, err)
	hash := strings.Repeat("a", 64)
	limits := nodejob.Limits{RuntimeSeconds: 60, CPUSeconds: 120, MemoryBytes: 1 << 30, DiskBytes: 1 << 30, LogBytes: 1 << 20}
	probe := workerprobe.Report{Version: workerprobe.Version, BuildID: hash, BuildVersion: "test", WorkerProtocol: workercontrol.Version, OS: "darwin", Arch: "arm64", Available: true, MLXVersion: "0.32.1", MLXSupported: true, DeviceKind: "metal", DeviceName: "test", Backends: []string{"ring"}, DTypes: []string{"fp32"}, CustomOps: []string{"mixlab-ir-v1"}}
	check(t, s.InstallProfile(context.Background(), nodeagent.Profile{Version: nodeagent.ProfileVersion, Node: node, DisplayName: "test", Generation: 1, ProbeObservedAt: f.now.Unix(), Probe: probe, Limits: limits, Datasets: []nodeagent.LocalDataset{{Dataset: nodeagent.Dataset{Selector: "toy", ID: hash}, TrainPattern: "/private/local/train.bin"}}}))
	r := recruitment.Requirements{Cluster: f.a.Cluster(), Count: 2, BuildID: hash, MLXVersion: "0.32.1", DeviceKind: "metal", DType: "fp32", CustomOps: []string{"mixlab-ir-v1"}, DatasetSelector: "toy", DatasetID: hash, Limits: limits, ProbeMaxAge: time.Minute}
	return s, nodePolicy, clientPolicy, r
}

func serveCapabilities(t *testing.T, policy *managedtls.Policy, handler http.Handler) *httptest.Server {
	t.Helper()
	s := httptest.NewUnstartedServer(handler)
	s.TLS = policy.ServerConfig()
	s.Config.ErrorLog = log.New(io.Discard, "", 0)
	s.StartTLS()
	t.Cleanup(s.Close)
	return s
}

func TestNodeCapabilitiesTLSRecruitmentAndRevocation(t *testing.T) {
	f := newTLSFixture(t)
	var hints []discovery.Hint
	var clientPolicy *managedtls.Policy
	var requirements recruitment.Requirements
	for range 2 {
		store, policy, client, req := capabilityNode(t, f)
		clientPolicy, requirements = client, req
		h, err := clusterapp.NodeCapabilitiesHandler(store, policy, f.clock)
		check(t, err)
		s := serveCapabilities(t, policy, h)
		hints = append(hints, discovery.Hint{Service: discovery.Node, Endpoint: strings.TrimPrefix(s.URL, "https://"), Claims: discovery.Claims{Cluster: "false", Node: "false"}})
	}
	lookup := func(ctx context.Context, endpoint string) (recruitment.Observation, error) {
		return clusterapp.QueryNodeCapabilities(ctx, endpoint, clientPolicy)
	}
	selection, err := recruitment.Select(context.Background(), hints, requirements, lookup, f.clock)
	check(t, err)
	inventory, err := recruitment.Inventory(context.Background(), hints, requirements.Cluster, lookup, f.clock)
	check(t, err)
	if len(inventory) != 2 || inventory[0].Reason != "available" || inventory[1].Reason != "available" {
		t.Fatal(inventory)
	}
	if len(selection.Selected) != 2 || selection.Selected[0].Capabilities.Node >= selection.Selected[1].Capabilities.Node {
		t.Fatal(selection)
	}
	b, err := json.Marshal(selection)
	check(t, err)
	if strings.Contains(string(b), "/private/local") || strings.Contains(string(b), `"false"`) {
		t.Fatal("local path or forged discovery identity leaked into selected evidence")
	}
	f.mu.Lock()
	f.snapshot.Payload.Generation++
	f.snapshot.Payload.Revocations = []trust.Revocation{{Kind: "principal", ID: selection.Selected[0].Capabilities.Node, Mode: "compromise", Reason: "test", FirstGeneration: 2}}
	f.signSnapshot(t)
	f.mu.Unlock()
	after, err := recruitment.Select(context.Background(), hints, requirements, lookup, f.clock)
	if err == nil || len(after.Selected) != 0 {
		t.Fatal("revoked node recruited", err)
	}
}

func TestNodeCapabilityResponseBoundToTLSAndStrictJSON(t *testing.T) {
	for _, mode := range []string{"wrong-node", "unknown-field", "duplicate-field", "trailing", "oversize", "redirect", "stale-after-read"} {
		t.Run(mode, func(t *testing.T) {
			f := newTLSFixture(t)
			store, policy, client, _ := capabilityNode(t, f)
			actor, err := trust.AuthenticatePrincipal(f.a, f.view, f.client.Chain, f.now)
			check(t, err)
			c, err := store.Capabilities(context.Background(), actor, f.now)
			check(t, err)
			if mode == "wrong-node" {
				c.Node = id(t)
			}
			b, err := json.Marshal(c)
			check(t, err)
			switch mode {
			case "unknown-field":
				b = append([]byte(`{"secret":1,`), b[1:]...)
			case "duplicate-field":
				b = append([]byte(`{"version":"ignored",`), b[1:]...)
			case "trailing":
				b = append(b, []byte(` {}`)...)
			case "oversize":
				b = []byte(strings.Repeat(" ", 256<<10+1))
			}
			s := serveCapabilities(t, policy, policy.AuthenticateHTTP(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if mode == "redirect" {
					http.Redirect(w, r, "/somewhere", http.StatusTemporaryRedirect)
					return
				}
				if mode == "stale-after-read" {
					f.mu.Lock()
					f.now = f.now.Add(trust.SnapshotLifetime + time.Second)
					f.mu.Unlock()
				}
				_, _ = w.Write(b)
			})))
			_, err = clusterapp.QueryNodeCapabilities(context.Background(), strings.TrimPrefix(s.URL, "https://"), client)
			if err == nil {
				t.Fatal("accepted invalid capability response")
			}
		})
	}
}
