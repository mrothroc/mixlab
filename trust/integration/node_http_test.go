package integration

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/trust"
)

type nodeHTTPJobs struct{ prepares atomic.Int32 }

func (j *nodeHTTPJobs) Prepare(_ context.Context, _ trust.AuthenticatedPrincipal, q clusterapp.NodePrepareRequest) (nodeagent.PreparedWorkload, error) {
	j.prepares.Add(1)
	return nodeagent.PreparedWorkload{Job: nodeagent.Job{ID: q.Signed.Manifest.Job}}, nil
}
func (*nodeHTTPJobs) Transport(context.Context, trust.AuthenticatedPrincipal, string, clusterapp.NodeTransportRequest) (nodeagent.Job, error) {
	return nodeagent.Job{}, errors.New("private secret path /do/not/leak")
}
func (*nodeHTTPJobs) Start(context.Context, trust.AuthenticatedPrincipal, nodeagent.StartCommand) (nodeagent.Job, error) {
	return nodeagent.Job{}, errors.New("private secret path /do/not/leak")
}

func TestNodeManagementHTTPClosedSchemasAndLeases(t *testing.T) {
	f := newTLSFixture(t)
	store, policy, clientPolicy, _ := capabilityNode(t, f)
	jobs := &nodeHTTPJobs{}
	h, err := clusterapp.NodeManagementHandler(store, policy, f.clock, jobs)
	check(t, err)
	server := serveCapabilities(t, policy, h)
	client, err := clientPolicy.HTTPClient(5 * time.Second)
	check(t, err)
	defer client.CloseIdleConnections()
	request := func(method, path string, value any, extra bool) (int, []byte) {
		t.Helper()
		b, err := json.Marshal(value)
		check(t, err)
		if extra {
			b = append(b[:len(b)-1], []byte(`,"credential_envelope":{"ciphertext":"not-supported"}}`)...)
		}
		r, err := http.NewRequest(method, server.URL+path, bytes.NewReader(b))
		check(t, err)
		r.Header.Set("Content-Type", "application/json")
		resp, err := client.Do(r)
		check(t, err)
		defer func() { _ = resp.Body.Close() }()
		out, err := io.ReadAll(resp.Body)
		check(t, err)
		return resp.StatusCode, out
	}
	q := nodeagent.Reserve{IdempotencyKey: id(t), ExpectedNodeVersion: 1, CapabilityGeneration: 1, Run: id(t), TTLSeconds: 300}
	if code, _ := request(http.MethodPost, "/v1/agent/leases", q, true); code != http.StatusBadRequest {
		t.Fatal(code)
	}
	available, err := store.Availability(f.now)
	check(t, err)
	if !available.Available {
		t.Fatal("unknown envelope caused side effect")
	}
	code, b := request(http.MethodPost, "/v1/agent/leases", q, false)
	if code != http.StatusOK {
		t.Fatal(code, string(b))
	}
	var lease nodeagent.Lease
	check(t, json.Unmarshal(b, &lease))
	again, b := request(http.MethodPost, "/v1/agent/leases", q, false)
	if again != http.StatusOK {
		t.Fatal(again)
	}
	var retry nodeagent.Lease
	check(t, json.Unmarshal(b, &retry))
	if retry != lease {
		t.Fatal("HTTP reserve retry changed lease")
	}
	job := signedJobFixture(t, f, lease.Node, lease)
	p := clusterapp.NodePrepareRequest{ExpectedLeaseVersion: lease.Version, Signed: job}
	path := "/v1/agent/jobs/" + job.Manifest.Job + "/prepare"
	if code, _ := request(http.MethodPut, path, p, true); code != http.StatusBadRequest {
		t.Fatal("extra envelope accepted", code)
	}
	if jobs.prepares.Load() != 0 {
		t.Fatal("invalid prepare reached application")
	}
	if code, _ := request(http.MethodPut, path, p, false); code != http.StatusOK {
		t.Fatal(code)
	}
	if jobs.prepares.Load() != 1 {
		t.Fatal("valid prepare not dispatched")
	}
	if code, _ := request(http.MethodPut, "/v1/agent/jobs/"+id(t)+"/prepare", p, false); code != http.StatusBadRequest {
		t.Fatal("route substitution accepted", code)
	}
	change := nodeagent.LeaseCommand{IdempotencyKey: id(t), Lease: lease.ID, ExpectedVersion: lease.Version}
	if code, _ := request(http.MethodDelete, "/v1/agent/leases/"+id(t), change, false); code != http.StatusBadRequest {
		t.Fatal("lease route substitution accepted")
	}
	if code, _ := request(http.MethodDelete, "/v1/agent/leases/"+lease.ID, change, false); code != http.StatusOK {
		t.Fatal(code)
	}
	available, err = store.Availability(f.now)
	check(t, err)
	if available.Available || available.Lease.State != nodeagent.Releasing {
		t.Fatal("HTTP release bypassed cleanup")
	}
	if code, b := request(http.MethodPost, "/v1/agent/jobs/"+job.Manifest.Job+"/start", nodeagent.StartCommand{IdempotencyKey: id(t), Job: job.Manifest.Job, ExpectedVersion: 1}, false); code != http.StatusConflict || strings.Contains(string(b), "/do/not/leak") {
		t.Fatal(code, string(b))
	}
	f.mu.Lock()
	f.snapshot.Payload.Generation++
	f.snapshot.Payload.Revocations = []trust.Revocation{{Kind: "principal", ID: f.clientID, Mode: "compromise", Reason: "test", FirstGeneration: 2}}
	f.signSnapshot(t)
	f.mu.Unlock()
	// Either TLS or the per-request check must reject the revoked controller.
	b, err = json.Marshal(q)
	check(t, err)
	req, err := http.NewRequest(http.MethodPost, server.URL+"/v1/agent/leases", bytes.NewReader(b))
	check(t, err)
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
	if err == nil {
		defer func() { _ = resp.Body.Close() }()
		if resp.StatusCode == http.StatusOK {
			t.Fatal("revoked controller reused connection")
		}
	}
}
