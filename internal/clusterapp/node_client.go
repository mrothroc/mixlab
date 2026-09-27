package clusterapp

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"time"

	"github.com/mrothroc/mixlab/discovery"
	"github.com/mrothroc/mixlab/internal/strictjson"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/transport/managedtls"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/principal"
)

// NodeClient binds every operation to the selected node's cryptographic
// identity, even if DNS/discovery changes. It holds no retry or saga state.
type NodeClient struct {
	endpoint, node string
	policy         func() (*managedtls.Policy, error)
}

func NewNodeClient(p *principal.Store, endpoint, node string, clock func() time.Time) (*NodeClient, error) {
	if p == nil || clock == nil {
		return nil, fmt.Errorf("current controller principal required")
	}
	return NewNodeClientWithPolicy(endpoint, node, func() (*managedtls.Policy, error) {
		s, _, err := p.Active(clock())
		if err != nil {
			return nil, err
		}
		if s.Role != trust.Controller {
			return nil, fmt.Errorf("controller principal required")
		}
		return managedPrincipalPolicy(p, trust.Node, clock)
	})
}

// NewNodeClientWithPolicy accepts an owning-context current-trust policy port;
// the client additionally pins the exact selected node before any request.
func NewNodeClientWithPolicy(endpoint, node string, policy func() (*managedtls.Policy, error)) (*NodeClient, error) {
	if !nodeRouteID(node) || policy == nil {
		return nil, fmt.Errorf("selected node identity and current TLS policy required")
	}
	if _, err := (discovery.Explicit{Addresses: map[discovery.Service][]string{discovery.Node: {endpoint}}}).Browse(context.Background(), discovery.Node); err != nil {
		return nil, err
	}
	return &NodeClient{endpoint, node, policy}, nil
}

func (c *NodeClient) request(ctx context.Context, method, path string, body, out any) error {
	p, err := c.policy()
	if err != nil {
		return err
	}
	p, err = p.RequirePeer(trust.Node, c.node)
	if err != nil {
		return err
	}
	client, err := p.HTTPClient(nodeOperationTimeout(method, path) + 5*time.Second)
	if err != nil {
		return err
	}
	defer client.CloseIdleConnections()
	var encoded []byte
	if body != nil {
		encoded, err = json.Marshal(body)
		if err != nil {
			return err
		}
	}
	if len(encoded) > 2<<20 {
		return fmt.Errorf("node request too large")
	}
	q, err := http.NewRequestWithContext(ctx, method, (&url.URL{Scheme: "https", Host: c.endpoint, Path: path}).String(), bytes.NewReader(encoded))
	if err != nil {
		return err
	}
	if body != nil {
		q.Header.Set("Content-Type", "application/json")
	}
	r, err := client.Do(q)
	if err != nil {
		return err
	}
	defer func() { _ = r.Body.Close() }()
	if r.StatusCode != http.StatusOK || r.TLS == nil {
		return fmt.Errorf("node operation rejected (HTTP %d)", r.StatusCode)
	}
	b, err := io.ReadAll(io.LimitReader(r.Body, (2<<20)+1))
	if err != nil {
		return err
	}
	if len(b) > 2<<20 {
		return fmt.Errorf("node response too large")
	}
	if _, err := p.Authenticate(*r.TLS); err != nil {
		return err
	}
	if err := strictjson.Validate(b, 32); err != nil {
		return err
	}
	d := json.NewDecoder(bytes.NewReader(b))
	d.DisallowUnknownFields()
	if err := d.Decode(out); err != nil {
		return err
	}
	return ctx.Err()
}

func (c *NodeClient) Reserve(ctx context.Context, q nodeagent.Reserve) (nodeagent.Lease, error) {
	var out nodeagent.Lease
	err := c.request(ctx, http.MethodPost, "/v1/agent/leases", q, &out)
	if err == nil && (out.Node != c.node || out.Run != q.Run || !nodeRouteID(out.ID) || out.Version == 0 || out.CapabilityGeneration != q.CapabilityGeneration) {
		err = fmt.Errorf("reservation response binding mismatch")
	}
	return out, err
}

func (c *NodeClient) AbortReservation(ctx context.Context, q nodeagent.Reserve) (nodeagent.ReservationAbort, error) {
	var out nodeagent.ReservationAbort
	if !nodeRouteID(q.IdempotencyKey) {
		return out, fmt.Errorf("invalid reservation ID")
	}
	err := c.request(ctx, http.MethodPost, "/v1/agent/reservations/"+q.IdempotencyKey+"/abort", q, &out)
	if err == nil && out.Node != c.node {
		err = fmt.Errorf("reservation abort target mismatch")
	}
	return out, err
}
func (c *NodeClient) LeaseStatus(ctx context.Context, id string) (nodeagent.Lease, error) {
	var out nodeagent.Lease
	if !nodeRouteID(id) {
		return out, fmt.Errorf("invalid lease ID")
	}
	err := c.request(ctx, http.MethodGet, "/v1/agent/leases/"+id, nil, &out)
	if err == nil && (out.Node != c.node || out.ID != id || out.Version == 0) {
		err = fmt.Errorf("lease response binding mismatch")
	}
	return out, err
}
func (c *NodeClient) leaseCommand(ctx context.Context, q nodeagent.LeaseCommand, renew bool) (nodeagent.Lease, error) {
	var out nodeagent.Lease
	if !nodeRouteID(q.Lease) {
		return out, fmt.Errorf("invalid lease ID")
	}
	method, path := http.MethodDelete, "/v1/agent/leases/"+q.Lease
	if renew {
		method, path = http.MethodPut, path+"/renew"
	}
	err := c.request(ctx, method, path, q, &out)
	if err == nil && (out.Node != c.node || out.ID != q.Lease || out.Version < q.ExpectedVersion) {
		err = fmt.Errorf("lease command response binding mismatch")
	}
	return out, err
}
func (c *NodeClient) Release(ctx context.Context, q nodeagent.LeaseCommand) (nodeagent.Lease, error) {
	return c.leaseCommand(ctx, q, false)
}
func (c *NodeClient) Renew(ctx context.Context, q nodeagent.LeaseCommand) (nodeagent.Lease, error) {
	return c.leaseCommand(ctx, q, true)
}
func (c *NodeClient) Prepare(ctx context.Context, q NodePrepareRequest) (nodeagent.PreparedWorkload, error) {
	var out nodeagent.PreparedWorkload
	if q.Signed.Manifest.Node != c.node || !nodeRouteID(q.Signed.Manifest.Job) {
		return out, fmt.Errorf("prepare target mismatch")
	}
	err := c.request(ctx, http.MethodPut, "/v1/agent/jobs/"+q.Signed.Manifest.Job+"/prepare", q, &out)
	return out, err
}
func (c *NodeClient) Transport(ctx context.Context, job string, q NodeTransportRequest) (nodeagent.Job, error) {
	return c.jobRequest(ctx, http.MethodPut, job, "/transport", q)
}
func (c *NodeClient) Start(ctx context.Context, q nodeagent.StartCommand) (nodeagent.Job, error) {
	return c.jobRequest(ctx, http.MethodPost, q.Job, "/start", q)
}
func (c *NodeClient) Cancel(ctx context.Context, job string, q NodeCancelRequest) (nodeagent.Job, error) {
	return c.jobRequest(ctx, http.MethodPost, job, "/cancel", q)
}
func (c *NodeClient) JobStatus(ctx context.Context, job string) (nodeagent.Job, error) {
	return c.jobRequest(ctx, http.MethodGet, job, "", nil)
}
func (c *NodeClient) jobRequest(ctx context.Context, method, job, suffix string, q any) (nodeagent.Job, error) {
	var out nodeagent.Job
	if !nodeRouteID(job) {
		return out, fmt.Errorf("invalid job ID")
	}
	err := c.request(ctx, method, "/v1/agent/jobs/"+job+suffix, q, &out)
	if err == nil && (out.ID != job || out.Version == 0) {
		err = fmt.Errorf("job response binding mismatch")
	}
	return out, err
}
