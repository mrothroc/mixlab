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

	"github.com/mrothroc/mixlab/admission"
	"github.com/mrothroc/mixlab/artifact"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/internal/strictjson"
	"github.com/mrothroc/mixlab/transport/managedtls"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/principal"
)

type DiLoCoClient struct {
	endpoint, coordinator string
	policy                func() (*managedtls.Policy, error)
}

func NewDiLoCoClient(p *principal.Store, endpoint, id string, clock func() time.Time) (*DiLoCoClient, error) {
	if p == nil || clock == nil {
		return nil, fmt.Errorf("current controller required")
	}
	return NewDiLoCoClientWithPolicy(endpoint, id, func() (*managedtls.Policy, error) {
		s, _, err := p.Active(clock())
		if err != nil {
			return nil, err
		}
		if s.Role != trust.Controller {
			return nil, fmt.Errorf("controller required")
		}
		return managedPrincipalPolicy(p, trust.Coordinator, clock)
	})
}
func NewDiLoCoClientWithPolicy(endpoint, id string, policy func() (*managedtls.Policy, error)) (*DiLoCoClient, error) {
	u, err := url.Parse(endpoint)
	if err != nil || u.Scheme != "https" || u.Hostname() == "" || u.User != nil || u.Path != "" || u.RawQuery != "" || u.Fragment != "" || u.Opaque != "" || !nodeRouteID(id) || policy == nil {
		return nil, fmt.Errorf("HTTPS coordinator origin, exact principal ID and TLS policy required")
	}
	return &DiLoCoClient{endpoint, id, policy}, nil
}
func (c *DiLoCoClient) request(ctx context.Context, method, path string, body io.Reader, length int64, digest string, out any) error {
	p, err := c.policy()
	if err != nil {
		return err
	}
	p, err = p.RequirePeer(trust.Coordinator, c.coordinator)
	if err != nil {
		return err
	}
	h, err := p.HTTPClient(time.Minute)
	if err != nil {
		return err
	}
	defer h.CloseIdleConnections()
	q, err := http.NewRequestWithContext(ctx, method, c.endpoint+path, body)
	if err != nil {
		return err
	}
	q.ContentLength = length
	if digest != "" {
		q.Header.Set("X-Mixlab-SHA256", digest)
		q.Header.Set("Content-Type", "application/octet-stream")
	} else if body != nil {
		q.Header.Set("Content-Type", "application/json")
	}
	r, err := h.Do(q)
	if err != nil {
		return fmt.Errorf("coordinator request failed (route/TLS/trust observation, not proof of Local Network Privacy denial): %w", err)
	}
	defer func() { _ = r.Body.Close() }()
	if r.TLS == nil || r.StatusCode != http.StatusOK {
		return fmt.Errorf("coordinator returned HTTP %d; inspect coordinator status and service diagnostics", r.StatusCode)
	}
	b, err := io.ReadAll(io.LimitReader(r.Body, (8<<20)+1))
	if err != nil {
		return err
	}
	if len(b) > 8<<20 {
		return fmt.Errorf("coordinator response exceeds bound")
	}
	if _, err := p.Authenticate(*r.TLS); err != nil {
		return err
	}
	if err := strictjson.Validate(b, 64); err != nil {
		return err
	}
	d := json.NewDecoder(bytes.NewReader(b))
	d.DisallowUnknownFields()
	if err := d.Decode(out); err != nil {
		return err
	}
	return ctx.Err()
}
func (c *DiLoCoClient) Info(ctx context.Context) (DiLoCoInfo, error) {
	var out DiLoCoInfo
	err := c.request(ctx, http.MethodGet, DiLoCoInfoRoute, nil, 0, "", &out)
	if err == nil {
		err = validateDiLoCoInfo(out, c.coordinator)
	}
	if err == nil && out.Endpoint != c.endpoint {
		err = fmt.Errorf("coordinator origin differs from installed service")
	}
	return out, err
}
func (c *DiLoCoClient) Submit(ctx context.Context, q admission.Submission) (DiLoCoSubmissionStatus, error) {
	var out DiLoCoSubmissionStatus
	if err := q.Validate(); err != nil {
		return out, err
	}
	b, err := json.Marshal(q)
	if err != nil {
		return out, err
	}
	if len(b) > 8<<20 {
		return out, fmt.Errorf("submission exceeds bound")
	}
	err = c.request(ctx, http.MethodPost, DiLoCoSubmissionRoute, bytes.NewReader(b), int64(len(b)), "", &out)
	if err == nil && (out.RunID != q.Launch.Run || out.Admission != "ACCEPTED") {
		err = fmt.Errorf("submission acknowledgement mismatch")
	}
	return out, err
}
func (c *DiLoCoClient) Status(ctx context.Context, id string, cancel bool) (DiLoCoSubmissionStatus, error) {
	var out DiLoCoSubmissionStatus
	if !nodeRouteID(id) {
		return out, fmt.Errorf("canonical run ID required")
	}
	path, method := DiLoCoSubmissionRoute+"/"+id, http.MethodGet
	if cancel {
		path += "/cancel"
		method = http.MethodPost
	}
	err := c.request(ctx, method, path, nil, 0, "", &out)
	if err == nil && out.RunID != id {
		err = fmt.Errorf("run status identity mismatch")
	}
	return out, err
}
func (c *DiLoCoClient) UploadInitialModel(ctx context.Context, r artifact.Ref, reader io.Reader) (run.Blob, error) {
	var out run.Blob
	if err := r.Validate(); err != nil {
		return out, err
	}
	err := c.request(ctx, http.MethodPut, DiLoCoInitialModelRoute, reader, int64(r.Bytes), r.SHA256, &out)
	if err == nil {
		err = out.Validate()
	}
	if err == nil && out.Kind != "model_f32" {
		err = fmt.Errorf("initial model kind changed")
	}
	return out, err
}
