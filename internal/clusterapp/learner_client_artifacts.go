package clusterapp

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"

	"github.com/mrothroc/mixlab/artifact"
	"github.com/mrothroc/mixlab/coordinator/artifactapi"
	private "github.com/mrothroc/mixlab/diloco/learner"
	"github.com/mrothroc/mixlab/diloco/run"
)

// Read streams into a caller-owned staging destination. Publish only on success.
func (c *LearnerClient) Read(ctx context.Context, a run.Assignment, b run.Blob, dst io.Writer) error {
	if dst == nil {
		return fmt.Errorf("artifact destination required")
	}
	if err := a.ValidateFor(c.options.Plan, c.options.Commit); err != nil {
		return err
	}
	return c.transfer(ctx, private.ArtifactRequest{Assignment: a, Blob: b, Operation: "read"}, dst, nil)
}
func (c *LearnerClient) Write(ctx context.Context, a run.Assignment, b run.Blob, src io.Reader) error {
	if src == nil {
		return fmt.Errorf("artifact source required")
	}
	if err := a.ValidateFor(c.options.Plan, c.options.Commit); err != nil {
		return err
	}
	return c.transfer(ctx, private.ArtifactRequest{Assignment: a, Blob: b, Operation: "create"}, nil, src)
}
func (c *LearnerClient) ReadEvaluation(ctx context.Context, a run.EvaluationAssignment, dst io.Writer) error {
	if dst == nil {
		return fmt.Errorf("evaluation destination required")
	}
	return c.transfer(ctx, private.ArtifactRequest{Evaluation: &a, Blob: a.Model, Operation: "read"}, dst, nil)
}
func (c *LearnerClient) transfer(ctx context.Context, request private.ArtifactRequest, dst io.Writer, src io.Reader) error {
	b := request.Blob
	q := c.request("artifact")
	op := "read"
	route := artifactapi.ReadRoute
	if src != nil {
		op = "create"
		route = artifactapi.CreateRoute
	}
	q.Artifact = &request
	out, err := c.call(ctx, q)
	if err != nil {
		return err
	}
	g := out.Grant.Body
	// Stable grant + use IDs make a lost response consume no additional budget.
	use := run.GrantUse{Format: "mixlab_artifact_use_v1", GrantHash: g.SHA256, GrantID: g.ID, Operation: op, UploadID: g.UploadID, IdempotencyID: g.ID}
	var envelope bytes.Buffer
	if err = artifactapi.WriteRequest(&envelope, artifactapi.Request{Grant: *out.Grant, Use: use}); err != nil {
		return err
	}
	var body io.Reader = &envelope
	if src != nil {
		body = io.MultiReader(&envelope, io.LimitReader(src, int64(b.Ref.Bytes)+1))
	}
	h, p, err := c.httpClient(true)
	if err != nil {
		return err
	}
	defer h.CloseIdleConnections()
	r, err := http.NewRequestWithContext(ctx, http.MethodPost, c.options.Endpoint+route, body)
	if err != nil {
		return err
	}
	r.Header.Set("Content-Type", "application/octet-stream")
	resp, err := h.Do(r)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	want := http.StatusOK
	if src != nil {
		want = http.StatusNoContent
	}
	if resp.StatusCode != want || resp.TLS == nil {
		return fmt.Errorf("artifact transfer rejected (HTTP %d)", resp.StatusCode)
	}
	if _, err = p.Authenticate(*resp.TLS); err != nil {
		return err
	}
	if dst != nil {
		if err = artifact.Copy(ctx, dst, resp.Body, b.Ref); err != nil {
			return err
		}
	}
	if _, err = p.Authenticate(*resp.TLS); err != nil {
		return err
	}
	return ctx.Err()
}
