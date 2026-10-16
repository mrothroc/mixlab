package clusterapp

import (
	"context"
	"fmt"
	"net/http"

	"github.com/mrothroc/mixlab/diloco/authenticated"
)

func (c *NodeClient) PrepareLearner(ctx context.Context, q NodeLearnerPrepareRequest) (NodeLearnerPrepared, error) {
	var out NodeLearnerPrepared
	m := q.Signed.Manifest
	if m.Slot.Allocation.NodeID != c.node || !nodeRouteID(m.Slot.Allocation.JobID) {
		return out, fmt.Errorf("learner target mismatch")
	}
	err := c.request(ctx, http.MethodPut, "/v1/agent/learners/"+m.Slot.Allocation.JobID+"/prepare", q, &out)
	if err != nil {
		return out, err
	}
	if err = out.Registration.ValidateFor(m.Plan); err != nil {
		return out, err
	}
	req, err := m.SigningRequest()
	if err != nil {
		return out, err
	}
	if out.Registration.Slot != m.Slot || out.Registration.JobManifestHash != req.Digest || out.Registration.AttemptID != m.Attempt || out.Registration.WorkloadGeneration != m.WorkloadGeneration {
		return out, fmt.Errorf("learner registration substituted identity")
	}
	scope, err := m.WorkloadScope(out.Request.Request.Scope.Workload)
	if err != nil || out.Request.Request.Scope != scope {
		return out, fmt.Errorf("learner workload substituted scope")
	}
	_, err = out.Request.GrantRequest()
	return out, err
}
func (c *NodeClient) learnerCommand(ctx context.Context, method, id, operation string, q any) error {
	if !nodeRouteID(id) {
		return fmt.Errorf("canonical learner job ID required")
	}
	var out NodeLearnerAck
	if err := c.request(ctx, method, "/v1/agent/learners/"+id+"/"+operation, q, &out); err != nil {
		return err
	}
	if out.Job != id {
		return fmt.Errorf("learner response job mismatch")
	}
	return nil
}
func (c *NodeClient) CredentialLearner(ctx context.Context, id string, q NodeLearnerCredentialRequest) error {
	return c.learnerCommand(ctx, http.MethodPut, id, "credential", q)
}
func (c *NodeClient) StartLearner(ctx context.Context, id string, q NodeLearnerStartRequest) error {
	return c.learnerCommand(ctx, http.MethodPost, id, "start", q)
}
func (c *NodeClient) FinalizeLearner(ctx context.Context, id string, q authenticated.Commit) error {
	return c.learnerCommand(ctx, http.MethodPut, id, "finalize", q)
}
