package clusterapp

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"time"

	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/principal"
	"github.com/mrothroc/mixlab/trust/workload"
)

func IssueRemoteWorkload(ctx context.Context, p *principal.Store, endpoint string, grant workload.Grant, clock func() time.Time) (workload.Result, error) {
	if err := RefreshRemoteTrust(ctx, p, endpoint, clock); err != nil {
		return workload.Result{}, err
	}
	s, _, err := p.Active(clock())
	if err != nil {
		return workload.Result{}, err
	}
	if s.Role != trust.Controller || grant.Request.Request.Scope.Controller != s.Principal {
		return workload.Result{}, fmt.Errorf("owning controller required")
	}
	policy, err := managedPrincipalPolicy(p, trust.Authority, clock)
	if err != nil {
		return workload.Result{}, err
	}
	c, err := policy.HTTPClient(15 * time.Second)
	if err != nil {
		return workload.Result{}, err
	}
	defer c.CloseIdleConnections()
	base, err := url.Parse(endpoint)
	if err != nil {
		return workload.Result{}, err
	}
	var out workload.Result
	if err := enrollmentJSON(ctx, c, http.MethodPost, base.ResolveReference(&url.URL{Path: workloadIssueRoute}).String(), grant, &out); err != nil {
		return workload.Result{}, err
	}
	a, err := trust.PinRoot(s.Root, s.Fingerprint, clock())
	if err != nil {
		return workload.Result{}, err
	}
	if err := workload.ValidateResult(a, grant, out, clock()); err != nil {
		return workload.Result{}, err
	}
	if err := refreshPrincipal(ctx, p, out.Snapshot, clock()); err != nil {
		return workload.Result{}, err
	}
	return out, nil
}
