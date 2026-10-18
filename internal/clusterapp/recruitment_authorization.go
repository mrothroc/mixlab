package clusterapp

import (
	"context"
	"fmt"

	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/workload"
)

// Supplemental authorization is confined to initial learner admission. The
// ordinary grant path and coordinator-owned recovery retain their own proofs.
func (o RecruitmentOptions) verifyRequestAuthorization(ctx context.Context, r workload.SignedRequest, authorization workload.RequestAuthorization) error {
	if o.Principal == nil || o.Clock == nil {
		return fmt.Errorf("controller and clock required")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	a, _, err := controllerTrust(o.Principal, o.Clock())
	if err != nil {
		return err
	}
	s, _, err := o.Principal.Active(o.Clock())
	if err != nil {
		return err
	}
	scope := r.Request.Scope
	if scope.Profile != workload.LearnerVersion || scope.GrantRole() != trust.Controller || scope.Controller != s.Principal || scope.Cluster != s.Cluster {
		return fmt.Errorf("supplemental grant targets another controller or admission path")
	}
	_, err = trust.CurrentForEvidence(ctx, a, authorization.Proof.Evidence.Snapshot,
		func(context.Context) (trust.VerifiedSnapshot, error) {
			_, view, err := controllerTrust(o.Principal, o.Clock())
			return view, err
		}, func(ctx context.Context) error { return refreshPrincipalTrust(ctx, o.Principal, o.Clock) }, o.Clock,
		func(view trust.VerifiedSnapshot) error {
			return workload.VerifyRequestAuthorization(a, view, r, authorization, o.Clock())
		})
	return err
}

func (o RecruitmentOptions) grantWithAuthorization(ctx context.Context, r workload.SignedRequest, authorization workload.RequestAuthorization) (workload.Grant, error) {
	if err := o.verifyRequestAuthorization(ctx, r, authorization); err != nil {
		return workload.Grant{}, err
	}
	request, err := r.GrantRequest()
	if err != nil {
		return workload.Grant{}, err
	}
	proof, err := o.sign(ctx, request)
	return workload.Grant{Request: r, Proof: proof}, err
}

// authorizeGrant must receive the immutable grant returned by Inbox.WorkloadGrant,
// including on replay. Supplemental proofs never replace that persisted grant.
func (o RecruitmentOptions) authorizeGrant(ctx context.Context, persisted workload.Grant, authorization workload.RequestAuthorization) (workload.AuthorizedGrant, error) {
	var zero workload.AuthorizedGrant
	if err := o.verifyRequestAuthorization(ctx, persisted.Request, authorization); err != nil {
		return zero, err
	}
	request, err := persisted.ReauthorizationRequest()
	if err != nil {
		return zero, err
	}
	proof, err := o.sign(ctx, request)
	if err != nil {
		return zero, err
	}
	out := workload.AuthorizedGrant{Version: workload.AuthorizedGrantVersion, Grant: persisted, NodeAuthorization: authorization, ControllerAuthorization: proof}
	a, view, err := controllerTrust(o.Principal, o.Clock())
	if err != nil {
		return zero, err
	}
	if err := workload.VerifyAuthorizedGrant(a, view, out, o.Clock()); err != nil {
		return zero, err
	}
	if err := ctx.Err(); err != nil {
		return zero, err
	}
	return out, nil
}
