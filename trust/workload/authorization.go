package workload

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/trust"
)

const RequestAuthorizationVersion = "mixlab_workload_request_authorization_v1"
const AuthorizedGrantVersion = "mixlab_workload_authorized_grant_v1"

type RequestAuthorization struct {
	Version string            `json:"version"`
	Proof   trust.SignedProof `json:"proof"`
}

// AuthorizedGrant supplements immutable admission evidence. Neither proof is
// included in GrantHash; refreshes cannot replace the grant or transport key.
type AuthorizedGrant struct {
	Version                 string               `json:"version"`
	Grant                   Grant                `json:"grant"`
	NodeAuthorization       RequestAuthorization `json:"node_authorization"`
	ControllerAuthorization trust.SignedProof    `json:"controller_authorization"`
}

func (r SignedRequest) ReauthorizationRequest() (trust.SignRequest, error) {
	if _, err := r.GrantRequest(); err != nil {
		return trust.SignRequest{}, err
	}
	if b, err := json.Marshal(r); err != nil || len(b) > trust.MaxTrustBytes+8192 {
		return trust.SignRequest{}, fmt.Errorf("oversized workload request")
	}
	s := r.Request.Scope
	return trust.SignRequest{Version: trust.ProofVersion, Purpose: trust.WorkloadRequestAuthorization, Digest: digest(r), Context: s.Job + "/" + s.Attempt, Audience: s.Controller}, nil
}

func (g Grant) ReauthorizationRequest() (trust.SignRequest, error) {
	q, err := g.Request.GrantRequest()
	if err != nil {
		return trust.SignRequest{}, err
	}
	s := g.Request.Request.Scope
	if g.Proof.Request != q || g.Proof.Evidence.Principal != s.Controller || g.Proof.Evidence.Role != s.GrantRole() {
		return trust.SignRequest{}, fmt.Errorf("original workload grant binding mismatch")
	}
	if b, err := json.Marshal(g); err != nil || len(b) > 2*trust.MaxTrustBytes+8192 {
		return trust.SignRequest{}, fmt.Errorf("oversized workload grant")
	}
	return trust.SignRequest{Version: trust.ProofVersion, Purpose: trust.WorkloadGrantAuthorization, Digest: digest(g), Context: s.Job + "/" + s.Attempt, Audience: s.Node}, nil
}

func validateRequestAuthorization(original SignedRequest, auth RequestAuthorization) error {
	q, err := original.ReauthorizationRequest()
	if err != nil {
		return err
	}
	s := original.Request.Scope
	if auth.Version != RequestAuthorizationVersion || auth.Proof.Request != q || auth.Proof.Evidence.Principal != s.Node || auth.Proof.Evidence.Role != trust.Node || auth.Proof.Evidence.Cluster != s.Cluster {
		return fmt.Errorf("workload request authorization binding mismatch")
	}
	return nil
}

// Validate checks structural bindings only, not cryptographic acceptance.
func (g AuthorizedGrant) Validate() error {
	if g.Version != AuthorizedGrantVersion {
		return fmt.Errorf("invalid authorized workload grant version")
	}
	if b, err := json.Marshal(g); err != nil || len(b) > 4*trust.MaxTrustBytes+16384 {
		return fmt.Errorf("oversized authorized workload grant")
	}
	if err := validateRequestAuthorization(g.Grant.Request, g.NodeAuthorization); err != nil {
		return err
	}
	q, err := g.Grant.ReauthorizationRequest()
	if err != nil {
		return err
	}
	s := g.Grant.Request.Request.Scope
	p := g.ControllerAuthorization
	if p.Request != q || p.Evidence.Principal != s.Controller || p.Evidence.Role != s.GrantRole() || p.Evidence.Cluster != s.Cluster {
		return fmt.Errorf("workload grant authorization binding mismatch")
	}
	return nil
}

func liveAuthorization(a trust.Anchor, s Scope, now time.Time) error {
	if s.Cluster != a.Cluster() || now.Unix() < s.Created || now.Unix() >= s.AdmitUntil {
		return fmt.Errorf("workload authorization cluster or admission deadline mismatch")
	}
	return nil
}

func VerifyRequestAuthorization(a trust.Anchor, view trust.VerifiedSnapshot, original SignedRequest, auth RequestAuthorization, now time.Time) error {
	if err := validateRequestAuthorization(original, auth); err != nil {
		return err
	}
	if err := liveAuthorization(a, original.Request.Scope, now); err != nil {
		return err
	}
	q, _ := original.ReauthorizationRequest()
	_, err := trust.VerifyProof(a, view, auth.Proof, q, now)
	return err
}

// VerifyAuthorizedGrant verifies fresh supplemental authorization, never a
// claim of historical acceptance. IssueAuthorized alone owns replay receipts.
func VerifyAuthorizedGrant(a trust.Anchor, view trust.VerifiedSnapshot, g AuthorizedGrant, now time.Time) error {
	_, _, err := verifyAuthorizedGrant(a, view, g, now)
	return err
}

func verifyAuthorizedGrant(a trust.Anchor, view trust.VerifiedSnapshot, g AuthorizedGrant, now time.Time) (trust.AcceptedProof, trust.AcceptedProof, error) {
	var zero trust.AcceptedProof
	if err := g.Validate(); err != nil {
		return zero, zero, err
	}
	if err := liveAuthorization(a, g.Grant.Request.Request.Scope, now); err != nil {
		return zero, zero, err
	}
	nq, _ := g.Grant.Request.ReauthorizationRequest()
	nc, err := trust.VerifyProof(a, view, g.NodeAuthorization.Proof, nq, now)
	if err != nil {
		return zero, zero, err
	}
	gq, _ := g.Grant.ReauthorizationRequest()
	gc, err := trust.VerifyProof(a, view, g.ControllerAuthorization, gq, now)
	return nc, gc, err
}
