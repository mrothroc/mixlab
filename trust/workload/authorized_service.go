package workload

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/trust"
)

// Stored separately from Grant so neither renewed signatures nor acceptance
// receipts alter the immutable issuance identity. Omitted on legacy entries.
type authorizationEvidence struct {
	Node       RequestAuthorization `json:"node"`
	Controller trust.SignedProof    `json:"controller"`
}

func (e authorizationEvidence) grant(g Grant) AuthorizedGrant {
	return AuthorizedGrant{Version: AuthorizedGrantVersion, Grant: g, NodeAuthorization: e.Node, ControllerAuthorization: e.Controller}
}

// IssueAuthorized accepts fresh authorization for an exact frozen grant. An
// existing entry always retains its original authorization receipts and result;
// new supplements cannot erase compromised history or create a second issuance.
func (s *Service) IssueAuthorized(ctx context.Context, callerChain [][]byte, g AuthorizedGrant, view trust.VerifiedSnapshot, now time.Time) (Result, error) {
	if err := g.Validate(); err != nil {
		return Result{}, err
	}
	b, err := json.Marshal(g)
	if err != nil {
		return Result{}, err
	}
	var frozen AuthorizedGrant
	if err := json.Unmarshal(b, &frozen); err != nil {
		return Result{}, err
	}
	auth := &authorizationEvidence{Node: frozen.NodeAuthorization, Controller: frozen.ControllerAuthorization}
	return s.issue(ctx, callerChain, frozen.Grant, auth, view, now)
}

func (e entry) acceptanceProofs() (trust.SignedProof, trust.SignRequest, trust.SignedProof, trust.SignRequest, error) {
	np, gp := e.Grant.Request.Proof, e.Grant.Proof
	nq, err := e.Grant.Request.Request.SigningRequest()
	if err != nil {
		return np, nq, gp, trust.SignRequest{}, err
	}
	gq, err := e.Grant.Request.GrantRequest()
	if err != nil || e.Authorization == nil {
		return np, nq, gp, gq, err
	}
	if err := e.Authorization.grant(e.Grant).Validate(); err != nil {
		return np, nq, gp, gq, err
	}
	np, gp = e.Authorization.Node.Proof, e.Authorization.Controller
	nq, err = e.Grant.Request.ReauthorizationRequest()
	if err != nil {
		return np, nq, gp, gq, err
	}
	gq, err = e.Grant.ReauthorizationRequest()
	return np, nq, gp, gq, err
}

func (e entry) validateAcceptance() error {
	np, _, gp, _, err := e.acceptanceProofs()
	if err != nil {
		return err
	}
	if e.NodeCommit.Digest != digest(np) || e.GrantCommit.Digest != digest(gp) || e.NodeCommit.Generation < np.Evidence.Snapshot.Payload.Generation || e.GrantCommit.Generation < gp.Evidence.Snapshot.Payload.Generation {
		return fmt.Errorf("workload receipt differs from accepted proof")
	}
	return nil
}

func (e entry) verifyHistory(a trust.Anchor, view trust.VerifiedSnapshot, now time.Time) error {
	np, nq, gp, gq, err := e.acceptanceProofs()
	if err != nil {
		return err
	}
	if err := trust.VerifyHistoricalProof(a, view, np, nq, commitJournal{e}, now); err != nil {
		return err
	}
	return trust.VerifyHistoricalProof(a, view, gp, gq, commitJournal{e}, now)
}

func (e entry) verifyReplacement(a trust.Anchor, view trust.VerifiedSnapshot, incoming *authorizationEvidence, now time.Time) error {
	// Exact saved proofs were just verified against the owner journal. Only
	// replacements require new acceptance, without overwriting that journal.
	nq, err := e.Grant.Request.ReauthorizationRequest()
	if err != nil {
		return err
	}
	if digest(incoming.Node.Proof) != e.NodeCommit.Digest {
		if _, err := trust.VerifyProof(a, view, incoming.Node.Proof, nq, now); err != nil {
			return err
		}
	}
	gq, err := e.Grant.ReauthorizationRequest()
	if err != nil {
		return err
	}
	if digest(incoming.Controller) != e.GrantCommit.Digest {
		_, err = trust.VerifyProof(a, view, incoming.Controller, gq, now)
	}
	return err
}
