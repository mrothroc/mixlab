package clusterapp

import (
	"context"
	"time"

	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/workload"
)

type nodeCallerCheckKey struct{}

func checkNodeCaller(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if check, ok := ctx.Value(nodeCallerCheckKey{}).(func() error); ok {
		return check()
	}
	return nil // In-process callers already own their authentication boundary.
}

func (a *NodeJobs) currentForEvidence(ctx context.Context, snapshot trust.SignedSnapshot, verify func(trust.VerifiedSnapshot) error) (trust.VerifiedSnapshot, error) {
	refresh := a.options.RefreshTrust
	if refresh != nil {
		refresh = func(ctx context.Context) error {
			ctx, cancel := context.WithTimeout(ctx, 15*time.Second)
			defer cancel()
			return a.options.RefreshTrust(ctx)
		}
	}
	v, err := trust.CurrentForEvidence(ctx, a.options.Anchor, snapshot, a.current, refresh, a.options.Clock, verify)
	if err != nil {
		return trust.VerifiedSnapshot{}, err
	}
	// The proof signer need not be the TLS caller (for example, a controller
	// forwarding a coordinator commit). Recheck that caller after catch-up.
	if err := checkNodeCaller(ctx); err != nil {
		return trust.VerifiedSnapshot{}, err
	}
	return v, nil
}

// Multi-evidence operations must validate every component before refreshing,
// then validate them together against the returned local authority view.
func nodeEvidenceSnapshot(proof, credential trust.SignedSnapshot) trust.SignedSnapshot {
	if credential.Payload.Generation > proof.Payload.Generation {
		return credential
	}
	return proof
}

func (a *NodeJobs) verifyCredential(v trust.VerifiedSnapshot, grant workload.Grant, credential workload.Result) error {
	if err := workload.ValidateResult(a.options.Anchor, grant, credential, a.options.Clock()); err != nil {
		return err
	}
	return trust.VerifyWorkload(a.options.Anchor, v, credential.Chain, credential.Binding, a.options.Clock())
}
