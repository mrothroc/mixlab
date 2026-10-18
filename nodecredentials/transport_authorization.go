package nodecredentials

import (
	"context"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/workload"
)

// AuthorizeRequest signs a fresh supplement to the exact durable request. It
// never creates or recovers a key, replaces the original proof, or rewrites the
// credential journal. Destruction shares transportLock with this operation.
func (c *Transport) AuthorizeRequest(ctx context.Context, current trust.VerifiedSnapshot, sign NodeSigner, now time.Time) (workload.RequestAuthorization, error) {
	var out workload.RequestAuthorization
	if sign == nil || now.Unix() < c.scope.Created || now.Unix() >= c.scope.AdmitUntil {
		return out, fmt.Errorf("live workload admission and node signer required")
	}
	ctx, cancel := context.WithTimeout(ctx, time.Unix(c.scope.AdmitUntil, 0).Sub(now))
	defer cancel()
	err := c.path.WithProcessLock(ctx, transportLock, func() error {
		_, r, err := c.load()
		if err != nil {
			return err
		}
		if r.Request == nil || (r.Stage != "requested" && r.Stage != "installed") {
			return fmt.Errorf("existing live transport request required")
		}
		key, err := c.signer()
		if err != nil {
			return err
		}
		public, err := r.Request.Request.PublicKey()
		if err != nil || !samePublicValue(public, key.Public()) {
			return fmt.Errorf("stored transport request differs from protected key")
		}
		q, err := r.Request.ReauthorizationRequest()
		if err != nil {
			return err
		}
		proof, err := sign(ctx, q)
		if err != nil {
			return err
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		out = workload.RequestAuthorization{Version: workload.RequestAuthorizationVersion, Proof: proof}
		if err := workload.VerifyRequestAuthorization(c.anchor, current, *r.Request, out, now); err != nil {
			return err
		}
		return ctx.Err()
	})
	if err != nil {
		return workload.RequestAuthorization{}, err
	}
	return out, nil
}
