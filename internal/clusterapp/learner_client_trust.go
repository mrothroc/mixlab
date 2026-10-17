package clusterapp

import (
	"context"
	"crypto/tls"
	"fmt"

	"github.com/mrothroc/mixlab/transport/managedtls"
	"github.com/mrothroc/mixlab/trust"
)

// Grant catch-up can revoke either connection principal independently of the
// proof signer. Do not release a grant to the node-authenticated byte transport
// after learning that its owning workload or actual TLS responder was revoked.
func (c *LearnerClient) verifyResponsePrincipals(ctx context.Context, policy *managedtls.Policy, connection tls.ConnectionState) error {
	v, err := c.options.Trust.Current(ctx, c.options.Clock())
	if err != nil {
		return err
	}
	certificates := policy.ClientConfig().Certificates
	if len(certificates) != 1 {
		return fmt.Errorf("exact local learner credential required")
	}
	if err := trust.VerifyWorkload(c.options.Anchor, v, certificates[0].Certificate, c.options.Workload, c.options.Clock()); err != nil {
		return err
	}
	chain := make([][]byte, len(connection.PeerCertificates))
	for i, cert := range connection.PeerCertificates {
		chain[i] = cert.Raw
	}
	peer, err := trust.AuthenticatePrincipal(c.options.Anchor, v, chain, c.options.Clock())
	if err != nil {
		return err
	}
	if peer.Role != trust.Coordinator || peer.Principal != c.options.Plan.Coordinator.PrincipalID {
		return fmt.Errorf("learner response coordinator changed")
	}
	_, err = policy.Authenticate(connection)
	return err
}
