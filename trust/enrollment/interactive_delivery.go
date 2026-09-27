package enrollment

import (
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/trust"
)

// The receipt attests to delivery of an immutable approval and exact issued
// chain under current trust. It does not replace or renew that approval, reopen
// an expired window, or authorize reattachment to a different TLS channel.
func interactiveDeliveryRequest(r InteractiveResult) trust.SignRequest {
	payload := struct {
		Evidence InteractiveEvidence `json:"evidence"`
		Approval Approval            `json:"approval"`
		Proof    trust.SignedProof   `json:"proof"`
		Commit   trust.AcceptedProof `json:"commit"`
		Chain    [][]byte            `json:"chain"`
	}{r.Evidence, r.Approval, r.Proof, r.Commit, r.Chain}
	return trust.SignRequest{Version: trust.ProofVersion, Purpose: trust.EnrollmentReceipt,
		Digest: digestText("mixlab_interactive_delivery_v1", payload), Context: r.Approval.RequestID, Audience: r.Approval.Audience}
}

func (s *Service) signInteractiveDelivery(r *InteractiveResult, v trust.VerifiedSnapshot, now time.Time) error {
	var err error
	r.Snapshot, err = snapshotValue(v)
	if err != nil {
		return err
	}
	r.Delivery, err = s.signPrincipal(v, interactiveDeliveryRequest(*r), now)
	return err
}

func verifyInteractiveDelivery(a trust.Anchor, v trust.VerifiedSnapshot, r InteractiveResult, now time.Time) error {
	if r.Delivery.Evidence.Principal != r.Proof.Evidence.Principal {
		return fmt.Errorf("delivery and approval authorities differ")
	}
	_, err := trust.VerifyProof(a, v, r.Delivery, interactiveDeliveryRequest(r), now)
	return err
}
