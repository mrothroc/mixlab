package enrollment

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/internal/certificates"
)

const receiptVersion = "mixlab_enrollment_receipt_v1"
const approvalVersion = "mixlab_enrollment_approval_v1"

type Receipt struct {
	Version     string  `json:"version"`
	Invitation  string  `json:"invitation"`
	Purpose     Purpose `json:"purpose"`
	Cluster     string  `json:"cluster"`
	Audience    string  `json:"audience"`
	RequestHash string  `json:"request_hash"`
	Generation  uint64  `json:"trust_generation"`
	Sequence    uint64  `json:"sequence"`
}

type Approval struct {
	Version      string     `json:"version"`
	RequestID    string     `json:"request_id"`
	Policy       string     `json:"policy"`
	Cluster      string     `json:"cluster"`
	Fingerprint  string     `json:"fingerprint"`
	Audience     string     `json:"audience"`
	Purpose      Purpose    `json:"purpose"`
	Role         trust.Role `json:"role"`
	Principal    string     `json:"principal"`
	RequestHash  string     `json:"request_hash"`
	EvidenceHash string     `json:"evidence_hash"`
	Sequence     uint64     `json:"sequence"`
	ExpiresAt    int64      `json:"expires_at"`
}

type Result struct {
	Receipt        Receipt              `json:"receipt"`
	ReceiptProof   trust.SignedProof    `json:"receipt_proof"`
	ReceiptCommit  trust.AcceptedProof  `json:"receipt_commit"`
	Approval       Approval             `json:"approval"`
	ApprovalProof  trust.SignedProof    `json:"approval_proof"`
	ApprovalCommit trust.AcceptedProof  `json:"approval_commit"`
	Chain          [][]byte             `json:"chain"`
	Snapshot       trust.SignedSnapshot `json:"snapshot"`
}

func receiptRequest(r Receipt) trust.SignRequest {
	return trust.SignRequest{Version: trust.ProofVersion, Purpose: trust.EnrollmentReceipt,
		Digest: digestText(receiptVersion, r), Context: r.Invitation, Audience: r.Audience}
}
func approvalRequest(a Approval) trust.SignRequest {
	return trust.SignRequest{Version: trust.ProofVersion, Purpose: trust.EnrollmentApproval,
		Digest: digestText(approvalVersion, a), Context: a.RequestID, Audience: a.Audience}
}
func receiptEvidence(r Result) string {
	return digestText("mixlab_enrollment_receipt_evidence_v1", struct {
		Receipt Receipt           `json:"receipt"`
		Proof   trust.SignedProof `json:"proof"`
	}{r.Receipt, r.ReceiptProof})
}

func (s *Service) approve(i Invitation, req SignedRequest, seq uint64, v trust.VerifiedSnapshot, now time.Time) (Result, error) {
	var out Result
	id, err := certificates.NewID()
	if err != nil {
		return out, err
	}
	digest := digestText(requestVersion, req)
	out.Receipt = Receipt{receiptVersion, i.ID, i.Purpose, i.Cluster, i.Audience, digest, v.Generation(), seq}
	out.ReceiptProof, err = s.signPrincipal(v, receiptRequest(out.Receipt), now)
	if err != nil {
		return out, err
	}
	out.ReceiptCommit, err = trust.VerifyProof(s.a.Anchor, v, out.ReceiptProof, receiptRequest(out.Receipt), now)
	if err != nil {
		return out, err
	}
	out.Approval = Approval{approvalVersion, i.ID, "provisioned", i.Cluster, i.Fingerprint, i.Audience, i.Purpose, i.Role, id,
		digest, receiptEvidence(out), seq, i.ExpiresAt}
	out.ApprovalProof, err = s.signPrincipal(v, approvalRequest(out.Approval), now)
	if err != nil {
		return out, err
	}
	out.ApprovalCommit, err = trust.VerifyProof(s.a.Anchor, v, out.ApprovalProof, approvalRequest(out.Approval), now)
	return out, err
}

func (r Result) validate(e entry, sequence uint64) error {
	i, q := e.Invitation, *e.Request
	a, b := r.Approval, r.Receipt
	if a.Version != approvalVersion || a.RequestID != i.ID || a.Policy != "provisioned" || a.Cluster != i.Cluster ||
		a.Fingerprint != i.Fingerprint || a.Audience != i.Audience || a.Purpose != i.Purpose || a.Role != i.Role ||
		!certificates.ValidID(a.Principal) || a.RequestHash != digestText(requestVersion, q) || a.EvidenceHash != receiptEvidence(r) ||
		a.Sequence == 0 || a.Sequence > sequence || a.ExpiresAt != i.ExpiresAt || b.Version != receiptVersion ||
		b.Invitation != i.ID || b.Purpose != i.Purpose || b.Cluster != i.Cluster || b.Audience != i.Audience ||
		b.RequestHash != a.RequestHash || b.Sequence != a.Sequence || b.Generation == 0 ||
		r.ApprovalProof.Request != approvalRequest(a) || r.ReceiptProof.Request != receiptRequest(b) ||
		!digestOK(r.ApprovalCommit.Digest) || !digestOK(r.ReceiptCommit.Digest) ||
		r.ApprovalCommit.Generation != b.Generation || r.ReceiptCommit.Generation != b.Generation {
		return fmt.Errorf("invalid enrollment approval/receipt cross-binding")
	}
	if (e.Stage == "approved" && len(r.Chain) != 0) || (e.Stage == "issued" && len(r.Chain) != 3) {
		return fmt.Errorf("invalid issuance outcome")
	}
	return nil
}

// commitJournal is made only from this authority's persisted result, never
// supplied by a network request. It bridges trust's historical-proof verifier.
type commitJournal struct{ approval, receipt trust.AcceptedProof }

func (j commitJournal) AcceptedGeneration(digest string) (uint64, bool, error) {
	for _, c := range []trust.AcceptedProof{j.approval, j.receipt} {
		if c.Digest == digest {
			return c.Generation, true, nil
		}
	}
	return 0, false, nil
}

func (s *Service) verifyApproved(r Result, v trust.VerifiedSnapshot, now time.Time) error {
	if r.ApprovalProof.Evidence.Principal != s.owner || r.ReceiptProof.Evidence.Principal != s.owner {
		return fmt.Errorf("approval by another enrollment authority")
	}
	j := commitJournal{r.ApprovalCommit, r.ReceiptCommit}
	if err := trust.VerifyHistoricalProof(s.a.Anchor, v, r.ReceiptProof, receiptRequest(r.Receipt), j, now); err != nil {
		return err
	}
	return trust.VerifyHistoricalProof(s.a.Anchor, v, r.ApprovalProof, approvalRequest(r.Approval), j, now)
}

// RecoverApproved is authority-local reconciliation, not an anonymous retry
// route. It never consumes an available invitation or replaces an issued leaf.
func (s *Service) RecoverApproved(ctx context.Context, invitationID string, v trust.VerifiedSnapshot, now time.Time) (Result, error) {
	var out Result
	err := s.store.WithProcessLock(ctx, lockFile, func() error {
		var err error
		out, err = s.issueLocked(invitationID, v, now)
		return err
	})
	return out, err
}

func (s *Service) issueLocked(id string, v trust.VerifiedSnapshot, now time.Time) (Result, error) {
	old, r, err := s.loadForOperation(v, now)
	if err != nil {
		return Result{}, err
	}
	for n := range r.Entries {
		e := &r.Entries[n]
		if e.Invitation.ID != id {
			continue
		}
		if e.Result == nil || e.Request == nil {
			return Result{}, fmt.Errorf("invitation has no durable approval")
		}
		out := *e.Result
		if err := s.verifyApproved(out, v, now); err != nil {
			return Result{}, err
		}
		if e.Stage == "issued" {
			if err := validateChain(s.a.Anchor, v, *e.Request, out, now); err != nil {
				return Result{}, err
			}
			return out, nil
		}
		if now.Unix() >= out.Approval.ExpiresAt {
			return Result{}, ErrExpired
		}
		if !endpointAllowed(v, e.Invitation.Endpoint, e.Invitation.Audience, e.Invitation.Role) {
			return Result{}, fmt.Errorf("approval endpoint or role no longer eligible")
		}
		der, err := s.issueCertificate(e.Request.Request.PrincipalRequest, out.Approval.Principal, now)
		if err != nil {
			return Result{}, err
		}
		out.Chain = [][]byte{der, bytes.Clone(s.a.Issuer), s.a.Anchor.DER()}
		if err := validateChain(s.a.Anchor, v, *e.Request, out, now); err != nil {
			return Result{}, err
		}
		out.Snapshot, err = snapshotValue(v)
		if err != nil {
			return Result{}, err
		}
		e.Result, e.Stage = &out, "issued"
		r.Sequence++
		if err := s.write(old, r); err != nil {
			return Result{}, err
		}
		return out, nil
	}
	return Result{}, fmt.Errorf("unknown approved enrollment")
}

func validateChain(a trust.Anchor, v trust.VerifiedSnapshot, req SignedRequest, r Result, now time.Time) error {
	return validatePrincipalChain(a, v, req.Request.PrincipalRequest, r.Approval.Principal, r.Chain, now)
}

func validatePrincipalChain(a trust.Anchor, v trust.VerifiedSnapshot, req PrincipalRequest, principal string, chain [][]byte, now time.Time) error {
	id, err := trust.AuthenticatePrincipal(a, v, chain, now)
	if err != nil {
		return err
	}
	if id.Principal != principal || id.Role != req.Role {
		return fmt.Errorf("issued identity mismatch")
	}
	c, parsed, err := certificates.Parse(chain[0])
	if err != nil {
		return err
	}
	if !bytes.Equal(c.PublicKey.(ed25519.PublicKey), req.PublicKey) || !bytes.Equal(parsed.EnvelopeKey, req.EnvelopeKey) {
		return fmt.Errorf("issued key mismatch")
	}
	return nil
}

// ValidateResult is the enrollee's acceptance check, before atomically
// publishing a credential. The returned snapshot is verified under the pin
// from the protected invitation, never a root proposed by the server.
func ValidateResult(i Invitation, req SignedRequest, r Result, now time.Time) (trust.VerifiedSnapshot, error) {
	var zero trust.VerifiedSnapshot
	a, err := i.ValidateTarget(req.Request.Endpoint, req.Request.Audience, now)
	if err != nil {
		return zero, err
	}
	if err := req.verify(); err != nil {
		return zero, err
	}
	if !matches(i, req.Request) {
		return zero, fmt.Errorf("result for another invitation")
	}
	e := entry{Invitation: i, Stage: "issued", Request: &req, Result: &r}
	if err := r.validate(e, r.Approval.Sequence); err != nil {
		return zero, err
	}
	v, err := trust.VerifySnapshot(a, r.Snapshot, now)
	if err != nil {
		return zero, err
	}
	if !endpointAllowed(v, i.Endpoint, i.Audience, i.Role) {
		return zero, fmt.Errorf("result endpoint or role mismatch")
	}
	if r.ReceiptProof.Evidence.Principal != r.ApprovalProof.Evidence.Principal {
		return zero, fmt.Errorf("approval and receipt authorities differ")
	}
	if _, err := trust.VerifyProof(a, v, r.ReceiptProof, receiptRequest(r.Receipt), now); err != nil {
		return zero, err
	}
	if _, err := trust.VerifyProof(a, v, r.ApprovalProof, approvalRequest(r.Approval), now); err != nil {
		return zero, err
	}
	if err := validateChain(a, v, req, r, now); err != nil {
		return zero, err
	}
	return v, nil
}
