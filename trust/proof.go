package trust

import (
	"bytes"
	"crypto"
	"crypto/ed25519"
	"encoding/hex"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/trust/internal/certificates"
)

type Purpose string

const (
	ProofVersion                      = "mixlab_principal_signature_v1"
	EnrollmentApproval        Purpose = "enrollment_approval"
	EnrollmentReceipt         Purpose = "enrollment_receipt"
	NodeJob                   Purpose = "node_job"
	SecureTransportPlan       Purpose = "secure_transport_plan"
	RunPlan                   Purpose = "run_plan"
	RunCommit                 Purpose = "run_commit"
	RoundGrant                Purpose = "round_grant"
	RecoveryGrant             Purpose = "recovery_grant"
	ArtifactGrant             Purpose = "artifact_grant"
	CredentialEnvelopePurpose Purpose = "credential_envelope"
	WorkloadKeyRequest        Purpose = "workload_key_request"
	WorkloadGrant             Purpose = "workload_grant"
)

type SignRequest struct {
	Version  string  `json:"version"`
	Purpose  Purpose `json:"purpose"`
	Digest   string  `json:"digest"`
	Context  string  `json:"context"`
	Audience string  `json:"audience"`
}

func (r SignRequest) validate(role Role) error {
	digest, err := hex.DecodeString(r.Digest)
	if err != nil || len(digest) != 32 || hex.EncodeToString(digest) != r.Digest || r.Version != ProofVersion || !validText(r.Context, 256) || !validText(r.Audience, 256) {
		return fmt.Errorf("invalid principal signing request")
	}
	allowed := false
	switch role {
	case Authority:
		allowed = r.Purpose == EnrollmentApproval || r.Purpose == EnrollmentReceipt
	case Controller:
		allowed = r.Purpose == NodeJob || r.Purpose == SecureTransportPlan || r.Purpose == CredentialEnvelopePurpose || r.Purpose == WorkloadGrant
	case Node:
		allowed = r.Purpose == WorkloadKeyRequest
	case Coordinator:
		allowed = r.Purpose == RunPlan || r.Purpose == RunCommit || r.Purpose == RoundGrant || r.Purpose == RecoveryGrant || r.Purpose == ArtifactGrant
	}
	if !allowed {
		return fmt.Errorf("role %q cannot sign purpose %q", role, r.Purpose)
	}
	return nil
}

type TrustEvidence struct {
	Cluster   string         `json:"cluster"`
	Principal string         `json:"principal"`
	Role      Role           `json:"role"`
	Serial    string         `json:"serial"`
	Chain     [][]byte       `json:"chain"`
	Snapshot  SignedSnapshot `json:"snapshot"`
	Algorithm string         `json:"algorithm"`
	SignedAt  int64          `json:"signed_at"` // diagnostic only, NEVER an acceptance clock
}

type SignedProof struct {
	Request   SignRequest   `json:"request"`
	Evidence  TrustEvidence `json:"evidence"`
	Signature []byte        `json:"signature"`
}

func proofBody(p SignedProof) any {
	return struct {
		Request  SignRequest   `json:"request"`
		Evidence TrustEvidence `json:"evidence"`
	}{p.Request, p.Evidence}
}

type AcceptedProof struct {
	Digest     string
	Generation uint64
}

// CommitJournal is supplied ONLY by the owning job/run/artifact context. It
// returns a generation from its durable acceptance journal, never from the
// proof, a network caller, or a signer-supplied timestamp. Trust owns no journal.
type CommitJournal interface {
	AcceptedGeneration(proofDigest string) (generation uint64, found bool, err error)
}

func principal(a Anchor, chain [][]byte, v VerifiedSnapshot, now time.Time) (certificates.Identity, error) {
	var zero certificates.Identity
	if len(chain) != 3 || !bytes.Equal(chain[2], a.DER()) || v.anchor.Fingerprint() != a.Fingerprint() {
		return zero, fmt.Errorf("wrong principal chain or pinned root")
	}
	c, id, err := certificates.Verify(a, chain[0], chain[1], certificates.Principal, now)
	if err != nil {
		return zero, err
	}
	if !v.eligible(id.Role) {
		return zero, fmt.Errorf("principal role is not eligible")
	}
	active := false
	for _, issuer := range v.signed.Payload.Issuers {
		if bytes.Equal(issuer, chain[1]) {
			active = true
		}
		ca, _, err := certificates.Parse(issuer)
		if err != nil || bytes.Equal(c.RawSubjectPublicKeyInfo, ca.RawSubjectPublicKeyInfo) {
			return zero, fmt.Errorf("principal reuses a CA key")
		}
	}
	signer, _, err := certificates.Parse(v.signed.SignerCertificate)
	if err != nil || bytes.Equal(c.RawSubjectPublicKeyInfo, signer.RawSubjectPublicKeyInfo) {
		return zero, fmt.Errorf("principal reuses the snapshot-signing key")
	}
	if !active {
		return zero, fmt.Errorf("principal issuer is not active")
	}
	return id, nil
}

func (v VerifiedSnapshot) checkRevocation(id certificates.Identity, acceptedGeneration uint64) error {
	for _, r := range v.signed.Payload.Revocations {
		if (r.Kind == "principal" && r.ID == id.Principal) || (r.Kind == "certificate" && r.ID == id.Serial) {
			if acceptedGeneration == 0 || r.Mode == "compromise" || acceptedGeneration >= r.FirstGeneration {
				return fmt.Errorf("principal or certificate revoked (%s)", r.Mode)
			}
		}
	}
	return nil
}

// SignPrincipalProof is a purpose-restricted operation over a caller-owned
// digest/context/audience. It does not parse manifests or grant their actions.
func SignPrincipalProof(a Anchor, chain [][]byte, key crypto.Signer, v VerifiedSnapshot, r SignRequest, now time.Time) (SignedProof, error) {
	if err := v.fresh(now); err != nil {
		return SignedProof{}, err
	}
	id, err := principal(a, chain, v, now)
	if err != nil {
		return SignedProof{}, err
	}
	if err = r.validate(id.Role); err != nil {
		return SignedProof{}, err
	}
	if err = v.checkRevocation(id, 0); err != nil {
		return SignedProof{}, err
	}
	c, _, err := certificates.Parse(chain[0])
	if err != nil {
		return SignedProof{}, err
	}
	if err = certificates.MatchSigner(c, key); err != nil {
		return SignedProof{}, err
	}
	p := SignedProof{Request: r, Evidence: TrustEvidence{Cluster: a.Cluster(), Principal: id.Principal, Role: id.Role, Serial: id.Serial, Chain: chain, Snapshot: v.signed, Algorithm: "ed25519", SignedAt: now.Unix()}}
	// Freeze evidence and reserve the complete wire size before using the key.
	p.Signature = make([]byte, ed25519.SignatureSize)
	b, err := canonical(p)
	if err != nil {
		return SignedProof{}, err
	}
	var frozen SignedProof
	if err = decodeCanonical(b, &frozen); err != nil {
		return SignedProof{}, err
	}
	frozen.Signature, err = sign(ProofVersion, proofBody(frozen), key)
	return frozen, err
}

// VerifyProof performs initial acceptance against fresh CURRENT trust. The
// caller must atomically journal the returned digest/generation with its own
// domain commit; signature verification alone is not historical acceptance.
func VerifyProof(a Anchor, v VerifiedSnapshot, p SignedProof, want SignRequest, now time.Time) (AcceptedProof, error) {
	if err := v.fresh(now); err != nil {
		return AcceptedProof{}, err
	}
	id, err := verifyProofEvidence(a, v, p, want, now, false)
	if err != nil {
		return AcceptedProof{}, err
	}
	if err = v.checkRevocation(id, 0); err != nil {
		return AcceptedProof{}, err
	}
	if _, err = principal(a, p.Evidence.Chain, v, now); err != nil {
		return AcceptedProof{}, err
	}
	digest, err := objectDigest(p)
	return AcceptedProof{Digest: digest, Generation: v.Generation()}, err
}

// VerifyHistoricalProof preserves ordinary expired/renewed proofs only with
// owning-context commit evidence. Prospective revocation never creates a new
// acceptance from old evidence; compromise revocation invalidates all history.
func VerifyHistoricalProof(a Anchor, v VerifiedSnapshot, p SignedProof, want SignRequest, journal CommitJournal, now time.Time) error {
	if err := v.fresh(now); err != nil {
		return err
	}
	if journal == nil {
		return fmt.Errorf("historical verification requires an owning-context commit journal")
	}
	digest, err := objectDigest(p)
	if err != nil {
		return err
	}
	generation, found, err := journal.AcceptedGeneration(digest)
	if err != nil {
		return err
	}
	if !found || generation == 0 || generation > v.Generation() || generation < p.Evidence.Snapshot.Payload.Generation {
		return fmt.Errorf("proof has no valid prior commit")
	}
	id, err := verifyProofEvidence(a, v, p, want, now, true)
	if err != nil {
		return err
	}
	return v.checkRevocation(id, generation)
}

func verifyProofEvidence(a Anchor, current VerifiedSnapshot, p SignedProof, want SignRequest, now time.Time, historical bool) (certificates.Identity, error) {
	var zero certificates.Identity
	if p.Request != want || current.anchor.Fingerprint() != a.Fingerprint() || p.Evidence.Cluster != a.Cluster() ||
		p.Evidence.Algorithm != "ed25519" || p.Evidence.Snapshot.Payload.Generation > current.Generation() || p.Evidence.SignedAt <= 0 {
		return zero, fmt.Errorf("proof request or evidence binding mismatch")
	}
	if _, err := canonical(p); err != nil {
		return zero, err
	}
	// Only a verified prior journal entry permits these historical validation
	// times. SignedAt is never used, so a stolen key cannot backdate acceptance.
	snapshotTime := now
	certificateTime := now
	if historical {
		snapshotTime = time.Unix(p.Evidence.Snapshot.Payload.IssuedAt, 0)
		if len(p.Evidence.Chain) != 3 {
			return zero, fmt.Errorf("invalid historical chain")
		}
		c, _, err := certificates.Parse(p.Evidence.Chain[0])
		if err != nil {
			return zero, err
		}
		certificateTime = c.NotBefore
	}
	evidenceView, err := VerifySnapshot(a, p.Evidence.Snapshot, snapshotTime)
	if err != nil {
		return zero, err
	}
	id, err := principal(a, p.Evidence.Chain, evidenceView, certificateTime)
	if err != nil {
		return zero, err
	}
	if id.Principal != p.Evidence.Principal || id.Role != p.Evidence.Role || id.Serial != p.Evidence.Serial {
		return zero, fmt.Errorf("proof signer identity mismatch")
	}
	if err = p.Request.validate(id.Role); err != nil {
		return zero, err
	}
	if err = evidenceView.checkRevocation(id, 0); err != nil {
		return zero, err
	}
	c, _, err := certificates.Parse(p.Evidence.Chain[0])
	if err != nil {
		return zero, err
	}
	if err = verify(ProofVersion, proofBody(p), c.PublicKey, p.Signature); err != nil {
		return zero, err
	}
	return id, nil
}

func DecodeProof(body []byte) (SignedProof, error) {
	var p SignedProof
	err := decodeCanonical(body, &p)
	return p, err
}
