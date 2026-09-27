// Package workload owns bounded workload certificate issuance. Scope is an
// opaque admission binding supplied by node/controller applications. This
// context never reads leases, job manifests, membership plans or datasets.
package workload

import (
	"bytes"
	"crypto"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/trust"
)

const Version = "mixlab_ddp_workload_request_v1"
const Audience = "ddp/data"

type Scope struct {
	Cluster        string `json:"cluster"`
	Controller     string `json:"controller"`
	Node           string `json:"node"`
	Lease          string `json:"lease"`
	Run            string `json:"run"`
	Job            string `json:"job"`
	Attempt        string `json:"attempt"`
	Group          string `json:"group"`
	Member         string `json:"member"`
	Workload       string `json:"workload"`
	Generation     uint64 `json:"generation"`
	Rank           int    `json:"rank"`
	MembershipHash string `json:"membership_hash"`
	ManifestHash   string `json:"manifest_hash"`
	Created        int64  `json:"created"`
	AdmitUntil     int64  `json:"admit_until"`
	Deadline       int64  `json:"deadline"`
}

func hexSize(s string, n int) bool {
	b, err := hex.DecodeString(s)
	return err == nil && len(b) == n && hex.EncodeToString(b) == s
}

func (s Scope) Validate() error {
	for _, id := range []string{s.Cluster, s.Controller, s.Node, s.Lease, s.Run, s.Job, s.Attempt, s.Group, s.Member, s.Workload} {
		if !hexSize(id, 16) {
			return fmt.Errorf("invalid workload scope identity")
		}
	}
	if !hexSize(s.MembershipHash, 32) || !hexSize(s.ManifestHash, 32) || s.Generation == 0 || s.Rank < 0 || s.Rank >= 64 || s.Created <= 0 || s.AdmitUntil <= s.Created || s.AdmitUntil-s.Created > 3600 || s.Deadline <= s.AdmitUntil || s.Deadline-s.Created > int64(trust.MaxWorkloadLifetime/time.Second) {
		return fmt.Errorf("invalid workload scope binding or lifetime")
	}
	return nil
}

func digest(v any) string {
	b, _ := json.Marshal(v)
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}

type Request struct {
	Version string `json:"version"`
	Scope   Scope  `json:"scope"`
	CSR     []byte `json:"csr"`
}
type SignedRequest struct {
	Request Request           `json:"request"`
	Proof   trust.SignedProof `json:"proof"`
}
type Grant struct {
	Request SignedRequest     `json:"request"`
	Proof   trust.SignedProof `json:"proof"`
}
type Result struct {
	GrantHash string                `json:"grant_hash"`
	Binding   trust.WorkloadBinding `json:"binding"`
	Chain     [][]byte              `json:"chain"`
	Snapshot  trust.SignedSnapshot  `json:"snapshot"`
}

// NewRequest takes a protected, locally generated transport signing handle.
// The standard PKCS#10 signature proves possession independently of the node's
// principal signature; its subject binds every requested admission field.
func NewRequest(scope Scope, key crypto.Signer) (Request, error) {
	if err := scope.Validate(); err != nil {
		return Request{}, err
	}
	if key == nil {
		return Request{}, fmt.Errorf("local workload signer required")
	}
	if p, ok := key.Public().(ed25519.PublicKey); !ok || len(p) != ed25519.PublicKeySize {
		return Request{}, fmt.Errorf("Ed25519 workload key required")
	}
	csr, err := x509.CreateCertificateRequest(rand.Reader, &x509.CertificateRequest{Subject: pkix.Name{CommonName: digest(scope)}}, key)
	return Request{Version, scope, csr}, err
}

func (r Request) PublicKey() (ed25519.PublicKey, error) {
	if r.Version != Version || len(r.CSR) == 0 || len(r.CSR) > 4096 {
		return nil, fmt.Errorf("invalid workload key request")
	}
	if err := r.Scope.Validate(); err != nil {
		return nil, err
	}
	csr, err := x509.ParseCertificateRequest(r.CSR)
	if err != nil {
		return nil, err
	}
	if err := csr.CheckSignature(); err != nil {
		return nil, err
	}
	p, ok := csr.PublicKey.(ed25519.PublicKey)
	if !ok || csr.Version != 0 || csr.SignatureAlgorithm != x509.PureEd25519 || csr.Subject.CommonName != digest(r.Scope) || len(csr.Extensions) != 0 || len(csr.Subject.Names) != 1 {
		return nil, fmt.Errorf("workload CSR profile/scope mismatch")
	}
	return bytes.Clone(p), nil
}

func (r Request) SigningRequest() (trust.SignRequest, error) {
	if _, err := r.PublicKey(); err != nil {
		return trust.SignRequest{}, err
	}
	return trust.SignRequest{Version: trust.ProofVersion, Purpose: trust.WorkloadKeyRequest, Digest: digest(r), Context: r.Scope.Job + "/" + r.Scope.Attempt, Audience: r.Scope.Controller}, nil
}

func (r SignedRequest) GrantRequest() (trust.SignRequest, error) {
	want, err := r.Request.SigningRequest()
	if err != nil {
		return trust.SignRequest{}, err
	}
	if r.Proof.Request != want || r.Proof.Evidence.Principal != r.Request.Scope.Node || r.Proof.Evidence.Role != trust.Node {
		return trust.SignRequest{}, fmt.Errorf("node request proof binding mismatch")
	}
	return trust.SignRequest{Version: trust.ProofVersion, Purpose: trust.WorkloadGrant, Digest: digest(r), Context: want.Context, Audience: r.Request.Scope.Node}, nil
}

func (s Scope) binding(issued int64) trust.WorkloadBinding {
	return trust.WorkloadBinding{Version: trust.WorkloadBindingVersion, Cluster: s.Cluster, Role: trust.Worker, Principal: s.Workload, Participant: s.Node, Run: s.Run, Job: s.Job, Lease: s.Lease, ManifestHash: s.ManifestHash, Attempt: s.Attempt, Audience: Audience, IssuedAt: issued, ExpiresAt: s.Deadline, Group: s.Group, Generation: s.Generation, MembershipHash: s.MembershipHash, Member: s.Member, Rank: s.Rank}
}

func ValidateResult(a trust.Anchor, grant Grant, out Result, now time.Time) error {
	s := grant.Request.Request.Scope
	public, err := grant.Request.Request.PublicKey()
	if err != nil {
		return err
	}
	if out.GrantHash != digest(grant) || len(out.Chain) != 3 || out.Binding.IssuedAt < s.Created || out.Binding.IssuedAt >= s.AdmitUntil || out.Binding != s.binding(out.Binding.IssuedAt) {
		return fmt.Errorf("workload result changed admission binding")
	}
	cert, err := x509.ParseCertificate(out.Chain[0])
	if err != nil {
		return err
	}
	p, ok := cert.PublicKey.(ed25519.PublicKey)
	if !ok || !bytes.Equal(public, p) {
		return fmt.Errorf("workload result substituted transport key")
	}
	v, err := trust.VerifySnapshot(a, out.Snapshot, now)
	if err != nil {
		return err
	}
	return trust.VerifyWorkload(a, v, out.Chain, out.Binding, now)
}
