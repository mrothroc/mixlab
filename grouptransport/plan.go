// Package grouptransport binds encrypted connectivity to immutable DDP
// membership. It does not choose peers, issue certificates or launch workers.
package grouptransport

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net"
	"strconv"
	"time"

	"github.com/mrothroc/mixlab/distributed"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/trust"
)

const Version = "mixlab_secure_group_transport_v1"
const TLSPolicy = "mixlab_ddp_tls13_v1"
const Audience = "ddp/data"

type Member struct {
	Node            string                `json:"node"`
	Job             string                `json:"job"`
	MemberID        string                `json:"member_id"`
	Rank            int                   `json:"rank"`
	ManifestHash    string                `json:"manifest_hash"`
	Endpoint        string                `json:"endpoint"`
	CertificateHash string                `json:"certificate_hash"`
	Chain           [][]byte              `json:"chain"`
	Binding         trust.WorkloadBinding `json:"binding"`
}
type Plan struct {
	Version       string                         `json:"version"`
	TLSPolicy     string                         `json:"tls_policy"`
	Cluster       string                         `json:"cluster"`
	Controller    string                         `json:"controller"`
	Attempt       string                         `json:"attempt"`
	Nonce         string                         `json:"nonce"`
	Membership    distributed.DDPGroupMembership `json:"membership"`
	Members       []Member                       `json:"members"`
	LoopbackPorts []int                          `json:"loopback_ports"`
	Created       int64                          `json:"created"`
	Expires       int64                          `json:"expires"`
}
type Signed struct {
	Plan  Plan              `json:"plan"`
	Proof trust.SignedProof `json:"proof"`
}

func hexValue(s string, size int) bool {
	b, e := hex.DecodeString(s)
	return e == nil && len(b) == size && hex.EncodeToString(b) == s
}

func (p Plan) Validate() error {
	if p.Version != Version || p.TLSPolicy != TLSPolicy || p.Created <= 0 || p.Expires <= p.Created || p.Expires-p.Created > int64(trust.MaxWorkloadLifetime/time.Second) {
		return fmt.Errorf("invalid secure transport profile/lifetime")
	}
	for _, s := range []string{p.Cluster, p.Controller, p.Attempt, p.Nonce, p.Membership.RunID, p.Membership.GroupID} {
		if !hexValue(s, 16) {
			return fmt.Errorf("invalid transport identity")
		}
	}
	m, err := p.Membership.Canonical()
	if err != nil {
		return err
	}
	if m.Backend != "ring" || m.Generation == 0 || m.WorldSize() < 2 || m.WorldSize() > 64 || len(p.Members) != m.WorldSize() || len(p.LoopbackPorts) != m.WorldSize() {
		return fmt.Errorf("secure transport requires exact fixed ring membership")
	}
	hash := m.MembersHash
	ports := map[int]bool{}
	for _, port := range p.LoopbackPorts {
		if port < 1024 || port > 65535 || ports[port] {
			return fmt.Errorf("invalid or duplicate loopback ring port")
		}
		ports[port] = true
	}
	nodes, jobs, certs, endpoints := map[string]bool{}, map[string]bool{}, map[string]bool{}, map[string]bool{}
	for i, v := range p.Members {
		if !hexValue(v.Node, 16) || !hexValue(v.Job, 16) || !hexValue(v.MemberID, 16) || !hexValue(v.ManifestHash, 32) || !hexValue(v.CertificateHash, 32) || v.Rank != i || v.MemberID != m.OrderedMembers[i].MemberID || nodes[v.Node] || jobs[v.Job] || certs[v.CertificateHash] || endpoints[v.Endpoint] {
			return fmt.Errorf("duplicate or mismatched transport member")
		}
		nodes[v.Node], jobs[v.Job], certs[v.CertificateHash], endpoints[v.Endpoint] = true, true, true, true
		host, port, e := net.SplitHostPort(v.Endpoint)
		n, pe := strconv.Atoi(port)
		ip := net.ParseIP(host)
		// Signed literal addresses avoid name-resolution changes selecting a
		// different transport route after admission. Discovery may use DNS.
		if e != nil || pe != nil || ip == nil || ip.IsUnspecified() || ip.IsMulticast() || n < 1024 || n > 65535 || net.JoinHostPort(ip.String(), strconv.Itoa(n)) != v.Endpoint {
			return fmt.Errorf("invalid signed transport endpoint")
		}
		if len(v.Chain) != 3 || len(v.Chain[0]) > 16384 || len(v.Chain[1]) > 16384 || len(v.Chain[2]) > 16384 || nodejob.Hash(v.Chain[0]) != v.CertificateHash {
			return fmt.Errorf("transport certificate fingerprint mismatch")
		}
		b := v.Binding
		if !hexValue(b.Lease, 16) || b.ManifestHash != v.ManifestHash {
			return fmt.Errorf("transport workload admission binding mismatch")
		}
		if b.Version != trust.WorkloadBindingVersion || b.Cluster != p.Cluster || b.Role != trust.Worker || !hexValue(b.Principal, 16) || b.Participant != v.Node || b.Job != v.Job || b.Run != m.RunID || b.Group != m.GroupID || b.Generation != m.Generation || b.MembershipHash != hash || b.Member != v.MemberID || b.Rank != i || b.Attempt != p.Attempt || b.Audience != Audience || b.IssuedAt > p.Created || b.ExpiresAt < p.Expires {
			return fmt.Errorf("transport workload binding mismatch")
		}
	}
	b, err := json.Marshal(p)
	if err != nil || len(b) > 1<<20 {
		return fmt.Errorf("transport plan too large")
	}
	return nil
}

func (p Plan) SigningRequest() (trust.SignRequest, error) {
	if err := p.Validate(); err != nil {
		return trust.SignRequest{}, err
	}
	b, err := json.Marshal(p)
	if err != nil {
		return trust.SignRequest{}, err
	}
	return trust.SignRequest{Version: trust.ProofVersion, Purpose: trust.SecureTransportPlan, Digest: nodejob.Hash(b), Context: p.Membership.RunID + "/" + p.Membership.GroupID + "/" + p.Attempt, Audience: p.Membership.RunID}, nil
}

// Accepted is immutable verification evidence, not an API input. Each node
// also compares the exact local certificate against its prepared key request.
type Accepted struct {
	signed Signed
	commit trust.AcceptedProof
	job    string
}

func Accept(a trust.Anchor, v trust.VerifiedSnapshot, actor trust.AuthenticatedPrincipal, job nodejob.Manifest, localChain [][]byte, s Signed, now time.Time) (Accepted, error) {
	b, err := json.Marshal(s)
	if err != nil || len(b) > 1<<20+trust.MaxTrustBytes {
		return Accepted{}, fmt.Errorf("signed transport plan too large")
	}
	var copy Signed
	if err := json.Unmarshal(b, &copy); err != nil {
		return Accepted{}, err
	}
	s = copy
	request, err := s.Plan.SigningRequest()
	if err != nil {
		return Accepted{}, err
	}
	j, err := job.SigningRequest()
	if err != nil {
		return Accepted{}, err
	}
	p := s.Plan
	deadline, err := job.WorkloadDeadline()
	if err != nil || p.Created < job.Created || p.Expires > deadline.Unix() {
		return Accepted{}, fmt.Errorf("transport exceeds admitted job lifetime")
	}
	if actor.Cluster != a.Cluster() || p.Cluster != a.Cluster() || actor.Role != trust.Controller || actor.Principal != p.Controller || actor.Principal != job.Controller || s.Proof.Evidence.Principal != actor.Principal || now.Before(actor.NotBefore) || !now.Before(actor.ExpiresAt) || now.Unix() < p.Created || now.Unix() >= p.Expires || p.Attempt != job.Attempt {
		return Accepted{}, fmt.Errorf("transport controller, attempt or lifetime mismatch")
	}
	jm, _ := json.Marshal(job.Membership)
	pm, _ := json.Marshal(p.Membership)
	if !bytes.Equal(jm, pm) || len(job.Members) != len(p.Members) {
		return Accepted{}, fmt.Errorf("transport changed membership")
	}
	for i, member := range job.Members {
		if member.Node != p.Members[i].Node || member.MemberID != p.Members[i].MemberID || member.Rank != p.Members[i].Rank {
			return Accepted{}, fmt.Errorf("transport changed node assignment")
		}
	}
	local := p.Members[job.Rank]
	if local.Binding.IssuedAt < job.Created || local.Binding.ExpiresAt > deadline.Unix() {
		return Accepted{}, fmt.Errorf("workload certificate exceeds admitted job lifetime")
	}
	if local.Node != job.Node || local.Job != job.Job || local.Binding.Lease != job.Lease || local.ManifestHash != j.Digest || len(localChain) != 3 {
		return Accepted{}, fmt.Errorf("transport does not match local prepared job")
	}
	for i := range localChain {
		if !bytes.Equal(localChain[i], local.Chain[i]) {
			return Accepted{}, fmt.Errorf("transport substituted local workload key/certificate")
		}
	}
	commit, err := trust.VerifyProof(a, v, s.Proof, request, now)
	if err != nil {
		return Accepted{}, err
	}
	for _, peer := range p.Members {
		if err := trust.VerifyWorkload(a, v, peer.Chain, peer.Binding, now); err != nil {
			return Accepted{}, err
		}
	}
	return Accepted{s, commit, job.Job}, nil
}

func (a Accepted) Value() (Signed, trust.AcceptedProof, string, error) {
	if a.commit.Generation == 0 {
		return Signed{}, trust.AcceptedProof{}, "", fmt.Errorf("unverified secure transport plan")
	}
	b, err := json.Marshal(a.signed)
	if err != nil {
		return Signed{}, trust.AcceptedProof{}, "", err
	}
	var out Signed
	err = json.Unmarshal(b, &out)
	return out, a.commit, a.job, err
}
