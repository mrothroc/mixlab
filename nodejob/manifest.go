// Package nodejob defines signed managed DDP jobs. It binds authorization to
// immutable membership and logical data/build identities, never shell commands,
// local paths, environment variables, or executable selection.
package nodejob

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/distributed"
	"github.com/mrothroc/mixlab/executionlimits"
	"github.com/mrothroc/mixlab/internal/strictjson"
	"github.com/mrothroc/mixlab/trust"
)

const Version = "mixlab_node_job_v1"
const MaxManifestBytes = 320 << 10

type Limits = executionlimits.Limits
type ArtifactRef struct {
	SHA256 string `json:"sha256"`
	Bytes  uint64 `json:"bytes"`
	Kind   string `json:"kind"`
}
type Member struct {
	Node     string `json:"node"`
	MemberID string `json:"member_id"`
	Rank     int    `json:"rank"`
}
type Manifest struct {
	Version          string                         `json:"version"`
	Job              string                         `json:"job"`
	Attempt          string                         `json:"attempt"`
	Lease            string                         `json:"lease"`
	Node             string                         `json:"node"`
	Controller       string                         `json:"controller"`
	Nonce            string                         `json:"nonce"`
	Membership       distributed.DDPGroupMembership `json:"membership"`
	Members          []Member                       `json:"members"`
	Rank             int                            `json:"rank"`
	BuildID          string                         `json:"build_id"`
	Config           json.RawMessage                `json:"config"`
	ConfigHash       string                         `json:"config_hash"`
	ProgramHash      string                         `json:"program_hash"`
	WeightLayoutHash string                         `json:"weight_layout_hash"`
	OptimizerHash    string                         `json:"optimizer_hash"`
	DatasetSelector  string                         `json:"dataset_selector"`
	DatasetID        string                         `json:"dataset_id"`
	Artifacts        []ArtifactRef                  `json:"artifacts"`
	CheckpointAt     uint64                         `json:"checkpoint_at,omitempty"`
	Mode             string                         `json:"mode"`
	Transport        string                         `json:"transport"`
	Limits           Limits                         `json:"limits"`
	Created          int64                          `json:"created"`
	Expires          int64                          `json:"expires"`
}
type Signed struct {
	Manifest Manifest          `json:"manifest"`
	Proof    trust.SignedProof `json:"proof"`
}

func hexID(s string, n int) bool {
	b, e := hex.DecodeString(s)
	return e == nil && len(b) == n && hex.EncodeToString(b) == s
}
func Hash(b []byte) string { d := sha256.Sum256(b); return hex.EncodeToString(d[:]) }
func CanonicalConfig(b []byte) ([]byte, error) {
	if len(b) == 0 || len(b) > 256<<10 {
		return nil, fmt.Errorf("bounded config object required")
	}
	if err := strictjson.Validate(b, 64); err != nil {
		return nil, err
	}
	var object map[string]json.RawMessage
	if err := json.Unmarshal(b, &object); err != nil || object == nil {
		return nil, fmt.Errorf("config must be a JSON object")
	}
	// Canonical compact bytes preserve explicit zeros and number spelling. The
	// trainer remains responsible for its full schema/default validation.
	var out bytes.Buffer
	if err := json.Compact(&out, b); err != nil {
		return nil, err
	}
	return out.Bytes(), nil
}
func (m Manifest) Validate() error {
	if m.CheckpointAt > 1<<31-1 {
		return fmt.Errorf("checkpoint stop exceeds supported step range")
	}
	if m.Version != Version || m.Mode != "arch" || m.Transport != "tls13-ring" || m.Created <= 0 || m.Expires <= m.Created || m.Expires-m.Created > int64(time.Hour/time.Second) {
		return fmt.Errorf("invalid managed job format, mode, transport or admission lifetime")
	}
	for _, id := range []string{m.Job, m.Attempt, m.Lease, m.Node, m.Controller, m.Nonce, m.Membership.RunID, m.Membership.GroupID} {
		if !hexID(id, 16) {
			return fmt.Errorf("managed job requires canonical random identifiers")
		}
	}
	for _, h := range []string{m.BuildID, m.ConfigHash, m.ProgramHash, m.WeightLayoutHash, m.OptimizerHash, m.DatasetID} {
		if !hexID(h, 32) {
			return fmt.Errorf("managed job requires exact SHA256 identities")
		}
	}
	config, err := CanonicalConfig(m.Config)
	if err != nil || !bytes.Equal(config, m.Config) || Hash(config) != m.ConfigHash {
		return fmt.Errorf("noncanonical or mismatched embedded config")
	}
	if m.DatasetSelector == "" || len(m.DatasetSelector) > 128 {
		return fmt.Errorf("logical dataset selector required")
	}
	for _, c := range m.DatasetSelector {
		valid := (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.'
		if !valid {
			return fmt.Errorf("dataset selector cannot contain paths")
		}
	}
	if m.DatasetSelector == "." || m.DatasetSelector == ".." {
		return fmt.Errorf("invalid dataset selector")
	}
	membership, err := m.Membership.Canonical()
	if err != nil {
		return err
	}
	if membership.Backend != "ring" || membership.Generation == 0 || membership.WorldSize() < 2 || membership.WorldSize() > 64 || len(m.Members) != membership.WorldSize() || m.Rank < 0 || m.Rank >= len(m.Members) {
		return fmt.Errorf("fixed managed ring membership required")
	}
	nodes := map[string]bool{}
	for i, member := range m.Members {
		if !hexID(member.Node, 16) || nodes[member.Node] || member.Rank != i || member.MemberID != membership.OrderedMembers[i].MemberID || !hexID(member.MemberID, 16) {
			return fmt.Errorf("member/node/rank assignment mismatch")
		}
		nodes[member.Node] = true
	}
	if m.Members[m.Rank].Node != m.Node {
		return fmt.Errorf("manifest targets another rank's node")
	}
	l := m.Limits
	if err := l.Validate(); err != nil {
		return err
	}
	if _, err := m.WorkloadDeadline(); err != nil {
		return err
	}
	if len(m.Artifacts) > 8 {
		return fmt.Errorf("too many managed job artifacts")
	}
	seen := map[string]bool{}
	var total uint64
	for _, a := range m.Artifacts {
		if !hexID(a.SHA256, 32) || a.Bytes == 0 || a.Bytes > 1<<30 || (a.Kind != "weights" && a.Kind != "checkpoint") || seen[a.SHA256] {
			return fmt.Errorf("invalid managed artifact reference")
		}
		seen[a.SHA256] = true
		total += a.Bytes
	}
	if total+l.LogBytes > l.DiskBytes {
		return fmt.Errorf("artifact/log limits exceed job disk budget")
	}
	b, err := json.Marshal(m)
	if err != nil || len(b) > MaxManifestBytes {
		return fmt.Errorf("manifest too large")
	}
	return nil
}
func (m Manifest) SigningRequest() (trust.SignRequest, error) {
	if err := m.Validate(); err != nil {
		return trust.SignRequest{}, err
	}
	b, err := json.Marshal(m)
	if err != nil {
		return trust.SignRequest{}, err
	}
	return trust.SignRequest{Version: trust.ProofVersion, Purpose: trust.NodeJob, Digest: Hash(b), Context: m.Job + "/" + m.Attempt, Audience: m.Node}, nil
}

// Accepted is constructed only after signature and current controller checks.
// It is not a JSON input type and carries no process-launch authority itself.
type Accepted struct {
	job    Signed
	commit trust.AcceptedProof
}

func Accept(a trust.Anchor, v trust.VerifiedSnapshot, actor trust.AuthenticatedPrincipal, node string, job Signed, now time.Time) (Accepted, error) {
	b, err := json.Marshal(job)
	if err != nil || len(b) > MaxManifestBytes+trust.MaxTrustBytes {
		return Accepted{}, fmt.Errorf("signed job too large")
	}
	var copy Signed
	if err := json.Unmarshal(b, &copy); err != nil {
		return Accepted{}, err
	}
	job = copy
	m := job.Manifest
	request, err := m.SigningRequest()
	if err != nil {
		return Accepted{}, err
	}
	if m.Node != node || actor.Role != trust.Controller || actor.Cluster != a.Cluster() || actor.Principal != m.Controller || actor.Principal != job.Proof.Evidence.Principal || now.Before(actor.NotBefore) || !now.Before(actor.ExpiresAt) || now.Unix() < m.Created || now.Unix() >= m.Expires {
		return Accepted{}, fmt.Errorf("job controller, target or admission deadline mismatch")
	}
	commit, err := trust.VerifyProof(a, v, job.Proof, request, now)
	if err != nil {
		return Accepted{}, err
	}
	return Accepted{job, commit}, nil
}
func (a Accepted) Value() (Signed, trust.AcceptedProof, error) {
	if a.commit.Generation == 0 {
		return Signed{}, trust.AcceptedProof{}, fmt.Errorf("unverified job")
	}
	b, err := json.Marshal(a.job)
	if err != nil {
		return Signed{}, trust.AcceptedProof{}, err
	}
	var out Signed
	err = json.Unmarshal(b, &out)
	return out, a.commit, err
}
