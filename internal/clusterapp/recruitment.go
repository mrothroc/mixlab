package clusterapp

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"time"

	"github.com/mrothroc/mixlab/distributed"
	"github.com/mrothroc/mixlab/grouptransport"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/recruitment"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/principal"
	"github.com/mrothroc/mixlab/trust/workload"
	"github.com/mrothroc/mixlab/workerprobe"
)

type RecruitmentOptions struct {
	Principal         *principal.Store
	Config            json.RawMessage
	NumericalPlan     workerprobe.Plan
	AuthorityEndpoint string
	LoopbackPorts     []int
	Clock             func() time.Time
	CheckpointAt      uint64
	ResumeSource      *nodejob.Manifest
	ResumeArtifact    *nodejob.ArtifactRef
	OpenResume        func() (io.ReadSeekCloser, error)
}

// RecruitmentAbortPorts requires no numerical runtime, config or local GPU.
// Recovery uses only identities and original requests in the durable journal.
func RecruitmentAbortPorts(p *principal.Store, clock func() time.Time) recruitment.LaunchPorts {
	return recruitment.LaunchPorts{
		AbortReservation: func(ctx context.Context, c recruitment.Candidate, q nodeagent.Reserve) (nodeagent.ReservationAbort, error) {
			n, err := NewNodeClient(p, c.Endpoint, c.Capabilities.Node, clock)
			if err != nil {
				return nodeagent.ReservationAbort{}, err
			}
			return n.AbortReservation(ctx, q)
		},
		LeaseStatus: func(ctx context.Context, c recruitment.Candidate, id string) (nodeagent.Lease, error) {
			n, err := NewNodeClient(p, c.Endpoint, c.Capabilities.Node, clock)
			if err != nil {
				return nodeagent.Lease{}, err
			}
			return n.LeaseStatus(ctx, id)
		},
	}
}

// RecruitmentPorts connects the durable launch owner to authenticated node and
// authority APIs. It does not launch a child or change the selected cohort.
func RecruitmentPorts(o RecruitmentOptions) (recruitment.LaunchPorts, error) {
	ports := RecruitmentAbortPorts(o.Principal, o.Clock)
	if o.Principal == nil || o.Clock == nil {
		return ports, fmt.Errorf("controller and clock required")
	}
	if err := o.NumericalPlan.Validate(); err != nil {
		return ports, err
	}
	if (o.ResumeSource == nil) != (o.ResumeArtifact == nil) || (o.ResumeSource == nil) != (o.OpenResume == nil) {
		return ports, fmt.Errorf("complete resume source, artifact and local reader required")
	}
	if o.ResumeSource != nil {
		s, n := o.ResumeSource, o.NumericalPlan
		if s.BuildID != n.BuildID || s.ConfigHash != n.ConfigHash || s.ProgramHash != n.ProgramHash || s.WeightLayoutHash != n.WeightLayoutHash || s.OptimizerHash != n.OptimizerHash {
			return ports, fmt.Errorf("resume source differs from local numerical plan")
		}
	}
	b, err := nodejob.CanonicalConfig(o.Config)
	if err != nil {
		return ports, err
	}
	if nodejob.Hash(b) != o.NumericalPlan.ConfigHash {
		return ports, fmt.Errorf("numerical plan does not describe submitted config")
	}
	o.Config = b
	o.LoopbackPorts = append([]int(nil), o.LoopbackPorts...)
	if len(o.LoopbackPorts) < 2 || len(o.LoopbackPorts) > 64 {
		return ports, fmt.Errorf("one loopback port per selected rank required")
	}
	seen := map[int]bool{}
	for _, port := range o.LoopbackPorts {
		if port < 1024 || port > 65535 || seen[port] {
			return ports, fmt.Errorf("invalid loopback port plan")
		}
		seen[port] = true
	}
	if _, _, err := controllerTrust(o.Principal, o.Clock()); err != nil {
		return ports, err
	}
	client := func(c recruitment.Candidate) (*NodeClient, error) {
		return NewNodeClient(o.Principal, c.Endpoint, c.Capabilities.Node, o.Clock)
	}
	ports.Clock = o.Clock
	ports.Reserve = func(ctx context.Context, c recruitment.Candidate, q nodeagent.Reserve) (nodeagent.Lease, error) {
		n, err := client(c)
		if err != nil {
			return nodeagent.Lease{}, err
		}
		return n.Reserve(ctx, q)
	}
	ports.Renew = func(ctx context.Context, c recruitment.Candidate, q nodeagent.LeaseCommand) (nodeagent.Lease, error) {
		n, err := client(c)
		if err != nil {
			return nodeagent.Lease{}, err
		}
		return n.Renew(ctx, q)
	}
	ports.Prepare = func(ctx context.Context, c recruitment.Candidate, version uint64, s nodejob.Signed) (nodeagent.PreparedWorkload, error) {
		n, err := client(c)
		if err != nil {
			return nodeagent.PreparedWorkload{}, err
		}
		if o.ResumeArtifact != nil {
			f, err := o.OpenResume()
			if err != nil {
				return nodeagent.PreparedWorkload{}, err
			}
			err = n.UploadCheckpoint(ctx, s, f)
			closeErr := f.Close()
			if err != nil {
				return nodeagent.PreparedWorkload{}, err
			}
			if closeErr != nil {
				return nodeagent.PreparedWorkload{}, closeErr
			}
		}
		return n.Prepare(ctx, NodePrepareRequest{version, s})
	}
	ports.Activate = func(ctx context.Context, c recruitment.Candidate, version uint64, s grouptransport.Signed, g workload.Grant, r workload.Result) (nodeagent.Job, error) {
		n, err := client(c)
		if err != nil {
			return nodeagent.Job{}, err
		}
		return n.Transport(ctx, g.Request.Request.Scope.Job, NodeTransportRequest{version, s, g, r})
	}
	ports.Start = func(ctx context.Context, c recruitment.Candidate, q nodeagent.StartCommand) (nodeagent.Job, error) {
		n, err := client(c)
		if err != nil {
			return nodeagent.Job{}, err
		}
		return n.Start(ctx, q)
	}
	ports.Status = func(ctx context.Context, c recruitment.Candidate, id string) (nodeagent.Job, error) {
		n, err := client(c)
		if err != nil {
			return nodeagent.Job{}, err
		}
		return n.JobStatus(ctx, id)
	}
	ports.BuildJobs = o.jobs
	ports.Grant = o.grant
	ports.Issue = func(ctx context.Context, g workload.Grant) (workload.Result, error) {
		return IssueRemoteWorkload(ctx, o.Principal, o.AuthorityEndpoint, g, o.Clock)
	}
	ports.BuildTransport = o.transport
	return ports, nil
}

func controllerTrust(p *principal.Store, now time.Time) (trust.Anchor, trust.VerifiedSnapshot, error) {
	s, _, err := p.Active(now)
	if err != nil {
		return trust.Anchor{}, trust.VerifiedSnapshot{}, err
	}
	if s.Role != trust.Controller {
		return trust.Anchor{}, trust.VerifiedSnapshot{}, fmt.Errorf("controller identity required")
	}
	a, err := trust.PinRoot(s.Root, s.Fingerprint, now)
	if err != nil {
		return a, trust.VerifiedSnapshot{}, err
	}
	v, err := trust.VerifySnapshot(a, s.Snapshot, now)
	return a, v, err
}
func (o RecruitmentOptions) sign(ctx context.Context, q trust.SignRequest) (trust.SignedProof, error) {
	if err := ctx.Err(); err != nil {
		return trust.SignedProof{}, err
	}
	now := o.Clock()
	a, v, err := controllerTrust(o.Principal, now)
	if err != nil {
		return trust.SignedProof{}, err
	}
	s, key, err := o.Principal.Active(now)
	if err != nil {
		return trust.SignedProof{}, err
	}
	return trust.SignPrincipalProof(a, s.Chain, key, v, q, now)
}
func controllerRandomID() (string, error) {
	var b [16]byte
	_, err := rand.Read(b[:])
	return hex.EncodeToString(b[:]), err
}

func (o RecruitmentOptions) jobs(ctx context.Context, p recruitment.LaunchPlan, leases []nodeagent.Lease) ([]nodejob.Signed, error) {
	if len(leases) != len(p.Selection.Selected) || len(leases) != len(o.LoopbackPorts) || p.Selection.Requirements.BuildID != o.NumericalPlan.BuildID || p.Selection.Requirements.DType != o.NumericalPlan.DType {
		return nil, fmt.Errorf("cohort differs from local numerical plan")
	}
	group, err := controllerRandomID()
	if err != nil {
		return nil, err
	}
	members := make([]distributed.DDPGroupMember, len(leases))
	assigned := make([]nodejob.Member, len(leases))
	for i, l := range leases {
		member, err := controllerRandomID()
		if err != nil {
			return nil, err
		}
		members[i] = distributed.DDPGroupMember{MemberID: member, Rank: i}
		assigned[i] = nodejob.Member{Node: l.Node, MemberID: member, Rank: i}
	}
	membership, err := distributed.NewDDPGroupMembership(p.Run, group, 1, "ring", members)
	if err != nil {
		return nil, err
	}
	if o.ResumeSource != nil {
		source := o.ResumeSource
		if source.Controller != p.Controller || source.Membership.RunID != p.Run || len(source.Members) != len(leases) || source.BuildID != o.NumericalPlan.BuildID || source.ConfigHash != o.NumericalPlan.ConfigHash || source.ProgramHash != o.NumericalPlan.ProgramHash || source.WeightLayoutHash != o.NumericalPlan.WeightLayoutHash || source.OptimizerHash != o.NumericalPlan.OptimizerHash || source.DatasetID != p.Selection.Requirements.DatasetID || source.DatasetSelector != p.Selection.Requirements.DatasetSelector {
			return nil, fmt.Errorf("resume source differs from submitted numerical plan or cohort")
		}
		for i, l := range leases {
			if source.Members[i].Node != l.Node {
				return nil, fmt.Errorf("exact resume requires original ordered nodes")
			}
		}
		membership = source.Membership
		assigned = append([]nodejob.Member(nil), source.Members...)
	}
	out := make([]nodejob.Signed, len(leases))
	now := o.Clock()
	for i, l := range leases {
		job, err := controllerRandomID()
		if err != nil {
			return nil, err
		}
		nonce, err := controllerRandomID()
		if err != nil {
			return nil, err
		}
		n := o.NumericalPlan
		r := p.Selection.Requirements
		m := nodejob.Manifest{Version: nodejob.Version, Job: job, Attempt: p.Attempt, Lease: l.ID, Node: l.Node, Controller: p.Controller, Nonce: nonce, Membership: membership, Members: assigned, Rank: i, BuildID: n.BuildID, Config: o.Config, ConfigHash: n.ConfigHash, ProgramHash: n.ProgramHash, WeightLayoutHash: n.WeightLayoutHash, OptimizerHash: n.OptimizerHash, DatasetSelector: r.DatasetSelector, DatasetID: r.DatasetID, Artifacts: []nodejob.ArtifactRef{}, Mode: "arch", Transport: "tls13-ring", Limits: r.Limits, Created: now.Unix(), Expires: now.Add(15 * time.Minute).Unix()}
		m.CheckpointAt = o.CheckpointAt
		if o.ResumeArtifact != nil {
			m.Artifacts = []nodejob.ArtifactRef{*o.ResumeArtifact}
		}
		q, err := m.SigningRequest()
		if err != nil {
			return nil, err
		}
		proof, err := o.sign(ctx, q)
		if err != nil {
			return nil, err
		}
		out[i] = nodejob.Signed{Manifest: m, Proof: proof}
	}
	return out, nil
}

func (o RecruitmentOptions) grant(ctx context.Context, r workload.SignedRequest) (workload.Grant, error) {
	a, v, err := controllerTrust(o.Principal, o.Clock())
	if err != nil {
		return workload.Grant{}, err
	}
	q, err := r.Request.SigningRequest()
	if err != nil {
		return workload.Grant{}, err
	}
	if _, err := trust.VerifyProof(a, v, r.Proof, q, o.Clock()); err != nil {
		return workload.Grant{}, err
	}
	q, err = r.GrantRequest()
	if err != nil {
		return workload.Grant{}, err
	}
	s, _, err := o.Principal.Active(o.Clock())
	if err != nil {
		return workload.Grant{}, err
	}
	if r.Request.Scope.Controller != s.Principal || r.Request.Scope.Cluster != s.Cluster {
		return workload.Grant{}, fmt.Errorf("workload grant targets another controller")
	}
	proof, err := o.sign(ctx, q)
	return workload.Grant{Request: r, Proof: proof}, err
}

func (o RecruitmentOptions) transport(ctx context.Context, p recruitment.LaunchPlan, jobs []nodejob.Signed, results []workload.Result) (grouptransport.Signed, error) {
	if len(jobs) != len(results) || len(jobs) != len(p.Selection.Selected) || len(jobs) != len(o.LoopbackPorts) {
		return grouptransport.Signed{}, fmt.Errorf("complete issued cohort required")
	}
	nonce, err := controllerRandomID()
	if err != nil {
		return grouptransport.Signed{}, err
	}
	plan := grouptransport.Plan{Version: grouptransport.Version, TLSPolicy: grouptransport.TLSPolicy, Cluster: p.Cluster, Controller: p.Controller, Attempt: p.Attempt, Nonce: nonce, Membership: jobs[0].Manifest.Membership, LoopbackPorts: o.LoopbackPorts, Created: o.Clock().Unix(), Expires: results[0].Binding.ExpiresAt}
	for i, j := range jobs {
		r := results[i]
		plan.Expires = min(plan.Expires, r.Binding.ExpiresAt)
		if len(r.Chain) != 3 {
			return grouptransport.Signed{}, fmt.Errorf("issued workload certificate missing")
		}
		plan.Members = append(plan.Members, grouptransport.Member{Node: j.Manifest.Node, Job: j.Manifest.Job, MemberID: j.Manifest.Members[i].MemberID, Rank: i, ManifestHash: j.Proof.Request.Digest, Endpoint: p.Selection.Selected[i].Capabilities.TransportEndpoint, CertificateHash: nodejob.Hash(r.Chain[0]), Chain: r.Chain, Binding: r.Binding})
	}
	q, err := plan.SigningRequest()
	if err != nil {
		return grouptransport.Signed{}, err
	}
	proof, err := o.sign(ctx, q)
	return grouptransport.Signed{Plan: plan, Proof: proof}, err
}
