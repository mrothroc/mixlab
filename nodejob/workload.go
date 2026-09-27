package nodejob

import "github.com/mrothroc/mixlab/trust/workload"

// WorkloadScope translates validated application admission into the opaque
// trust-issuance contract. Callers still require durable prepare acceptance;
// constructing this value alone does not authorize a certificate or launch.
func (m Manifest) WorkloadScope(cluster, principal string) (workload.Scope, error) {
	q, err := m.SigningRequest()
	if err != nil {
		return workload.Scope{}, err
	}
	deadline, err := m.WorkloadDeadline()
	if err != nil {
		return workload.Scope{}, err
	}
	s := workload.Scope{Cluster: cluster, Controller: m.Controller, Node: m.Node, Lease: m.Lease, Run: m.Membership.RunID, Job: m.Job, Attempt: m.Attempt, Group: m.Membership.GroupID, Member: m.Members[m.Rank].MemberID, Workload: principal, Generation: m.Membership.Generation, Rank: m.Rank, MembershipHash: m.Membership.MembersHash, ManifestHash: q.Digest, Created: m.Created, AdmitUntil: m.Expires, Deadline: deadline.Unix()}
	return s, s.Validate()
}
