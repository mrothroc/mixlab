package nodejob

import (
	"strings"
	"testing"
)

func TestWorkloadScopePreservesExactJobBindings(t *testing.T) {
	m := example(t)
	s, err := m.WorkloadScope(strings.Repeat("c", 32), strings.Repeat("d", 32))
	if err != nil {
		t.Fatal(err)
	}
	q, _ := m.SigningRequest()
	deadline, _ := m.WorkloadDeadline()
	if s.ManifestHash != q.Digest || s.MembershipHash != m.Membership.MembersHash || s.Node != m.Node || s.Member != m.Members[m.Rank].MemberID || s.Rank != m.Rank || s.Lease != m.Lease || s.Job != m.Job || s.Attempt != m.Attempt || s.Deadline != deadline.Unix() || s.AdmitUntil != m.Expires || s.Created != m.Created {
		t.Fatal("workload translation changed admission", s)
	}
	m.ConfigHash = strings.Repeat("f", 64)
	if _, err := m.WorkloadScope(s.Cluster, s.Workload); err == nil {
		t.Fatal("invalid manifest translated")
	}
}
