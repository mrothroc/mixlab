package distributed

import "testing"

func TestLearnerCohortIdentity(t *testing.T) {
	members := []LearnerMember{{WorkerID: "w0", ParticipantID: "n0", IslandID: "i0", DatasetPartitionID: "p0"}, {WorkerID: "w1", ParticipantID: "n1", IslandID: "i1", DatasetPartitionID: "p1"}}
	m, e := NewLearnerCohort("run", 0, members)
	if e != nil {
		t.Fatal(e)
	}
	members[0].WorkerID = "changed"
	if m.OrderedLearners[0].WorkerID != "w0" {
		t.Fatal("aliases caller slice")
	}
	c, e := m.Canonical()
	if e != nil {
		t.Fatal(e)
	}
	c.OrderedLearners[0].WorkerID = "changed"
	if _, e = m.Canonical(); e != nil {
		t.Fatal(e)
	}
	for _, change := range []func(*LearnerCohortMembership){func(v *LearnerCohortMembership) { v.Generation++ }, func(v *LearnerCohortMembership) { v.RunID = "other" }, func(v *LearnerCohortMembership) {
		v.OrderedLearners[0], v.OrderedLearners[1] = v.OrderedLearners[1], v.OrderedLearners[0]
	}, func(v *LearnerCohortMembership) { v.OrderedLearners[0].InnerGroupHash = "fake-singleton" }, func(v *LearnerCohortMembership) { v.OrderedLearners[1].ParticipantID = "n0" }, func(v *LearnerCohortMembership) { v.OrderedLearners[1].DatasetPartitionID = "p0" }} {
		v, _ := m.Canonical()
		change(&v)
		if _, e = v.Canonical(); e == nil {
			t.Fatal("accepted identity mutation")
		}
	}
}
