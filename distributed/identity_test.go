package distributed

import "testing"

func TestDDPMembersHash(t *testing.T) {
	members := []DDPGroupMember{
		{MemberID: "node-a", Rank: 0},
		{MemberID: "node-b", Rank: 1},
	}
	first, err := NewDDPGroupMembership("run-1", "workers", 7, "ring", members)
	if err != nil {
		t.Fatalf("NewDDPGroupMembership: %v", err)
	}
	second, err := NewDDPGroupMembership("run-1", "workers", 7, "ring", members)
	if err != nil {
		t.Fatalf("NewDDPGroupMembership repeat: %v", err)
	}
	if first.MembersHash != second.MembersHash {
		t.Fatalf("members hash is not deterministic: %q != %q", first.MembersHash, second.MembersHash)
	}
	if got := first.WorldSize(); got != len(members) {
		t.Fatalf("WorldSize=%d want %d", got, len(members))
	}

	reordered := []DDPGroupMember{
		{MemberID: "node-b", Rank: 0},
		{MemberID: "node-a", Rank: 1},
	}
	changed, err := NewDDPGroupMembership("run-1", "workers", 8, "ring", reordered)
	if err != nil {
		t.Fatalf("reordered membership: %v", err)
	}
	if first.MembersHash == changed.MembersHash {
		t.Fatal("reordered membership retained the same members hash")
	}

	firstView, err := NewLocalGroupView(first, "node-a", 0, "launch-1")
	if err != nil {
		t.Fatalf("NewLocalGroupView first: %v", err)
	}
	secondView, err := NewLocalGroupView(first, "node-a", 0, "launch-2")
	if err != nil {
		t.Fatalf("NewLocalGroupView relaunch: %v", err)
	}
	if firstView.Membership.Generation != secondView.Membership.Generation {
		t.Fatal("relaunch changed membership generation")
	}
	if firstView.LaunchAttemptID == secondView.LaunchAttemptID {
		t.Fatal("relaunch retained launch attempt ID")
	}

	members[0].MemberID = "mutated"
	if first.OrderedMembers[0].MemberID != "node-a" {
		t.Fatal("membership retained caller slice alias")
	}
	firstView.Membership.OrderedMembers[0].MemberID = "view-mutated"
	if first.OrderedMembers[0].MemberID != "node-a" {
		t.Fatal("local view retained membership slice alias")
	}
}

func TestDDPIdentityValidation(t *testing.T) {
	tests := []struct {
		name    string
		members []DDPGroupMember
	}{
		{name: "empty"},
		{name: "empty member", members: []DDPGroupMember{{Rank: 0}}},
		{name: "rank gap", members: []DDPGroupMember{{MemberID: "a", Rank: 1}}},
		{name: "duplicate", members: []DDPGroupMember{{MemberID: "a", Rank: 0}, {MemberID: "a", Rank: 1}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := NewDDPGroupMembership("run", "group", 0, "ring", tt.members); err == nil {
				t.Fatal("expected validation error")
			}
		})
	}
}
