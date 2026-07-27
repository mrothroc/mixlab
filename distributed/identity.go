package distributed

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
)

const membersHashVersion = "mixlab-ddp-members-v1"

// DDPGroupMember identifies one stable member and its position in a DDP group.
type DDPGroupMember struct {
	MemberID string `json:"member_id"`
	Rank     int    `json:"rank"`
}

// DDPGroupMembership is the immutable, ordered identity shared by every rank.
type DDPGroupMembership struct {
	RunID          string           `json:"run_id"`
	GroupID        string           `json:"group_id"`
	Generation     uint64           `json:"generation"`
	Backend        string           `json:"backend"`
	OrderedMembers []DDPGroupMember `json:"ordered_members"`
	MembersHash    string           `json:"members_hash"`
}

// LocalGroupView adds process-local launch identity to shared membership.
type LocalGroupView struct {
	Membership      DDPGroupMembership `json:"membership"`
	LocalMemberID   string             `json:"local_member_id"`
	LocalRank       int                `json:"local_rank"`
	LaunchAttemptID string             `json:"launch_attempt_id"`
}

// NewDDPGroupMembership validates and copies an ordered rank assignment.
func NewDDPGroupMembership(
	runID, groupID string,
	generation uint64,
	backend string,
	orderedMembers []DDPGroupMember,
) (DDPGroupMembership, error) {
	if runID == "" {
		return DDPGroupMembership{}, fmt.Errorf("distributed run_id is required")
	}
	if groupID == "" {
		return DDPGroupMembership{}, fmt.Errorf("distributed group_id is required")
	}
	if backend == "" {
		return DDPGroupMembership{}, fmt.Errorf("distributed backend is required")
	}
	members := cloneMembers(orderedMembers)
	hash, err := NewMembersHash(members)
	if err != nil {
		return DDPGroupMembership{}, err
	}
	return DDPGroupMembership{
		RunID:          runID,
		GroupID:        groupID,
		Generation:     generation,
		Backend:        backend,
		OrderedMembers: members,
		MembersHash:    hash,
	}, nil
}

// NewMembersHash returns the canonical integrity anchor for a complete ordered
// rank assignment. Its versioned, length-prefixed encoding is a stable
// protocol surface; changes require a new membersHashVersion.
func NewMembersHash(orderedMembers []DDPGroupMember) (string, error) {
	if len(orderedMembers) == 0 {
		return "", fmt.Errorf("distributed membership must contain at least one member")
	}
	seen := make(map[string]struct{}, len(orderedMembers))
	h := sha256.New()
	writeHashString(h, membersHashVersion)
	var scratch [8]byte
	binary.BigEndian.PutUint64(scratch[:], uint64(len(orderedMembers)))
	_, _ = h.Write(scratch[:])
	for index, member := range orderedMembers {
		if member.MemberID == "" {
			return "", fmt.Errorf("distributed member %d has an empty member_id", index)
		}
		if member.Rank != index {
			return "", fmt.Errorf(
				"distributed member %q has rank %d; ordered membership requires rank %d",
				member.MemberID, member.Rank, index,
			)
		}
		if _, ok := seen[member.MemberID]; ok {
			return "", fmt.Errorf("distributed member_id %q is duplicated", member.MemberID)
		}
		seen[member.MemberID] = struct{}{}
		writeHashString(h, member.MemberID)
		binary.BigEndian.PutUint64(scratch[:], uint64(member.Rank))
		_, _ = h.Write(scratch[:])
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// NewLocalGroupView validates local identity and takes a defensive membership
// copy. Relaunches use a new launchAttemptID without changing membership.
func NewLocalGroupView(
	membership DDPGroupMembership,
	localMemberID string,
	localRank int,
	launchAttemptID string,
) (LocalGroupView, error) {
	canonical, err := membership.Canonical()
	if err != nil {
		return LocalGroupView{}, err
	}
	if launchAttemptID == "" {
		return LocalGroupView{}, fmt.Errorf("distributed launch_attempt_id is required")
	}
	if localRank < 0 || localRank >= canonical.WorldSize() {
		return LocalGroupView{}, fmt.Errorf(
			"distributed local rank %d outside [0,%d)",
			localRank, canonical.WorldSize(),
		)
	}
	member := canonical.OrderedMembers[localRank]
	if localMemberID == "" || localMemberID != member.MemberID {
		return LocalGroupView{}, fmt.Errorf(
			"distributed local member %q does not match rank %d member %q",
			localMemberID, localRank, member.MemberID,
		)
	}
	return LocalGroupView{
		Membership:      canonical,
		LocalMemberID:   localMemberID,
		LocalRank:       localRank,
		LaunchAttemptID: launchAttemptID,
	}, nil
}

// WorldSize is derived from the ordered membership and is never stored
// independently.
func (m DDPGroupMembership) WorldSize() int {
	return len(m.OrderedMembers)
}

// Canonical validates the stored hash and returns a defensive copy.
func (m DDPGroupMembership) Canonical() (DDPGroupMembership, error) {
	canonical, err := NewDDPGroupMembership(
		m.RunID,
		m.GroupID,
		m.Generation,
		m.Backend,
		m.OrderedMembers,
	)
	if err != nil {
		return DDPGroupMembership{}, err
	}
	if m.MembersHash == "" {
		return DDPGroupMembership{}, fmt.Errorf("distributed members_hash is required")
	}
	if canonical.MembersHash != m.MembersHash {
		return DDPGroupMembership{}, fmt.Errorf(
			"distributed members_hash mismatch: got %s want %s",
			m.MembersHash, canonical.MembersHash,
		)
	}
	return canonical, nil
}

func cloneMembers(members []DDPGroupMember) []DDPGroupMember {
	return append([]DDPGroupMember(nil), members...)
}

type hashWriter interface {
	Write([]byte) (int, error)
}

func writeHashString(h hashWriter, value string) {
	var size [8]byte
	binary.BigEndian.PutUint64(size[:], uint64(len(value)))
	_, _ = h.Write(size[:])
	_, _ = h.Write([]byte(value))
}
