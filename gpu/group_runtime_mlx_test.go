//go:build mlx && cgo && (darwin || linux)

package gpu

import (
	"context"
	"os"
	"strconv"
	"testing"
	"time"

	mixdist "github.com/mrothroc/mixlab/distributed"
)

func TestGroupRuntimeStartupIdentity(t *testing.T) {
	lockMLXThread(t)
	if !Available() {
		t.Skip("MLX device unavailable")
	}
	if !DistributedBackendAvailable("ring") {
		t.Skip("MLX ring backend unavailable")
	}
	membership := testGroupMembership(t, []string{"rank-0"})
	view, err := mixdist.NewLocalGroupView(membership, "rank-0", 0, "attempt-1")
	if err != nil {
		t.Fatalf("NewLocalGroupView: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	runtime, err := NewSingletonGroupRuntime(ctx, view)
	if err != nil {
		t.Fatalf("NewSingletonGroupRuntime: %v", err)
	}
	defer runtime.Close()
	if runtime.Rank() != 0 || runtime.WorldSize() != 1 || runtime.Backend() != "ring" {
		t.Fatalf(
			"runtime identity rank=%d world=%d backend=%q",
			runtime.Rank(), runtime.WorldSize(), runtime.Backend(),
		)
	}

	wrongMembership := testGroupMembership(t, []string{"rank-0", "rank-1"})
	wrongView, err := mixdist.NewLocalGroupView(wrongMembership, "rank-0", 0, "attempt-2")
	if err != nil {
		t.Fatalf("wrong NewLocalGroupView: %v", err)
	}
	if _, err := NewSingletonGroupRuntime(ctx, wrongView); err == nil {
		t.Fatal("expected authoritative world-size mismatch")
	}

	badRankView := view
	badRankView.LocalRank = 1
	if _, err := NewSingletonGroupRuntime(ctx, badRankView); err == nil {
		t.Fatal("expected local rank mismatch")
	}
	badHashView := view
	badHashView.Membership.MembersHash = "stale-members-hash"
	if _, err := NewSingletonGroupRuntime(ctx, badHashView); err == nil {
		t.Fatal("expected members hash mismatch")
	}
	badBackend, err := mixdist.NewDDPGroupMembership(
		"runtime-test",
		"workers",
		3,
		"unsupported",
		[]mixdist.DDPGroupMember{{MemberID: "rank-0", Rank: 0}},
	)
	if err != nil {
		t.Fatalf("bad-backend membership: %v", err)
	}
	badBackendView, err := mixdist.NewLocalGroupView(
		badBackend,
		"rank-0",
		0,
		"attempt-3",
	)
	if err != nil {
		t.Fatalf("bad-backend local view: %v", err)
	}
	if _, err := NewSingletonGroupRuntime(ctx, badBackendView); err == nil {
		t.Fatal("expected backend mismatch")
	}
}

func TestGroupRuntimeStartupIdentityTwoRankGenerationMismatch(t *testing.T) {
	lockMLXThread(t)
	if os.Getenv("MIXLAB_DDP_TWO_RANK") != "1" {
		t.Skip("set MIXLAB_DDP_TWO_RANK=1 and launch with mlx.launch ring hostfile")
	}
	if !Available() {
		t.Skip("MLX device unavailable")
	}
	rank, err := strconv.Atoi(os.Getenv("MLX_RANK"))
	if err != nil || rank < 0 || rank > 1 {
		t.Fatalf("MLX_RANK=%q is not a two-rank launcher rank", os.Getenv("MLX_RANK"))
	}
	members := []mixdist.DDPGroupMember{
		{MemberID: "rank-0", Rank: 0},
		{MemberID: "rank-1", Rank: 1},
	}
	membership, err := mixdist.NewDDPGroupMembership(
		"runtime-test",
		"workers",
		uint64(3+rank),
		"ring",
		members,
	)
	if err != nil {
		t.Fatalf("NewDDPGroupMembership: %v", err)
	}
	view, err := mixdist.NewLocalGroupView(
		membership,
		members[rank].MemberID,
		rank,
		"mismatch-attempt",
	)
	if err != nil {
		t.Fatalf("NewLocalGroupView: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if runtime, err := NewGroupRuntime(ctx, view); err == nil {
		runtime.Close()
		t.Fatal("expected cross-rank generation mismatch")
	}
}

func TestGroupRuntimeStartupIdentityTwoRankHostfile(t *testing.T) {
	lockMLXThread(t)
	if os.Getenv("MIXLAB_DDP_TWO_RANK") != "1" {
		t.Skip("set MIXLAB_DDP_TWO_RANK=1 and launch with mlx.launch ring hostfile")
	}
	if !Available() {
		t.Skip("MLX device unavailable")
	}
	rank, err := strconv.Atoi(os.Getenv("MLX_RANK"))
	if err != nil || rank < 0 || rank > 1 {
		t.Fatalf("MLX_RANK=%q is not a two-rank launcher rank", os.Getenv("MLX_RANK"))
	}
	membership := testGroupMembership(t, []string{"rank-0", "rank-1"})
	memberID := membership.OrderedMembers[rank].MemberID
	view, err := mixdist.NewLocalGroupView(membership, memberID, rank, "hostfile-attempt")
	if err != nil {
		t.Fatalf("NewLocalGroupView: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	runtime, err := NewGroupRuntime(ctx, view)
	if err != nil {
		t.Fatalf("NewGroupRuntime: %v", err)
	}
	defer runtime.Close()
	if runtime.WorldSize() != 2 || runtime.Rank() != rank {
		t.Fatalf("runtime rank/world=%d/%d want %d/2", runtime.Rank(), runtime.WorldSize(), rank)
	}
	if os.Getenv("MLX_WORLD_SIZE") != "" {
		t.Fatalf("ring hostfile test must not require synthetic MLX_WORLD_SIZE, got %q", os.Getenv("MLX_WORLD_SIZE"))
	}
}

func testGroupMembership(t *testing.T, memberIDs []string) mixdist.DDPGroupMembership {
	t.Helper()
	members := make([]mixdist.DDPGroupMember, len(memberIDs))
	for rank, memberID := range memberIDs {
		members[rank] = mixdist.DDPGroupMember{MemberID: memberID, Rank: rank}
	}
	membership, err := mixdist.NewDDPGroupMembership(
		"runtime-test",
		"workers",
		3,
		"ring",
		members,
	)
	if err != nil {
		t.Fatalf("NewDDPGroupMembership: %v", err)
	}
	return membership
}
