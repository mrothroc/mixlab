package gpu

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"reflect"
	"runtime"

	mixdist "github.com/mrothroc/mixlab/distributed"
)

// BootstrapGroupRuntime obtains rank/world exclusively from strict MLX init.
// memberID maps that authoritative rank to a stable launcher-owned descriptor.
// expected is nil for fresh runs, or the exact checkpoint membership on resume.
func BootstrapGroupRuntime(ctx context.Context, backend string, memberID func(int) (string, error), expected *mixdist.DDPGroupMembership) (*GroupRuntime, error) {
	if memberID == nil {
		return nil, fmt.Errorf("DDP startup requires a launcher member identity provider")
	}
	if err := RequireDistributedBackend(backend); err != nil {
		return nil, err
	}
	if ctx == nil {
		ctx = context.Background()
	}
	ctx, cancel := context.WithTimeout(ctx, defaultGroupRuntimeTimeout)
	defer cancel()
	type result struct {
		group *GroupRuntime
		err   error
	}
	ch := make(chan result)
	go func() {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		group, err := bootstrapGroup(backend, memberID, expected)
		select {
		case ch <- result{group, err}:
		case <-ctx.Done():
			if group != nil {
				group.Close()
			}
		}
	}()
	select {
	case r := <-ch:
		return r.group, r.err
	case <-ctx.Done():
		return nil, fmt.Errorf("DDP strict %s startup: %w; launcher must terminate the fixed world", backend, ctx.Err())
	}
}

func bootstrapGroup(backend string, memberID func(int) (string, error), expected *mixdist.DDPGroupMembership) (_ *GroupRuntime, err error) {
	handle := mlxGroupRuntimeCreate(backend, true)
	if handle == 0 {
		return nil, fmt.Errorf("initialize strict MLX %s group", backend)
	}
	r := &GroupRuntime{handle: handle, backend: backend, rank: mlxGroupRuntimeRank(handle), world: mlxGroupRuntimeWorldSize(handle)}
	defer func() {
		if err != nil {
			r.Close()
		}
	}()
	if r.world <= 1 || r.world > 4096 || r.rank < 0 || r.rank >= r.world {
		return nil, fmt.Errorf("mode=ddp requires an MLX world size in [2,4096], got %d rank %d", r.world, r.rank)
	}
	id, idErr := memberID(r.rank)
	if idErr != nil || len(id) == 0 || len(id) > 512 {
		id = ""
	}
	members := make([]mixdist.DDPGroupMember, r.world)
	for root := 0; root < r.world; root++ {
		words := make([]int32, 513)
		words[0] = int32(len(id))
		for i := range id {
			words[i+1] = int32(id[i])
		}
		observed, e := r.BroadcastControl(root, words)
		if e != nil {
			return nil, e
		}
		if len(observed) != len(words) || observed[0] <= 0 || observed[0] > 512 {
			return nil, fmt.Errorf("rank %d has invalid launcher member identity", root)
		}
		b := make([]byte, int(observed[0]))
		for i := range b {
			b[i] = byte(observed[i+1])
		}
		members[root] = mixdist.DDPGroupMember{MemberID: string(b), Rank: root}
	}
	nonce := make([]byte, 32)
	if _, e := rand.Read(nonce); e != nil {
		return nil, e
	}
	words := make([]int32, len(nonce))
	for i, b := range nonce {
		words[i] = int32(b)
	}
	observed, e := r.BroadcastControl(0, words)
	if e != nil {
		return nil, e
	}
	for i := range nonce {
		nonce[i] = byte(observed[i])
	}
	attempt := hex.EncodeToString(nonce)
	membership, e := mixdist.NewDDPGroupMembership(attempt, "ddp", 0, backend, members)
	if e != nil {
		return nil, e
	}
	if expected != nil {
		canonical, e := expected.Canonical()
		if e != nil {
			return nil, e
		}
		if canonical.Backend != backend || !reflect.DeepEqual(canonical.OrderedMembers, members) {
			return nil, fmt.Errorf("distributed resume topology mismatch: launcher ordered membership/backend differs from checkpoint")
		}
		membership = canonical
	}
	r.view, err = mixdist.NewLocalGroupView(membership, id, r.rank, attempt)
	if err != nil {
		return nil, err
	}
	// Every rank must have chosen the same fresh/resumed identity.
	digest := digestWords(membership.RunID, membership.GroupID, membership.Backend, membership.MembersHash)
	expectedWords := make([]uint32, 0, r.world*8)
	for _, m := range members {
		d := digestWords(m.MemberID)
		expectedWords = append(expectedWords, d[:]...)
	}
	if status := mlxGroupRuntimeValidateIdentity(handle, membership.Generation, digest, expectedWords, digestWords(id)); status != 0 {
		return nil, fmt.Errorf("DDP bootstrap identity mismatch (status=%d)", status)
	}
	runtime.SetFinalizer(r, finalizeGroupRuntime)
	return r, nil
}
