package gpu

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	goruntime "runtime"
	"sync"
	"time"

	mixdist "github.com/mrothroc/mixlab/distributed"
)

const defaultGroupRuntimeTimeout = 30 * time.Second

// GroupRuntime owns one immutable MLX distributed group. Native calls are
// serialized because MLX group initialization and collectives are not a Go
// concurrency boundary.
type GroupRuntime struct {
	mu      sync.Mutex
	handle  int64
	view    mixdist.LocalGroupView
	backend string
	rank    int
	world   int
}

type InitializationAgreementField struct {
	Name  string
	Value uint64
}

// NewGroupRuntime strictly initializes the configured MLX backend and verifies
// the expected membership before any trainer or sampler is created.
func NewGroupRuntime(ctx context.Context, view mixdist.LocalGroupView) (*GroupRuntime, error) {
	return newGroupRuntime(ctx, view, true)
}

// NewSingletonGroupRuntime initializes the same runtime boundary with MLX's
// singleton fallback. It is intended for the Phase 1 walking skeleton.
func NewSingletonGroupRuntime(ctx context.Context, view mixdist.LocalGroupView) (*GroupRuntime, error) {
	return newGroupRuntime(ctx, view, false)
}

func newGroupRuntime(
	ctx context.Context,
	view mixdist.LocalGroupView,
	strict bool,
) (*GroupRuntime, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	canonical, err := mixdist.NewLocalGroupView(
		view.Membership,
		view.LocalMemberID,
		view.LocalRank,
		view.LaunchAttemptID,
	)
	if err != nil {
		return nil, fmt.Errorf("validate distributed local view: %w", err)
	}
	if err := RequireDistributedBackend(canonical.Membership.Backend); err != nil {
		return nil, err
	}
	if _, hasDeadline := ctx.Deadline(); !hasDeadline {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, defaultGroupRuntimeTimeout)
		defer cancel()
	}

	startup, err := createAndValidateGroupRuntimeBounded(ctx, canonical, strict)
	if err != nil {
		return nil, err
	}
	groupRuntime := &GroupRuntime{
		handle:  startup.handle,
		view:    canonical,
		backend: canonical.Membership.Backend,
		rank:    startup.rank,
		world:   startup.world,
	}
	goruntime.SetFinalizer(groupRuntime, finalizeGroupRuntime)
	return groupRuntime, nil
}

// Rank is the authoritative rank reported by MLX.
func (r *GroupRuntime) Rank() int {
	if r == nil {
		return -1
	}
	return r.rank
}

// WorldSize is the authoritative world size reported by MLX.
func (r *GroupRuntime) WorldSize() int {
	if r == nil {
		return 0
	}
	return r.world
}

// Backend is the immutable backend selected at construction.
func (r *GroupRuntime) Backend() string {
	if r == nil {
		return ""
	}
	return r.backend
}

// LocalView returns a defensive copy of the validated identity.
func (r *GroupRuntime) LocalView() mixdist.LocalGroupView {
	if r == nil {
		return mixdist.LocalGroupView{}
	}
	view := r.view
	view.Membership.OrderedMembers = append(
		[]mixdist.DDPGroupMember(nil),
		r.view.Membership.OrderedMembers...,
	)
	return view
}

func (r *GroupRuntime) ValidateInitializationAgreement(
	fields []InitializationAgreementField,
) error {
	if r == nil {
		return fmt.Errorf("distributed group runtime is nil")
	}
	if len(fields) == 0 {
		return fmt.Errorf("initialization agreement requires at least one field")
	}
	words := make([]uint64, len(fields))
	for i, field := range fields {
		if field.Name == "" {
			return fmt.Errorf("initialization agreement field %d has no name", i)
		}
		words[i] = field.Value
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.handle == 0 {
		return fmt.Errorf("distributed group runtime is closed")
	}
	status, mismatchRank, mismatchWord := mlxGroupRuntimeValidateManifest(r.handle, words)
	if status < 0 {
		return fmt.Errorf("initialization agreement collective failed (status=%d)", status)
	}
	if status > 0 {
		if mismatchWord < 0 || mismatchWord >= len(fields) {
			return fmt.Errorf(
				"initialization agreement mismatch at rank %d field index %d",
				mismatchRank,
				mismatchWord,
			)
		}
		return fmt.Errorf(
			"initialization agreement mismatch at rank %d field %q",
			mismatchRank,
			fields[mismatchWord].Name,
		)
	}
	return nil
}

func (r *GroupRuntime) BroadcastControl(
	rootRank int,
	localValues []int32,
) ([]int32, error) {
	if r == nil {
		return nil, fmt.Errorf("distributed group runtime is nil")
	}
	if len(localValues) == 0 {
		return nil, fmt.Errorf("distributed control tensor must not be empty")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.handle == 0 {
		return nil, fmt.Errorf("distributed group runtime is closed")
	}
	values, status := mlxGroupRuntimeBroadcastControl(
		r.handle,
		rootRank,
		localValues,
	)
	if status != 0 {
		return nil, fmt.Errorf(
			"broadcast distributed control tensor failed (status=%d)",
			status,
		)
	}
	return values, nil
}

// Close releases the native group handle. It is idempotent.
func (r *GroupRuntime) Close() {
	if r == nil {
		return
	}
	goruntime.SetFinalizer(r, nil)
	r.close()
}

func finalizeGroupRuntime(r *GroupRuntime) {
	r.close()
}

func (r *GroupRuntime) close() {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.handle == 0 {
		return
	}
	mlxGroupRuntimeDestroy(r.handle)
	r.handle = 0
}

type groupRuntimeStartup struct {
	handle int64
	rank   int
	world  int
}

func createAndValidateGroupRuntimeBounded(
	ctx context.Context,
	view mixdist.LocalGroupView,
	strict bool,
) (groupRuntimeStartup, error) {
	membershipDigest := digestWords(
		view.Membership.RunID,
		view.Membership.GroupID,
		view.Membership.Backend,
		view.Membership.MembersHash,
	)
	expectedMembers := make([]uint32, 0, view.Membership.WorldSize()*8)
	for _, member := range view.Membership.OrderedMembers {
		words := digestWords(member.MemberID)
		expectedMembers = append(expectedMembers, words[:]...)
	}
	localDigest := digestWords(view.LocalMemberID)

	type startupResult struct {
		startup groupRuntimeStartup
		err     error
	}
	ch := make(chan startupResult, 1)
	go func() {
		goruntime.LockOSThread()
		defer goruntime.UnlockOSThread()
		handle := mlxGroupRuntimeCreate(view.Membership.Backend, strict)
		if handle == 0 {
			ch <- startupResult{err: fmt.Errorf(
				"initialize MLX distributed backend %q",
				view.Membership.Backend,
			)}
			return
		}
		startup := groupRuntimeStartup{
			handle: handle,
			rank:   mlxGroupRuntimeRank(handle),
			world:  mlxGroupRuntimeWorldSize(handle),
		}
		var startupErr error
		switch {
		case startup.rank != view.LocalRank:
			startupErr = fmt.Errorf(
				"MLX distributed rank=%d does not match expected local rank=%d",
				startup.rank,
				view.LocalRank,
			)
		case startup.world != view.Membership.WorldSize():
			startupErr = fmt.Errorf(
				"MLX distributed world_size=%d does not match expected membership size=%d",
				startup.world,
				view.Membership.WorldSize(),
			)
		default:
			status := mlxGroupRuntimeValidateIdentity(
				handle,
				view.Membership.Generation,
				membershipDigest,
				expectedMembers,
				localDigest,
			)
			if status != 0 {
				startupErr = fmt.Errorf(
					"MLX distributed startup identity mismatch (status=%d)",
					status,
				)
			}
		}
		if startupErr != nil || ctx.Err() != nil {
			if handle != 0 {
				mlxGroupRuntimeDestroy(handle)
			}
			if startupErr == nil {
				startupErr = fmt.Errorf(
					"MLX distributed startup identity exchange: %w",
					ctx.Err(),
				)
			}
			ch <- startupResult{err: startupErr}
			return
		}
		ch <- startupResult{startup: startup}
	}()
	select {
	case <-ctx.Done():
		return groupRuntimeStartup{}, fmt.Errorf(
			"initialize MLX distributed backend %q: %w",
			view.Membership.Backend,
			ctx.Err(),
		)
	case result := <-ch:
		if result.err != nil {
			return groupRuntimeStartup{}, result.err
		}
		return result.startup, nil
	}
}

func digestWords(values ...string) [8]uint32 {
	h := sha256.New()
	var length [8]byte
	for _, value := range values {
		binary.BigEndian.PutUint64(length[:], uint64(len(value)))
		_, _ = h.Write(length[:])
		_, _ = h.Write([]byte(value))
	}
	sum := h.Sum(nil)
	return [8]uint32{
		binary.BigEndian.Uint32(sum[0:4]),
		binary.BigEndian.Uint32(sum[4:8]),
		binary.BigEndian.Uint32(sum[8:12]),
		binary.BigEndian.Uint32(sum[12:16]),
		binary.BigEndian.Uint32(sum[16:20]),
		binary.BigEndian.Uint32(sum[20:24]),
		binary.BigEndian.Uint32(sum[24:28]),
		binary.BigEndian.Uint32(sum[28:32]),
	}
}
