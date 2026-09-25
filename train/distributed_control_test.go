package train

import (
	"errors"
	"math"
	"testing"

	"github.com/mrothroc/mixlab/distributed"
)

type ddpControlFixture struct {
	rank   int
	values [][]int32
	err    error
}

func (f ddpControlFixture) Rank() int      { return f.rank }
func (f ddpControlFixture) WorldSize() int { return 2 }
func (f ddpControlFixture) LocalView() distributed.LocalGroupView {
	return distributed.LocalGroupView{}
}
func (f ddpControlFixture) BroadcastControl(root int, values []int32) ([]int32, error) {
	if f.err != nil {
		return nil, f.err
	}
	if f.values != nil {
		return f.values[root], nil
	}
	return values, nil
}

func TestDistributedControlDecisions(t *testing.T) {
	for _, tc := range []struct{ request, platform, want string }{{"auto", "darwin", "ring"}, {"", "linux", "nccl"}, {"ring", "darwin", "ring"}, {"nccl", "darwin", ""}, {"ring", "linux", ""}, {"auto", "windows", ""}} {
		got, err := distributedBackend(tc.request, tc.platform)
		if got != tc.want || (err != nil) != (tc.want == "") {
			t.Fatalf("%+v got %q %v", tc, got, err)
		}
	}
	stop, err := distributedRootDecision(ddpControlFixture{}, true, nil)
	if err != nil || !stop {
		t.Fatal(stop, err)
	}
	stop, err = distributedRootDecision(ddpControlFixture{rank: 1, values: [][]int32{{1, 0}}}, false, nil)
	if err != nil || !stop {
		t.Fatal("follower ignored root stop")
	}
	if _, err = distributedRootDecision(ddpControlFixture{rank: 1, values: [][]int32{{0, 1}}}, false, nil); err == nil {
		t.Fatal("follower ignored root failure")
	}
	want := errors.New("disk failure")
	if _, err = distributedRootDecision(ddpControlFixture{}, false, want); !errors.Is(err, want) {
		t.Fatal(err)
	}
	if _, err = distributedRootDecision(ddpControlFixture{values: [][]int32{{}}}, false, nil); err == nil {
		t.Fatal("accepted malformed control")
	}
	words := func(x float64) []int32 {
		b := math.Float64bits(x)
		return []int32{int32(uint32(b)), int32(uint32(b >> 32))}
	}
	mean, err := distributedMeanLoss(ddpControlFixture{values: [][]int32{words(3), words(7)}}, 3)
	if err != nil || mean != 5 {
		t.Fatal(mean, err)
	}
	if _, err = distributedMeanLoss(ddpControlFixture{err: want}, 3); !errors.Is(err, want) {
		t.Fatal(err)
	}
}
