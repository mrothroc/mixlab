package train

import (
	"reflect"
	"testing"
)

func TestRankKeyedRNG(t *testing.T) {
	rankZero := make([]int32, 16)
	rankOne := make([]int32, 16)
	fillDistributedDropoutKeys(rankZero, 17, 9, 0, 2)
	fillDistributedDropoutKeys(rankOne, 17, 9, 1, 2)
	if reflect.DeepEqual(rankZero, rankOne) {
		t.Fatal("rank-keyed dropout produced identical keys")
	}
	microstepZero := deterministicDistributedObjectiveRNG(17, 9, 1, 0, 123)
	microstepOne := deterministicDistributedObjectiveRNG(17, 9, 1, 1, 123)
	if microstepZero.Int63() == microstepOne.Int63() {
		t.Fatal("microstep-keyed objective RNG produced identical first value")
	}
	initialRankZero := initWeightSeed(17, 4)
	initialRankOne := initWeightSeed(17, 4)
	if initialRankZero != initialRankOne {
		t.Fatal("initial parameter seed depends on rank")
	}
}

func initWeightSeed(seed int64, weightIndex int) uint64 {
	return splitMix64(uint64(seed) ^ uint64(weightIndex+1)*0xd1b54a32d192ed03)
}
