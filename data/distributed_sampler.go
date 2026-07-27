package data

import (
	"fmt"
	"math/bits"
)

// DistributedSamplePartition returns one deterministic, disjoint rank view of
// a globally shuffled sample index set. A ragged tail smaller than worldSize is
// dropped so every rank executes the same number of steps.
func DistributedSamplePartition(
	total, worldSize, rank int,
	seed int64,
	epoch int,
) ([]int, error) {
	if total < 0 {
		return nil, fmt.Errorf("distributed sampler total must be non-negative")
	}
	if worldSize <= 0 {
		return nil, fmt.Errorf("distributed sampler world size must be positive")
	}
	if rank < 0 || rank >= worldSize {
		return nil, fmt.Errorf(
			"distributed sampler rank %d outside [0,%d)",
			rank,
			worldSize,
		)
	}
	if epoch < 0 {
		return nil, fmt.Errorf("distributed sampler epoch must be non-negative")
	}
	usable := (total / worldSize) * worldSize
	partition := make([]int, 0, usable/worldSize)
	key := uint64(seed) ^ uint64(epoch+1)*0x9e3779b97f4a7c15
	for ordinal := rank; ordinal < usable; ordinal += worldSize {
		partition = append(
			partition,
			int(distributedSamplePermutation(uint64(ordinal), uint64(total), key)),
		)
	}
	return partition, nil
}

// distributedSamplePermutation maps [0,total) onto itself without materializing
// the global permutation. It composes invertible operations in a power-of-two
// domain, then cycle-walks values outside the requested range.
func distributedSamplePermutation(index, total, key uint64) uint64 {
	if total <= 1 {
		return 0
	}
	width := bits.Len64(total - 1)
	mask := uint64(1)<<width - 1
	multiplier1 := (mixSamplerKey(key^0xbf58476d1ce4e5b9) | 1) & mask
	multiplier2 := (mixSamplerKey(key^0x94d049bb133111eb) | 1) & mask
	value := index
	for {
		value = (value ^ (key & mask)) & mask
		value ^= value >> 30
		value = (value * multiplier1) & mask
		value ^= value >> 27
		value = (value * multiplier2) & mask
		value ^= value >> 31
		value &= mask
		if value < total {
			return value
		}
	}
}

func mixSamplerKey(value uint64) uint64 {
	value += 0x9e3779b97f4a7c15
	value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9
	value = (value ^ (value >> 27)) * 0x94d049bb133111eb
	return value ^ (value >> 31)
}
