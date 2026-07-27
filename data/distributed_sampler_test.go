package data

import "testing"

func TestRankDisjointSampler(t *testing.T) {
	const total = 23
	const world = 4
	seen := make(map[int]int)
	for rank := 0; rank < world; rank++ {
		partition, err := DistributedSamplePartition(total, world, rank, 17, 3)
		if err != nil {
			t.Fatalf("rank %d: %v", rank, err)
		}
		again, err := DistributedSamplePartition(total, world, rank, 17, 3)
		if err != nil {
			t.Fatalf("rank %d replay: %v", rank, err)
		}
		if len(partition) != len(again) {
			t.Fatalf("rank %d replay length mismatch", rank)
		}
		for i, index := range partition {
			if again[i] != index {
				t.Fatalf("rank %d replay mismatch at %d", rank, i)
			}
			seen[index]++
		}
	}
	if len(seen) != (total/world)*world {
		t.Fatalf("covered %d samples, want %d", len(seen), (total/world)*world)
	}
	for index, count := range seen {
		if count != 1 {
			t.Fatalf("sample %d assigned %d times", index, count)
		}
	}
}

func TestDistributedSamplePermutationCoversArbitraryDomains(t *testing.T) {
	for _, total := range []int{1, 2, 3, 7, 8, 9, 23, 64, 65} {
		seen := make(map[uint64]bool, total)
		for index := 0; index < total; index++ {
			value := distributedSamplePermutation(
				uint64(index),
				uint64(total),
				0x123456789abcdef0,
			)
			if value >= uint64(total) {
				t.Fatalf("total=%d index=%d produced out-of-range %d", total, index, value)
			}
			if seen[value] {
				t.Fatalf("total=%d produced duplicate %d", total, value)
			}
			seen[value] = true
		}
		if len(seen) != total {
			t.Fatalf("total=%d covered %d values", total, len(seen))
		}
	}
}

func TestDistributedSamplePartitionChangesAcrossEpochs(t *testing.T) {
	first, err := DistributedSamplePartition(101, 3, 1, 17, 0)
	if err != nil {
		t.Fatalf("first epoch: %v", err)
	}
	second, err := DistributedSamplePartition(101, 3, 1, 17, 1)
	if err != nil {
		t.Fatalf("second epoch: %v", err)
	}
	equal := len(first) == len(second)
	for i := range first {
		if first[i] != second[i] {
			equal = false
			break
		}
	}
	if equal {
		t.Fatal("different epochs produced identical partitions")
	}
}
