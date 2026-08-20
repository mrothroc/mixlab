package data

import (
	"fmt"
	"math/rand"
)

type lengthBucketBatch struct {
	seqLen   int
	indices  []int
	realRows int
}

func validateLengthBuckets(buckets []int) error {
	previous := 0
	for i, width := range buckets {
		if width <= previous {
			return fmt.Errorf("length bucket %d=%d must be strictly greater than %d", i, width, previous)
		}
		previous = width
	}
	return nil
}

func bucketForLength(length int, buckets []int) (int, bool) {
	for _, width := range buckets {
		if length <= width {
			return width, true
		}
	}
	return 0, false
}

// buildLengthBucketSchedule partitions an already-shuffled record order. The
// relative order inside each bucket is therefore random without a second RNG
// pass. Batch visits are shuffled when multiple buckets are active so training
// does not drain the corpus in ascending length order.
func buildLengthBucketSchedule(order []int, lengths []int32, buckets []int, tokenBudget int, rng *rand.Rand) ([]lengthBucketBatch, error) {
	if len(buckets) == 0 {
		return nil, fmt.Errorf("length buckets are empty")
	}
	if err := validateLengthBuckets(buckets); err != nil {
		return nil, err
	}
	if tokenBudget < buckets[len(buckets)-1] {
		return nil, fmt.Errorf("batch token budget %d is smaller than largest length bucket %d", tokenBudget, buckets[len(buckets)-1])
	}
	byWidth := make(map[int][]int, len(buckets))
	for _, index := range order {
		if index < 0 || index >= len(lengths) {
			return nil, fmt.Errorf("record index %d outside length table [0,%d)", index, len(lengths))
		}
		width, ok := bucketForLength(int(lengths[index]), buckets)
		if !ok {
			return nil, fmt.Errorf("record %d valid length %d exceeds largest length bucket %d", index, lengths[index], buckets[len(buckets)-1])
		}
		byWidth[width] = append(byWidth[width], index)
	}
	schedule := make([]lengthBucketBatch, 0)
	for _, width := range buckets {
		indices := byWidth[width]
		rows := tokenBudget / width
		for start := 0; start < len(indices); start += rows {
			end := start + rows
			if end > len(indices) {
				end = len(indices)
			}
			realRows := end - start
			batchIndices := make([]int, rows)
			copy(batchIndices, indices[start:end])
			for row := realRows; row < rows; row++ {
				batchIndices[row] = batchIndices[0]
			}
			schedule = append(schedule, lengthBucketBatch{seqLen: width, indices: batchIndices, realRows: realRows})
		}
	}
	if len(buckets) > 1 && len(schedule) > 1 && rng != nil {
		rng.Shuffle(len(schedule), func(i, j int) { schedule[i], schedule[j] = schedule[j], schedule[i] })
	}
	return schedule, nil
}

func validateRecordLengthsForBuckets(source string, lengths []int32, buckets []int) error {
	if len(buckets) == 0 {
		return nil
	}
	largest := buckets[len(buckets)-1]
	for record, length := range lengths {
		if int(length) > largest {
			return fmt.Errorf("dataset record %s[%d] valid length %d exceeds largest training.length_buckets value %d", source, record, length, largest)
		}
	}
	return nil
}
