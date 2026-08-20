package data

import (
	"fmt"
	"math/rand"
)

type lengthBucketBatch struct {
	seqLen   int
	indices  []int
	sources  []int
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

type fixedLengthBucketRecord struct {
	source int
	index  int
	width  int
}

// buildFixedLengthBucketSchedule preserves full homogeneous bucket batches,
// then packs all source/bucket remainders together. Packing remainders is what
// makes an epoch exactly ceil(totalRecords/batchSize) batches instead of
// flushing one partial batch per bucket or shard.
func buildFixedLengthBucketSchedule(
	lengthsBySource [][]int32,
	buckets []int,
	batchSize int,
	rng *rand.Rand,
) ([]lengthBucketBatch, error) {
	if len(buckets) == 0 {
		return nil, fmt.Errorf("length buckets are empty")
	}
	if err := validateLengthBuckets(buckets); err != nil {
		return nil, err
	}
	if batchSize <= 0 {
		return nil, fmt.Errorf("fixed length-bucket batch size %d must be > 0", batchSize)
	}

	schedule := make([]lengthBucketBatch, 0)
	remainders := make([]fixedLengthBucketRecord, 0)
	for source, lengths := range lengthsBySource {
		remainderStart := len(remainders)
		order := make([]int, len(lengths))
		for i := range order {
			order[i] = i
		}
		if rng != nil && len(order) > 1 {
			rng.Shuffle(len(order), func(i, j int) { order[i], order[j] = order[j], order[i] })
		}
		byWidth := make(map[int][]int, len(buckets))
		for _, index := range order {
			width, ok := bucketForLength(int(lengths[index]), buckets)
			if !ok {
				return nil, fmt.Errorf(
					"source %d record %d valid length %d exceeds largest length bucket %d",
					source, index, lengths[index], buckets[len(buckets)-1],
				)
			}
			byWidth[width] = append(byWidth[width], index)
		}

		sourceBatches := make([]lengthBucketBatch, 0)
		for _, width := range buckets {
			indices := byWidth[width]
			fullEnd := len(indices) - len(indices)%batchSize
			for start := 0; start < fullEnd; start += batchSize {
				batchIndices := append([]int(nil), indices[start:start+batchSize]...)
				batchSources := make([]int, batchSize)
				for row := range batchSources {
					batchSources[row] = source
				}
				sourceBatches = append(sourceBatches, lengthBucketBatch{
					seqLen: width, indices: batchIndices, sources: batchSources, realRows: batchSize,
				})
			}
			for _, index := range indices[fullEnd:] {
				remainders = append(remainders, fixedLengthBucketRecord{source: source, index: index, width: width})
			}
		}
		if rng != nil && len(sourceBatches) > 1 && len(buckets) > 1 {
			rng.Shuffle(len(sourceBatches), func(i, j int) { sourceBatches[i], sourceBatches[j] = sourceBatches[j], sourceBatches[i] })
		}
		if rng != nil && len(remainders)-remainderStart > 1 {
			sourceRemainders := remainders[remainderStart:]
			rng.Shuffle(len(sourceRemainders), func(i, j int) {
				sourceRemainders[i], sourceRemainders[j] = sourceRemainders[j], sourceRemainders[i]
			})
		}
		schedule = append(schedule, sourceBatches...)
	}
	for start := 0; start < len(remainders); start += batchSize {
		end := start + batchSize
		if end > len(remainders) {
			end = len(remainders)
		}
		realRows := end - start
		indices := make([]int, batchSize)
		sources := make([]int, batchSize)
		width := 0
		for row, record := range remainders[start:end] {
			indices[row] = record.index
			sources[row] = record.source
			if record.width > width {
				width = record.width
			}
		}
		// Filler rows duplicate the LAST real row, not the first. Remainders are
		// appended per source, so a batch's rows are already grouped by source;
		// duplicating row 0 would send the tail back to the first source and
		// force the streams' single-slot shard cache to reload it. These rows
		// are masked out of loss and metrics either way.
		for row := realRows; row < batchSize; row++ {
			indices[row] = indices[realRows-1]
			sources[row] = sources[realRows-1]
		}
		schedule = append(schedule, lengthBucketBatch{
			seqLen: width, indices: indices, sources: sources, realRows: realRows,
		})
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
