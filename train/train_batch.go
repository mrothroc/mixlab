package train

import "github.com/mrothroc/mixlab/data"

// trainBatch is one prepared training batch plus runtime-only sequence packing
// and augmentation state. disableAugmentation is set for validation batches so
// eval loss is measured on un-augmented data.
type trainBatch struct {
	x, y                []int
	codebooks           []int32
	frames              []float32
	lossMask            []float32
	segmentIDs          []int32
	maskEligible        []uint8
	labels              []int32
	validMask           []float32
	exampleMask         []float32
	seqLen              int
	batchSize           int
	exampleCount        int
	disableAugmentation bool
	err                 error
}

func trainBatchFromDataBatch(batch data.Batch, err error) trainBatch {
	return trainBatch{
		x: batch.X, y: batch.Y, codebooks: batch.Codebooks, frames: batch.Frames, lossMask: batch.LossMask,
		segmentIDs: batch.SegmentIDs, maskEligible: batch.MaskEligible,
		labels: batch.Labels, validMask: batch.ValidMask, exampleMask: batch.ExampleMask,
		seqLen: batch.SeqLen, batchSize: batch.BatchSize, exampleCount: batch.ExampleCount, err: err,
	}
}

func trainBatchFromValBatch(batch data.ValBatch) trainBatch {
	return trainBatch{
		x: batch.X, y: batch.Y, codebooks: batch.Codebooks, frames: batch.Frames, lossMask: batch.LossMask,
		segmentIDs: batch.SegmentIDs, maskEligible: batch.MaskEligible,
		labels: batch.Labels, validMask: batch.ValidMask, exampleMask: batch.ExampleMask,
		seqLen: batch.SeqLen, batchSize: batch.BatchSize, exampleCount: batch.ExampleCount,
		disableAugmentation: true,
	}
}

func (b trainBatch) effectiveShape(defaultBatchSize, defaultSeqLen int) (batchSize, seqLen int) {
	batchSize, seqLen = defaultBatchSize, defaultSeqLen
	if b.batchSize > 0 {
		batchSize = b.batchSize
	}
	if b.seqLen > 0 {
		seqLen = b.seqLen
	}
	return batchSize, seqLen
}
