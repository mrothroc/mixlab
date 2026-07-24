package train

import "github.com/mrothroc/mixlab/data"

// trainBatch is one prepared training batch plus runtime-only sequence packing
// and augmentation state. disableAugmentation is set for validation batches so
// eval loss is measured on un-augmented data.
type trainBatch struct {
	x, y                []int
	lossMask            []float32
	segmentIDs          []int32
	maskEligible        []uint8
	labels              []int32
	validMask           []float32
	disableAugmentation bool
	err                 error
}

func trainBatchFromDataBatch(batch data.Batch, err error) trainBatch {
	return trainBatch{
		x: batch.X, y: batch.Y, lossMask: batch.LossMask,
		segmentIDs: batch.SegmentIDs, maskEligible: batch.MaskEligible,
		labels: batch.Labels, validMask: batch.ValidMask, err: err,
	}
}

func trainBatchFromValBatch(batch data.ValBatch) trainBatch {
	return trainBatch{
		x: batch.X, y: batch.Y, lossMask: batch.LossMask,
		segmentIDs: batch.SegmentIDs, maskEligible: batch.MaskEligible,
		labels: batch.Labels, validMask: batch.ValidMask,
		disableAugmentation: true,
	}
}
