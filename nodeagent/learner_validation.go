package nodeagent

import (
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/workerjob"
)

func localLearnerValidation(v run.ValidationDataset, path string) workerjob.LearnerValidation {
	return workerjob.LearnerValidation{Pattern: path, Hash: v.Hash, Split: v.Split, BatchTokens: v.BatchTokens, SequenceLength: v.SequenceLength, Batches: v.Batches}
}
