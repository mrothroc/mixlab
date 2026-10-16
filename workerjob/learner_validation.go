package workerjob

import (
	"fmt"
	"path/filepath"

	"github.com/mrothroc/mixlab/diloco/run"
)

// LearnerValidation is resolved only from the owning agent's data catalog.
// Split is the logical catalog selector; the coordinator never sends Pattern.
type LearnerValidation struct {
	Pattern        string `json:"local_pattern"`
	Hash           string `json:"dataset_hash"`
	Split          string `json:"split"`
	BatchTokens    uint64 `json:"batch_tokens"`
	SequenceLength uint64 `json:"seq_len"`
	Batches        uint64 `json:"batches"`
}

func (v LearnerValidation) Validate() error {
	if !filepath.IsAbs(v.Pattern) || filepath.Clean(v.Pattern) != v.Pattern || len(v.Pattern) > 4096 {
		return fmt.Errorf("invalid bounded local validation policy")
	}
	return (run.ValidationDataset{Hash: v.Hash, Split: v.Split, BatchTokens: v.BatchTokens, SequenceLength: v.SequenceLength, Batches: v.Batches}).Validate()
}

func (v LearnerValidation) Match(a run.EvaluationAssignment) error {
	if err := v.Validate(); err != nil {
		return err
	}
	if v.Hash != a.ValidationHash || v.Split != a.Split || v.BatchTokens != a.BatchTokens || v.SequenceLength != a.SequenceLength || v.Batches != a.Batches {
		return fmt.Errorf("global evaluation differs from locally approved validation policy")
	}
	return nil
}
