package run

import (
	"fmt"

	"github.com/mrothroc/mixlab/diloco/internal/wire"
)

// ValidationDataset is portable policy. Split is a logical catalog selector;
// no remote caller chooses a worker-local filesystem path.
type ValidationDataset struct {
	Hash           string `json:"dataset_hash"`
	Split          string `json:"split"`
	BatchTokens    uint64 `json:"batch_tokens"`
	SequenceLength uint64 `json:"seq_len"`
	Batches        uint64 `json:"batches"`
}

func (v ValidationDataset) Validate() error {
	if !wire.Hash(v.Hash) || v.Split == "" || len(v.Split) > 128 || v.Split == "." || v.Split == ".." || v.BatchTokens == 0 || v.BatchTokens > 1<<24 || v.SequenceLength == 0 || v.BatchTokens%v.SequenceLength != 0 || v.Batches == 0 || v.Batches > 65536 {
		return fmt.Errorf("invalid bounded validation policy")
	}
	for _, c := range v.Split {
		if (c < 'a' || c > 'z') && (c < 'A' || c > 'Z') && (c < '0' || c > '9') && c != '_' && c != '-' && c != '.' {
			return fmt.Errorf("invalid logical validation split")
		}
	}
	return nil
}
