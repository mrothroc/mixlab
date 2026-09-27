package workerjob

import (
	"fmt"

	"github.com/mrothroc/mixlab/artifact"
)

const ArtifactChunkBytes = 64 << 10
const FinalWeightsFile = "managed-final.safetensors"
const OutputReceiptFile = "output-artifact.json"

// ArtifactMessage is a child-to-host, single-output stream. Its envelope binds
// the job/attempt. Neither kind nor any client-controlled path selects a file.
type ArtifactMessage struct {
	Phase  string       `json:"phase"`
	Ref    artifact.Ref `json:"ref"`
	Offset uint64       `json:"offset"`
	Data   []byte       `json:"data"`
}

func (m ArtifactMessage) Validate() error {
	if err := m.Ref.Validate(); err != nil {
		return err
	}
	switch m.Phase {
	case "begin":
		if m.Offset != 0 || len(m.Data) != 0 {
			return fmt.Errorf("invalid artifact begin")
		}
	case "chunk":
		if len(m.Data) == 0 || len(m.Data) > ArtifactChunkBytes || m.Offset > m.Ref.Bytes || uint64(len(m.Data)) > m.Ref.Bytes-m.Offset {
			return fmt.Errorf("invalid artifact chunk")
		}
	case "end":
		if len(m.Data) != 0 || m.Offset != m.Ref.Bytes {
			return fmt.Errorf("invalid artifact end")
		}
	default:
		return fmt.Errorf("unknown artifact phase")
	}
	return nil
}

// OutputBudget reserves room for the trainer's export and the staged verified
// copy, plus logs and small control records. It does not replace disk monitoring.
func OutputBudget(disk, logs uint64) uint64 {
	if disk <= logs || disk-logs <= 1<<20 {
		return 0
	}
	return min(artifact.MaxBytes, (disk-logs-(1<<20))/2)
}

// ManagedOutputBudget accounts for retained upload chunks, verified input and
// extracted state, plus original/packed/staged checkpoint output when enabled.
func ManagedOutputBudget(disk, logs, input uint64, checkpoint bool) uint64 {
	if input > artifact.MaxBytes || disk <= logs || disk-logs <= 3*input+(1<<20) {
		return 0
	}
	remaining := disk - logs - 3*input - (1 << 20)
	copies := uint64(2)
	if checkpoint {
		copies = 3
	}
	return min(artifact.MaxBytes, remaining/copies)
}
