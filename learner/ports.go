// Package learner owns local round execution and durable private recovery.
// Transport authentication and process replacement are external authorities.
package learner

import (
	"context"
	"io"

	private "github.com/mrothroc/mixlab/diloco/learner"
	"github.com/mrothroc/mixlab/diloco/optimizer"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/precision"
)

// Snapshot components are engine-owned encodings, stored separately and never
// sent over the coordinator relay. Model tensors use the shared F32 codec.
type Snapshot struct {
	Model                                  []optimizer.Tensor
	Optimizer, Sampler, Schedule, Metadata []byte
	Replay                                 private.Replay
	Precision                              precision.Policy
}
type Counts struct {
	Attempted, Committed, Skipped, Tokens, Examples uint64
	Loss                                            float64
}
type Engine interface {
	Snapshot() (Snapshot, error)
	Restore(Snapshot) error
	Install([]optimizer.Tensor) error
	Train(context.Context, uint64) (Counts, error)
	Close()
}
type Peer interface {
	Read(context.Context, run.Assignment, run.Blob, io.Writer) error
	Write(context.Context, run.Assignment, run.Blob, io.Reader) error
	Start(context.Context, private.Start) error
	Submit(context.Context, private.Update) (private.Receipt, error)
}
type Factory func([]optimizer.Tensor) (Engine, error)

// ReceiptLookup recovers acceptance after the coordinator has already advanced.
// It is a query for the original semantic update, never permission to retrain.
type ReceiptLookup interface {
	Receipt(context.Context, private.Update) (private.Receipt, error)
}
