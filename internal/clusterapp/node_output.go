package clusterapp

import (
	"context"
	"fmt"
	"io"

	"github.com/mrothroc/mixlab/artifact"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/trust"
)

const outputChunkBytes = 1 << 20

type NodeOutputRequest struct {
	Ref    artifact.Ref `json:"ref"`
	Offset uint64       `json:"offset"`
	Bytes  uint64       `json:"bytes"`
}
type NodeOutputChunk struct {
	Ref    artifact.Ref `json:"ref"`
	Offset uint64       `json:"offset"`
	Data   []byte       `json:"data"`
}

type nodeOutputOperations interface {
	Output(context.Context, trust.AuthenticatedPrincipal, string, *NodeOutputRequest) (artifact.Ref, []byte, error)
}

// Output keeps authorization at the job boundary: a hash alone grants nothing.
func (a *NodeJobs) Output(ctx context.Context, actor trust.AuthenticatedPrincipal, id string, q *NodeOutputRequest) (artifact.Ref, []byte, error) {
	var zero artifact.Ref
	if _, err := a.current(ctx); err != nil {
		return zero, nil, err
	}
	j, err := a.options.Store.JobStatus(actor, id, a.options.Clock())
	if err != nil {
		return zero, nil, err
	}
	if j.State != nodeagent.JobExited || j.CancelRequested || a.options.Output == nil {
		return zero, nil, fmt.Errorf("successful job output required")
	}
	l, err := a.options.Store.LeaseStatus(ctx, actor, j.Lease, a.options.Clock())
	if err != nil {
		return zero, nil, err
	}
	if l.State != nodeagent.Released {
		return zero, nil, fmt.Errorf("physical cleanup required before output delivery")
	}
	p, ref, err := a.options.Output(ctx, j.ID, j.Attempt)
	if err != nil {
		return zero, nil, err
	}
	if err := ref.Validate(); err != nil {
		return zero, nil, err
	}
	if q == nil {
		return ref, nil, nil
	}
	if q.Ref != ref || q.Bytes == 0 || q.Bytes > outputChunkBytes || q.Offset >= ref.Bytes || q.Bytes > ref.Bytes-q.Offset {
		return zero, nil, fmt.Errorf("invalid output range or identity")
	}
	f, err := p.OpenRead(ref.SHA256)
	if err != nil {
		return zero, nil, err
	}
	defer func() { _ = f.Close() }()
	info, err := f.Stat()
	if err != nil {
		return zero, nil, err
	}
	if uint64(info.Size()) != ref.Bytes {
		return zero, nil, fmt.Errorf("output size changed")
	}
	b := make([]byte, int(q.Bytes))
	if _, err := io.ReadFull(io.NewSectionReader(f, int64(q.Offset), int64(q.Bytes)), b); err != nil {
		return zero, nil, err
	}
	return ref, b, ctx.Err()
}
