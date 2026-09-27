package clusterapp

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/mrothroc/mixlab/artifact"
	"github.com/mrothroc/mixlab/statehome"
)

// FetchOutput stages bytes privately and publishes only a fully verified file.
// Small authenticated requests retain the ordinary server deadlines and current
// trust checks. Pacing leaves room under the agent's request-rate budget.
func (c *NodeClient) FetchOutput(ctx context.Context, job string, destination statehome.Path, name string) (artifact.Ref, error) {
	var ref artifact.Ref
	if !nodeRouteID(job) {
		return ref, fmt.Errorf("canonical job ID required")
	}
	ctx, cancel := context.WithTimeout(ctx, 10*time.Minute)
	defer cancel()
	path := "/v1/agent/jobs/" + job + "/output"
	if err := c.request(ctx, http.MethodGet, path, nil, &ref); err != nil {
		return ref, err
	}
	if err := ref.Validate(); err != nil {
		return ref, err
	}
	tick := time.NewTicker(125 * time.Millisecond)
	defer tick.Stop()
	r := &outputReader{ctx: ctx, ref: ref, fetch: func(offset, size uint64) ([]byte, error) {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-tick.C:
		}
		var out NodeOutputChunk
		err := c.request(ctx, http.MethodPost, path, NodeOutputRequest{Ref: ref, Offset: offset, Bytes: size}, &out)
		if err != nil {
			return nil, err
		}
		if out.Ref != ref || out.Offset != offset || uint64(len(out.Data)) != size {
			return nil, fmt.Errorf("output chunk binding mismatch")
		}
		return out.Data, nil
	}}
	err := destination.PublishStream(name, int64(ref.Bytes), func(w io.Writer) error { return artifact.Copy(ctx, w, r, ref) })
	if errors.Is(err, statehome.ErrExists) {
		// Exact retries verify, never overwrite a corrupt or different result.
		f, openErr := destination.OpenRead(name)
		if openErr != nil {
			return ref, openErr
		}
		defer func() { _ = f.Close() }()
		err = artifact.Copy(ctx, io.Discard, f, ref)
	}
	return ref, err
}

type outputReader struct {
	ctx     context.Context
	ref     artifact.Ref
	offset  uint64
	pending []byte
	fetch   func(uint64, uint64) ([]byte, error)
}

func (r *outputReader) Read(b []byte) (int, error) {
	if err := r.ctx.Err(); err != nil {
		return 0, err
	}
	if len(b) == 0 {
		return 0, nil
	}
	if len(r.pending) == 0 {
		if r.offset == r.ref.Bytes {
			return 0, io.EOF
		}
		var err error
		r.pending, err = r.fetch(r.offset, min(outputChunkBytes, r.ref.Bytes-r.offset))
		if err != nil {
			return 0, err
		}
		if len(r.pending) == 0 {
			return 0, io.ErrNoProgress
		}
	}
	n := copy(b, r.pending)
	r.pending = r.pending[n:]
	r.offset += uint64(n)
	return n, nil
}
