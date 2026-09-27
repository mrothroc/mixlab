package clusterapp

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/mrothroc/mixlab/artifact"
	"github.com/mrothroc/mixlab/nodejob"
)

func (c *NodeClient) UploadCheckpoint(ctx context.Context, signed nodejob.Signed, source io.ReadSeeker) error {
	ref, err := checkpointRef(signed.Manifest)
	if err != nil {
		return err
	}
	if signed.Manifest.Node != c.node {
		return fmt.Errorf("input node binding mismatch")
	}
	if err := artifact.Copy(ctx, io.Discard, source, ref); err != nil {
		return err
	}
	if _, err := source.Seek(0, io.SeekStart); err != nil {
		return err
	}
	tick := time.NewTicker(125 * time.Millisecond)
	defer tick.Stop()
	send := func(offset uint64, b []byte, commit bool) error {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-tick.C:
		}
		var out NodeInputReceipt
		q := NodeInputRequest{Signed: signed, Offset: offset, Data: b, Commit: commit}
		if err := c.request(ctx, http.MethodPut, "/v1/agent/jobs/"+signed.Manifest.Job+"/input", q, &out); err != nil {
			return err
		}
		if out.Job != signed.Manifest.Job || out.Ref != ref || out.Offset != offset || out.Committed != commit {
			return fmt.Errorf("input receipt binding mismatch")
		}
		return nil
	}
	b := make([]byte, outputChunkBytes)
	for offset := uint64(0); offset < ref.Bytes; {
		chunk := b[:min(uint64(len(b)), ref.Bytes-offset)]
		if _, err := io.ReadFull(source, chunk); err != nil {
			return err
		}
		if err := send(offset, chunk, false); err != nil {
			return err
		}
		offset += uint64(len(chunk))
	}
	return send(ref.Bytes, nil, true)
}
