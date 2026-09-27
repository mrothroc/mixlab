package clusterapp

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/mrothroc/mixlab/artifact"
	artifactlocal "github.com/mrothroc/mixlab/artifact/local"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
)

// Input chunks are scoped to one signed job and its still-reserved lease.
// Paths, offsets of mutable files and arbitrary archive names are never inputs.
type NodeInputRequest struct {
	Signed nodejob.Signed `json:"signed"`
	Offset uint64         `json:"offset"`
	Data   []byte         `json:"data"`
	Commit bool           `json:"commit"`
}
type NodeInputReceipt struct {
	Job       string       `json:"job"`
	Ref       artifact.Ref `json:"ref"`
	Offset    uint64       `json:"offset"`
	Committed bool         `json:"committed"`
}

func checkpointRef(m nodejob.Manifest) (artifact.Ref, error) {
	if len(m.Artifacts) != 1 || m.Artifacts[0].Kind != "checkpoint" {
		return artifact.Ref{}, fmt.Errorf("one checkpoint input required")
	}
	a := m.Artifacts[0]
	r := artifact.Ref{SHA256: a.SHA256, Bytes: a.Bytes}
	return r, r.Validate()
}

func (a *NodeJobs) inputPath(m nodejob.Manifest) (statehome.Path, error) {
	if a.options.InputRoot.Kind() != statehome.Agent || !nodeRouteID(m.Lease) {
		return statehome.Path{}, fmt.Errorf("local input storage unavailable")
	}
	return statehome.Resolve(statehome.Options{ExactDir: filepath.Join(a.options.InputRoot.Dir(), "input-"+m.Lease)}, statehome.Context{Kind: statehome.Agent})
}

func (a *NodeJobs) Input(ctx context.Context, actor trust.AuthenticatedPrincipal, q NodeInputRequest) (out NodeInputReceipt, err error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	v, err := a.current(ctx)
	if err != nil {
		return out, err
	}
	if _, err := nodejob.Accept(a.options.Anchor, v, actor, a.options.Node, q.Signed, a.options.Clock()); err != nil {
		return out, err
	}
	m := q.Signed.Manifest
	ref, err := checkpointRef(m)
	if err != nil {
		return out, err
	}
	l, err := a.options.Store.LeaseStatus(ctx, actor, m.Lease, a.options.Clock())
	if err != nil {
		return out, err
	}
	if l.State != nodeagent.Reserved || l.Run != m.Membership.RunID || a.options.Clock().Unix() >= l.Expires {
		return out, fmt.Errorf("live reserved lease required")
	}
	caps, err := a.options.Store.Capabilities(ctx, actor, a.options.Clock())
	if err != nil {
		return out, err
	}
	if !m.Limits.Within(caps.Limits) {
		return out, fmt.Errorf("input exceeds administrator resource policy")
	}
	// Chunk staging + immutable input + extracted tensors, with room left for
	// output and logs. Retained input is part of the administrator's job budget.
	if ref.Bytes > (m.Limits.DiskBytes-m.Limits.LogBytes)/4 {
		return out, fmt.Errorf("checkpoint exceeds input disk budget")
	}
	p, err := a.inputPath(m)
	if err != nil {
		return out, err
	}
	if err := p.Ensure(); err != nil {
		return out, err
	}
	chunks, err := statehome.Resolve(statehome.Options{ExactDir: filepath.Join(p.Dir(), "chunks")}, statehome.Context{Kind: statehome.Agent})
	if err != nil {
		return out, err
	}
	if err := chunks.Ensure(); err != nil {
		return out, err
	}
	err = p.WithProcessLock(ctx, "input.lock", func() error {
		binding, err := json.Marshal(struct{ Job, Manifest string }{m.Job, q.Signed.Proof.Request.Digest})
		if err != nil {
			return err
		}
		if err := immutableInputBytes(p, "binding.json", binding); err != nil {
			return err
		}
		if q.Commit {
			if q.Offset != ref.Bytes || len(q.Data) != 0 {
				return fmt.Errorf("invalid input commit")
			}
			store, err := artifactlocal.Open(p)
			if err != nil {
				return err
			}
			if f, err := p.OpenRead(ref.SHA256); err == nil {
				defer func() { _ = f.Close() }()
				return artifact.Copy(ctx, io.Discard, f, ref)
			} else if !os.IsNotExist(err) {
				return err
			}
			r := &inputChunks{path: chunks, size: ref.Bytes}
			return store.Put(ctx, ref, r)
		}
		if q.Offset >= ref.Bytes || q.Offset%outputChunkBytes != 0 || uint64(len(q.Data)) != min(uint64(outputChunkBytes), ref.Bytes-q.Offset) {
			return fmt.Errorf("invalid checkpoint chunk")
		}
		return immutableInputBytes(chunks, fmt.Sprintf("chunk-%06d", q.Offset/outputChunkBytes), q.Data)
	})
	if err != nil {
		return out, err
	}
	return NodeInputReceipt{Job: m.Job, Ref: ref, Offset: q.Offset, Committed: q.Commit}, nil
}

func immutableInputBytes(p statehome.Path, name string, b []byte) error {
	old, err := p.ReadFileLimit(name, outputChunkBytes)
	if err == nil {
		if !bytes.Equal(old, b) {
			return fmt.Errorf("input retry differs")
		}
		return nil
	}
	if !os.IsNotExist(err) {
		return err
	}
	return p.CompareAndSwap(name, nil, b)
}

type inputChunks struct {
	path         statehome.Path
	size, offset uint64
	pending      *bytes.Reader
}

func (r *inputChunks) Read(p []byte) (int, error) {
	if len(p) == 0 {
		return 0, nil
	}
	if r.pending == nil || r.pending.Len() == 0 {
		if r.offset == r.size {
			return 0, io.EOF
		}
		b, err := r.path.ReadFileLimit(fmt.Sprintf("chunk-%06d", r.offset/outputChunkBytes), outputChunkBytes)
		if err != nil {
			return 0, err
		}
		if uint64(len(b)) != min(uint64(outputChunkBytes), r.size-r.offset) {
			return 0, fmt.Errorf("input chunk truncated")
		}
		r.offset += uint64(len(b))
		r.pending = bytes.NewReader(b)
	}
	return r.pending.Read(p)
}

func (a *NodeJobs) preparedInput(ctx context.Context, m nodejob.Manifest) (string, error) {
	ref, err := checkpointRef(m)
	if err != nil {
		return "", err
	}
	p, err := a.inputPath(m)
	if err != nil {
		return "", err
	}
	b, err := p.ReadFileLimit("binding.json", outputChunkBytes)
	if err != nil {
		return "", err
	}
	q, err := m.SigningRequest()
	if err != nil {
		return "", err
	}
	want, _ := json.Marshal(struct{ Job, Manifest string }{m.Job, q.Digest})
	if !bytes.Equal(b, want) {
		return "", fmt.Errorf("input belongs to another job")
	}
	s, err := artifactlocal.Open(p)
	if err != nil {
		return "", err
	}
	if err := s.Copy(ctx, ref, io.Discard); err != nil {
		return "", err
	}
	return filepath.Join(p.Dir(), ref.SHA256), nil
}
