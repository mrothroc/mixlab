package workerhost

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"path/filepath"

	"github.com/mrothroc/mixlab/artifact"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/workerjob"
)

// ExistingAttempt is read-only. Reads never allocate or repair runtime evidence.
func (s *RuntimeStore) ExistingAttempt(ctx context.Context, job, attempt string) (statehome.Path, error) {
	var zero statehome.Path
	if err := ctx.Err(); err != nil {
		return zero, err
	}
	if !runtimeID(job) || !runtimeID(attempt) {
		return zero, fmt.Errorf("canonical job and attempt required")
	}
	_, index, err := s.load()
	if err != nil {
		return zero, err
	}
	for _, entry := range index.Entries {
		if entry.Job != job || entry.Attempt != attempt || !entry.Ready {
			continue
		}
		p, err := statehome.Resolve(statehome.Options{ExactDir: filepath.Join(s.root.Dir(), job+"-"+attempt)}, statehome.Context{Kind: statehome.Worker})
		if err != nil {
			return zero, err
		}
		b, err := p.ReadFileLimit("runtime-identity", 128)
		if err != nil {
			return zero, err
		}
		if string(b) != job+"/"+attempt {
			return zero, fmt.Errorf("runtime identity changed")
		}
		return p, ctx.Err()
	}
	return zero, fmt.Errorf("published runtime attempt required")
}

// Output opens only the immutable receipt owned by this physical attempt.
// The application must authorize the job owner and successful cleanup first.
func (s *RuntimeStore) Output(ctx context.Context, job, attempt string) (statehome.Path, artifact.Ref, error) {
	p, err := s.ExistingAttempt(ctx, job, attempt)
	if err != nil {
		return p, artifact.Ref{}, err
	}
	b, err := p.ReadFileLimit(workerjob.OutputReceiptFile, 1024)
	if err != nil {
		return p, artifact.Ref{}, err
	}
	var ref artifact.Ref
	if err := json.Unmarshal(b, &ref); err != nil {
		return p, ref, err
	}
	again, _ := json.Marshal(ref)
	if !bytes.Equal(b, again) {
		return p, ref, fmt.Errorf("invalid output receipt")
	}
	return p, ref, ref.Validate()
}
