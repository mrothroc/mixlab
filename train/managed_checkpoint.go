package train

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/mrothroc/mixlab/artifact"
	"github.com/mrothroc/mixlab/artifact/checkpoint"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/workerjob"
)

func writeManagedCheckpointBundle(ctx context.Context, dir, output string, limit uint64) error {
	m, err := resolveDistributedResumeManifest(dir)
	if err != nil {
		return err
	}
	var files [2]*os.File
	defer func() {
		for _, f := range files {
			if f != nil {
				_ = f.Close()
			}
		}
	}()
	var members [3]checkpoint.Member
	for i, name := range []string{m.ModelFile, m.StateFile} {
		if filepath.Base(name) != name {
			return fmt.Errorf("checkpoint member must be a basename")
		}
		f, err := os.Open(filepath.Join(dir, name))
		if err != nil {
			return err
		}
		files[i] = f
		info, err := f.Stat()
		if err != nil {
			return err
		}
		if !info.Mode().IsRegular() || info.Size() <= 0 {
			return fmt.Errorf("invalid checkpoint member")
		}
		members[i+1] = checkpoint.Member{Size: uint64(info.Size()), Reader: f}
	}
	m.ModelFile, m.StateFile = checkpoint.Model, checkpoint.State
	b, err := json.Marshal(m)
	if err != nil {
		return err
	}
	members[0] = checkpoint.Member{Size: uint64(len(b)), Reader: bytes.NewReader(b)}
	abs, err := filepath.Abs(output)
	if err != nil {
		return err
	}
	p, err := statehome.Resolve(statehome.Options{ExactDir: filepath.Dir(abs)}, statehome.Context{Kind: statehome.Worker})
	if err != nil {
		return err
	}
	return p.PublishStream(filepath.Base(abs), int64(limit), func(w io.Writer) error { return checkpoint.Write(ctx, w, members) })
}

func prepareManagedResume(ctx context.Context, a workerjob.Assignment) (string, error) {
	if a.Resume == nil {
		return "", fmt.Errorf("resume artifact required")
	}
	if err := a.Validate(); err != nil {
		return "", err
	}
	source, err := statehome.Resolve(statehome.Options{ExactDir: filepath.Dir(a.ResumePath)}, statehome.Context{Kind: statehome.Agent})
	if err != nil {
		return "", err
	}
	f, err := source.OpenRead(filepath.Base(a.ResumePath))
	if err != nil {
		return "", err
	}
	defer func() { _ = f.Close() }()
	// Verify the complete immutable input before parsing even its bounded header.
	if err := artifact.Copy(ctx, io.Discard, f, *a.Resume); err != nil {
		return "", err
	}
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return "", err
	}
	dir, err := filepath.Abs("managed-resume")
	if err != nil {
		return "", err
	}
	dst, err := statehome.Resolve(statehome.Options{ExactDir: dir}, statehome.Context{Kind: statehome.Worker})
	if err != nil {
		return "", err
	}
	err = dst.Publish(func(stage statehome.Path) error {
		h := sha256.New()
		if err := checkpoint.Read(ctx, io.TeeReader(f, h), func(name string, n uint64, r io.Reader) error {
			return stage.PublishStream(name, int64(n), func(w io.Writer) error { _, err := io.Copy(w, r); return err })
		}); err != nil {
			return err
		}
		if hex.EncodeToString(h.Sum(nil)) != a.Resume.SHA256 {
			return fmt.Errorf("checkpoint changed during extraction")
		}
		return nil
	})
	if err != nil {
		return "", err
	}
	path := filepath.Join(dir, checkpoint.Manifest)
	m, err := readDistributedResumeManifest(path)
	if err != nil {
		return "", err
	}
	if m.ModelFile != checkpoint.Model || m.StateFile != checkpoint.State {
		return "", fmt.Errorf("checkpoint references non-container members")
	}
	return path, nil
}

func managedCheckpointStop(opts TrainOptions, start, steps int) (int, error) {
	if opts.managed == nil || opts.managed.assignment.CheckpointAt == 0 {
		return 0, nil
	}
	at := opts.managed.assignment.CheckpointAt
	if at <= uint64(start) || at > uint64(steps) {
		return 0, fmt.Errorf("checkpoint stop must be after resume attempt and at/before schedule horizon")
	}
	return int(at), nil
}
