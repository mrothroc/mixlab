//go:build darwin || linux

package workerhost

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"syscall"
	"time"

	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/workerjob"
	"github.com/mrothroc/mixlab/workerprobe"
)

// Probe invokes only the administrator-approved executable's fixed read-only
// mode. It inherits no credentials/debugger/library variables and is bounded
// even if device initialization hangs. The agent never links MLX itself.
func (s *Supervisor) Probe(parent context.Context, directory statehome.Path) (workerprobe.Report, error) {
	b, err := s.readOnlyCommand(parent, directory, "worker-probe", nil, workerprobe.MaxBytes)
	if err != nil {
		return workerprobe.Report{}, err
	}
	r, err := workerprobe.Decode(b)
	if err == nil && r.BuildID != s.build {
		err = fmt.Errorf("probe self-reported another executable")
	}
	return r, err
}

func (s *Supervisor) Plan(parent context.Context, directory statehome.Path, config []byte) (workerprobe.Plan, error) {
	if len(config) == 0 || len(config) > 256<<10 {
		return workerprobe.Plan{}, fmt.Errorf("bounded numerical config required")
	}
	b, err := s.readOnlyCommand(parent, directory, "worker-plan", config, 4096)
	if err != nil {
		return workerprobe.Plan{}, err
	}
	r, err := workerprobe.DecodePlan(b)
	if err == nil && r.BuildID != s.build {
		err = fmt.Errorf("plan self-reported another executable")
	}
	return r, err
}

func (s *Supervisor) readOnlyCommand(parent context.Context, directory statehome.Path, mode string, input []byte, limit int) ([]byte, error) {
	if directory.Kind() != statehome.Worker {
		return nil, fmt.Errorf("probe requires a private worker directory")
	}
	if err := directory.Validate(); err != nil {
		return nil, err
	}
	if hash, err := workerjob.FileDigest(s.binary); err != nil || hash != s.build {
		return nil, fmt.Errorf("approved probe binary changed")
	}
	ctx, cancel := context.WithTimeout(parent, 15*time.Second)
	defer cancel()
	cmd := exec.Command(s.binary, "-mode", mode)
	cmd.Stdin = bytes.NewReader(input)
	cmd.Dir = directory.Dir()
	cmd.Env = []string{"PATH=/usr/bin:/bin", "HOME=" + directory.Dir(), "TMPDIR=" + directory.Dir()}
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.WaitDelay = time.Second
	out, diagnostic := &probeBuffer{limit: limit, cancel: cancel}, &probeBuffer{limit: 4096, cancel: cancel}
	cmd.Stdout, cmd.Stderr = out, diagnostic
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	process := ownProcess(cmd)
	var err error
	select {
	case err = <-process.wait:
	case <-ctx.Done():
		_ = process.signal(syscall.SIGKILL)
		err = errors.Join(ctx.Err(), <-process.wait)
	}
	if err != nil {
		return nil, fmt.Errorf("approved worker %s: %w: %s", mode, err, diagnostic.buffer.String())
	}
	return out.buffer.Bytes(), ctx.Err()
}

type probeBuffer struct {
	buffer bytes.Buffer
	limit  int
	cancel context.CancelFunc
}

func (b *probeBuffer) Write(p []byte) (int, error) {
	if len(p) > b.limit-b.buffer.Len() {
		n, _ := b.buffer.Write(p[:b.limit-b.buffer.Len()])
		b.cancel()
		return n, io.ErrShortWrite
	}
	return b.buffer.Write(p)
}
