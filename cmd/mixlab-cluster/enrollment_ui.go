package main

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/trust/enrollment"
	"github.com/mrothroc/mixlab/trust/identity"
	"golang.org/x/sys/unix"
)

// Human decisions come from the controlling terminal, not piped stdin, an
// environment variable, or a network approval route. Nonblocking reads keep
// cancellation bounded without closing a descriptor concurrently with a read.
func terminalConfirm(ctx context.Context, w io.Writer, prompt, want string) (bool, error) {
	fd, err := unix.Open("/dev/tty", unix.O_RDWR|unix.O_NONBLOCK|unix.O_CLOEXEC, 0)
	if err != nil {
		return false, fmt.Errorf("verified enrollment requires a controlling terminal: %w", err)
	}
	defer func() { _ = unix.Close(fd) }()
	if _, err := fmt.Fprintf(w, "%s\nType %q to confirm, or anything else to reject: ", prompt, want); err != nil {
		return false, err
	}
	s := bufio.NewScanner(terminalInput{ctx: ctx, fd: fd})
	s.Buffer(make([]byte, 256), 4096)
	if !s.Scan() {
		if ctx.Err() != nil {
			return false, ctx.Err()
		}
		if s.Err() != nil {
			return false, s.Err()
		}
		return false, io.EOF
	}
	if err := ctx.Err(); err != nil {
		return false, err
	}
	return strings.TrimSpace(s.Text()) == want, nil
}

// Darwin /dev/tty does not support Go's file poller or poll(2) reliably. Retry
// nonblocking reads with a cancellable wait instead of exposing EAGAIN to
// Scanner. One owner retains the descriptor until the read has returned.
type terminalInput struct {
	ctx context.Context
	fd  int
}

func (r terminalInput) Read(p []byte) (int, error) {
	if len(p) == 0 {
		return 0, nil
	}
	for {
		if err := r.ctx.Err(); err != nil {
			return 0, err
		}
		n, err := unix.Read(r.fd, p)
		if errors.Is(err, unix.EINTR) {
			continue
		}
		if errors.Is(err, unix.EAGAIN) || errors.Is(err, unix.EWOULDBLOCK) {
			timer := time.NewTimer(25 * time.Millisecond)
			select {
			case <-r.ctx.Done():
				timer.Stop()
				return 0, r.ctx.Err()
			case <-timer.C:
			}
			continue
		}
		if err != nil {
			return 0, err
		}
		if n == 0 {
			return 0, io.EOF
		}
		return n, err
	}
}

func terminalEnrollmentUI(w io.Writer) clusterapp.EnrollmentUI {
	return clusterapp.EnrollmentUI{
		AcceptRoot: func(ctx context.Context, p identity.Presentation) (bool, error) {
			return terminalConfirm(ctx, w, fmt.Sprintf("Compare this cluster identity with the source operator over an independent channel.\nFingerprint: %s\nPhrase (%s): %s", p.Fingerprint, p.WordlistVersion, p.Phrase), p.Phrase)
		},
		ConfirmRequest: func(ctx context.Context, p enrollment.PendingApproval) (bool, error) {
			return terminalConfirm(ctx, w, requestPresentation(p), p.Phrase)
		},
	}
}

func requestPresentation(p enrollment.PendingApproval) string {
	warning := ""
	if p.Role != "node" {
		warning = "WARNING: this request grants a privileged controller/coordinator identity, not a worker node.\n"
	}
	return warning + fmt.Sprintf("Compare the exact request phrase with the other operator.\nRequest: %s\nPurpose: %s; role: %s\nRequest fingerprint: %s\nObserved peer/interface: %q / %q\nPhrase: %s", p.ID, p.Purpose, p.Role, p.RequestHash, p.Peer, p.Interface, p.Phrase)
}
