//go:build darwin || linux

package main

import (
	"context"
	"errors"
	"io"
	"testing"
	"time"

	"golang.org/x/sys/unix"
)

func TestTerminalInputReadinessAndCancellation(t *testing.T) {
	for _, mode := range []string{"read", "cancel", "closed", "hangup"} {
		t.Run(mode, func(t *testing.T) {
			fds := make([]int, 2)
			if err := unix.Pipe(fds); err != nil {
				t.Fatal(err)
			}
			defer func() { _ = unix.Close(fds[0]); _ = unix.Close(fds[1]) }()
			if err := unix.SetNonblock(fds[0], true); err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithTimeout(context.Background(), time.Second)
			defer cancel()
			switch mode {
			case "read":
				if _, err := unix.Write(fds[1], []byte("answer\n")); err != nil {
					t.Fatal(err)
				}
			case "cancel":
				ctx, cancel = context.WithTimeout(ctx, 10*time.Millisecond)
				defer cancel()
			case "closed":
				_ = unix.Close(fds[0])
				fds[0] = -1
			case "hangup":
				_ = unix.Close(fds[1])
				fds[1] = -1
			}
			b := make([]byte, 32)
			n, err := (terminalInput{ctx: ctx, fd: fds[0]}).Read(b)
			switch mode {
			case "read":
				if err != nil || string(b[:n]) != "answer\n" {
					t.Fatalf("read %q: %v", b[:n], err)
				}
			case "cancel":
				if !errors.Is(err, context.DeadlineExceeded) {
					t.Fatal("cancellation not propagated", err)
				}
			case "closed":
				if !errors.Is(err, unix.EBADF) {
					t.Fatal("closed descriptor not rejected", err)
				}
			case "hangup":
				if !errors.Is(err, io.EOF) {
					t.Fatal("hangup not reported", err)
				}
			}
		})
	}
}
