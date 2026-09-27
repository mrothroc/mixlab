//go:build darwin || linux

package main

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestVerifiedEnrollmentRealTerminal(t *testing.T) {
	if mode := os.Getenv("MIXLAB_TEST_TERMINAL_CONFIRM"); mode != "" {
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		if mode == "cancel" {
			ctx, cancel = context.WithTimeout(ctx, 150*time.Millisecond)
			defer cancel()
		}
		ok, err := terminalConfirm(ctx, os.Stdout, "Compare exact test phrase", "test phrase")
		if mode == "cancel" {
			if ok || !errors.Is(err, context.DeadlineExceeded) {
				t.Fatalf("cancel: ok=%v err=%v", ok, err)
			}
		} else if err != nil || ok != (mode == "accept") {
			t.Fatalf("confirmation: ok=%v err=%v", ok, err)
		}
		fmt.Println("TERMINAL_CONFIRMED_" + mode)
		return
	}
	for _, mode := range []string{"accept", "reject", "cancel"} {
		t.Run(mode, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			path, err := exec.LookPath("script")
			if err != nil {
				t.Skip("real controlling-terminal test requires script")
			}
			args := []string{"-q", "/dev/null", os.Args[0], "-test.run=^TestVerifiedEnrollmentRealTerminal$", "-test.count=1"}
			if runtime.GOOS == "linux" {
				child := "'" + strings.ReplaceAll(os.Args[0], "'", "'\\''") + "' -test.run=^TestVerifiedEnrollmentRealTerminal$ -test.count=1"
				args = []string{"-q", "-e", "-c", child, "/dev/null"}
			}
			cmd := exec.CommandContext(ctx, path, args...)
			cmd.Env = append(os.Environ(), "MIXLAB_TEST_TERMINAL_CONFIRM="+mode)
			input, err := cmd.StdinPipe()
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = input.Close() }()
			output, err := cmd.StdoutPipe()
			if err != nil {
				t.Fatal(err)
			}
			cmd.Stderr = cmd.Stdout
			if err := cmd.Start(); err != nil {
				t.Fatal(err)
			}
			reader := bufio.NewReader(output)
			var prefix strings.Builder
			for !strings.HasSuffix(prefix.String(), "to reject: ") {
				b, err := reader.ReadByte()
				if err != nil {
					cancel()
					_ = cmd.Wait()
					t.Fatalf("no terminal prompt: %v: %s", err, prefix.String())
				}
				prefix.WriteByte(b)
			}
			if mode != "cancel" {
				// Delayed human input must survive an initial nonblocking read.
				timer := time.NewTimer(100 * time.Millisecond)
				<-timer.C
				answer := "test phrase\n"
				if mode == "reject" {
					answer = "different phrase\n"
				}
				_, _ = io.WriteString(input, answer)
			}
			rest, readErr := io.ReadAll(reader)
			waitErr := cmd.Wait()
			if readErr != nil || waitErr != nil || !strings.Contains(string(rest), "TERMINAL_CONFIRMED_"+mode) {
				t.Fatalf("real terminal failed: read=%v wait=%v\n%s%s", readErr, waitErr, prefix.String(), rest)
			}
		})
	}
}
