//go:build darwin || linux

package local

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/statehome"
	wc "github.com/mrothroc/mixlab/workercontrol"
)

func TestLongStatePathUsesPrivateEphemeralSocket(t *testing.T) {
	root := directory(t)
	dir, err := statehome.Resolve(statehome.Options{ExactDir: filepath.Join(root.Dir(), strings.Repeat("x", 120))}, statehome.Context{Kind: statehome.Worker})
	if err != nil {
		t.Fatal(err)
	}
	if err := dir.Ensure(); err != nil {
		t.Fatal(err)
	}
	s, f, err := Listen(dir, limits())
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = s.Close() }()
	if len(s.Path()) >= 100 || filepath.Dir(s.Path()) == dir.Dir() {
		t.Fatal("long socket path not relocated")
	}
	short := filepath.Dir(s.Path())
	info, err := os.Lstat(short)
	if err != nil || info.Mode().Perm() != 0700 {
		t.Fatal("socket context not private", err)
	}
	cmd, log := child(t, s, f, "success")
	c, err := s.Accept(bounded(t), binding(), wc.PeerIdentity{PID: cmd.Process.Pid, UID: uint32(os.Getuid())})
	if err != nil {
		t.Fatal(err)
	}
	if err := c.Send(bounded(t), message(1)); err != nil {
		t.Fatal(err)
	}
	if _, err := c.Receive(bounded(t)); err != nil {
		t.Fatal(err)
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if err := cmd.Wait(); err != nil {
		t.Fatalf("child: %v %s", err, log)
	}
	if _, err := os.Stat(short); !errors.Is(err, os.ErrNotExist) {
		t.Fatal("transient socket directory leaked", err)
	}
	if err := dir.Validate(); err != nil {
		t.Fatal("persistent attempt directory removed", err)
	}
}
