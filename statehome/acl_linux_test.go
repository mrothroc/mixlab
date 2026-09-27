package statehome

import (
	"encoding/binary"
	"errors"
	"os"
	"path/filepath"
	"syscall"
	"testing"
)

func linuxACLFixture() []byte {
	// Linux posix_acl_xattr_header (version 2) and five 8-byte entries:
	// owner rwx, named user r, owning group none, mask r, other none.
	buf := make([]byte, 4+5*8)
	binary.LittleEndian.PutUint32(buf[:4], 2)
	for i, entry := range []struct {
		tag, perm uint16
		id        uint32
	}{
		{1, 7, 0xffffffff},
		{2, 4, uint32(os.Geteuid() + 1)},
		{4, 0, 0xffffffff},
		{16, 4, 0xffffffff},
		{32, 0, 0xffffffff},
	} {
		b := buf[4+i*8:]
		binary.LittleEndian.PutUint16(b[:2], entry.tag)
		binary.LittleEndian.PutUint16(b[2:4], entry.perm)
		binary.LittleEndian.PutUint32(b[4:8], entry.id)
	}
	return buf
}

func TestLinuxInheritedACLMask(t *testing.T) {
	base := privateTemp(t)
	acl := linuxACLFixture()
	if err := syscall.Setxattr(base, "system.posix_acl_default", acl, 0); err != nil {
		if errors.Is(err, syscall.ENOTSUP) {
			t.Skip("filesystem does not support POSIX ACLs")
		}
		t.Fatal(err)
	}
	p := resolved(t, Options{ExactDir: filepath.Join(base, "final")}, Context{Kind: Worker})
	if err := p.Publish(func(stage Path) error {
		return stage.WriteFile("key", []byte("private"))
	}); err != nil {
		t.Fatal(err)
	}
	// The default ACL really propagated, but restrictive creation modes masked
	// all named-user/group grants on each directory and protected temporary.
	if n, err := syscall.Getxattr(p.Dir(), "system.posix_acl_default", nil); err != nil || n == 0 {
		t.Fatalf("missing inherited default ACL: %d %v", n, err)
	}
	path := filepath.Join(p.Dir(), "key")
	if n, err := syscall.Getxattr(path, "system.posix_acl_access", nil); err != nil || n == 0 {
		t.Fatalf("missing inherited access ACL: %d %v", n, err)
	}
	if err := p.Validate(); err != nil {
		t.Fatal(err)
	}
	if _, err := p.ReadFile("key"); err != nil {
		t.Fatal(err)
	}
	if err := p.WriteFile("key", []byte("replacement")); err != nil {
		t.Fatal(err)
	}
	// Enabling the named-user grant expands the group-class mode mask and must
	// immediately make the same otherwise-protected file unsafe.
	binary.LittleEndian.PutUint16(acl[6:8], 6) // retain file owner rw, not rwx
	if err := syscall.Setxattr(path, "system.posix_acl_access", acl, 0); err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm() != 0640 {
		t.Fatalf("ACL did not change only the mask: %o", info.Mode().Perm())
	}
	if _, err := p.ReadFile("key"); !errors.Is(err, ErrUnsafe) {
		t.Fatalf("ACL mask accepted: %v", err)
	}
	if err := p.WriteFile("key", nil); !errors.Is(err, ErrUnsafe) {
		t.Fatalf("ACL mask replaced: %v", err)
	}
}
