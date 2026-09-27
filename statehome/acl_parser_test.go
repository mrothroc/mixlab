package statehome

import (
	"encoding/binary"
	"errors"
	"testing"
)

func aclBuffer(aceFlags ...uint32) []byte {
	buf := make([]byte, 12+darwinFileSecHeader+len(aceFlags)*darwinACESize)
	put := binary.LittleEndian.PutUint32
	put(buf[:4], uint32(len(buf)))
	put(buf[4:8], 8)
	put(buf[8:12], uint32(len(buf)-12))
	put(buf[12:16], 0x012cc16d)
	put(buf[48:52], uint32(len(aceFlags)))
	for i, flags := range aceFlags {
		entry := buf[12+darwinFileSecHeader+i*darwinACESize:]
		put(entry[16:20], flags)
		put(entry[20:24], 2) // read/list
	}
	return buf
}

func TestDarwinACLParser(t *testing.T) {
	for _, tc := range []struct {
		name   string
		flags  []uint32
		unsafe bool
	}{
		{name: "empty"},
		{name: "deny", flags: []uint32{2}},
		{name: "inherited-deny", flags: []uint32{2 | 0x10}},
		{name: "inherit-only-deny", flags: []uint32{2 | 0x100 | 0x60}},
		{name: "multiple-denies", flags: []uint32{2, 2}},
		{name: "permit", flags: []uint32{1}, unsafe: true},
		{name: "deny-then-permit", flags: []uint32{2, 1}, unsafe: true},
		{name: "inherited-permit", flags: []uint32{1 | 0x10}, unsafe: true},
		{name: "inherit-only-permit", flags: []uint32{1 | 0x100 | 0x60}, unsafe: true},
		{name: "unknown-kind", flags: []uint32{0}, unsafe: true},
		{name: "audit", flags: []uint32{3}, unsafe: true},
		{name: "unknown-flag", flags: []uint32{2 | 0x800}, unsafe: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := parseDarwinACL(aclBuffer(tc.flags...))
			if tc.unsafe && !errors.Is(err, ErrUnsafe) {
				t.Fatalf("expected unsafe, got %v", err)
			}
			if !tc.unsafe && err != nil {
				t.Fatal(err)
			}
		})
	}
	noACL := aclBuffer()
	binary.LittleEndian.PutUint32(noACL[48:52], 0xffffffff)
	if err := parseDarwinACL(noACL); err != nil {
		t.Fatal(err)
	}
	flags := make([]uint32, darwinACLMaxEntries)
	for i := range flags {
		flags[i] = 2
	}
	if err := parseDarwinACL(aclBuffer(flags...)); err != nil {
		t.Fatal(err)
	}
	empty := make([]byte, 12)
	binary.LittleEndian.PutUint32(empty[:4], 12)
	if err := parseDarwinACL(empty); err != nil {
		t.Fatal(err)
	}
	binary.LittleEndian.PutUint32(empty[4:8], 8)
	if err := parseDarwinACL(empty); err != nil {
		t.Fatal(err)
	}
}

func TestDarwinACLParserMalformed(t *testing.T) {
	put := binary.LittleEndian.PutUint32
	for _, tc := range []struct {
		name   string
		mutate func([]byte) []byte
	}{
		{"short-header", func(b []byte) []byte { return b[:11] }},
		{"truncated-payload", func(b []byte) []byte { return b[:len(b)-1] }},
		{"short-total", func(b []byte) []byte { put(b[:4], 4); return b }},
		{"large-total", func(b []byte) []byte { put(b[:4], 0xffffffff); return b }},
		{"negative-offset", func(b []byte) []byte { put(b[4:8], 0xfffffff8); return b }},
		{"large-offset", func(b []byte) []byte { put(b[4:8], 0x7ffffffc); return b }},
		{"unaligned-offset", func(b []byte) []byte { put(b[4:8], 9); return b }},
		{"header-overlap", func(b []byte) []byte { put(b[4:8], 0); return b }},
		{"large-size", func(b []byte) []byte { put(b[8:12], 0xffffffff); return b }},
		{"short-size", func(b []byte) []byte { put(b[8:12], 4); return b }},
		{"empty-with-payload", func(b []byte) []byte { put(b[8:12], 0); return b }},
		{"bad-magic", func(b []byte) []byte { put(b[12:16], 0); return b }},
		{"too-many-entries", func(b []byte) []byte { put(b[48:52], 129); return b }},
		{"count-size-mismatch", func(b []byte) []byte { put(b[48:52], 0); return b }},
		{"noacl-with-entry", func(b []byte) []byte { put(b[48:52], 0xffffffff); return b }},
		{"deferred-inheritance", func(b []byte) []byte { put(b[52:56], 1<<16); return b }},
		{"unknown-acl-flag", func(b []byte) []byte { put(b[52:56], 1<<18); return b }},
		{"zero-rights-permit", func(b []byte) []byte { put(b[72:76], 1); put(b[76:80], 0); return b }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := parseDarwinACL(tc.mutate(aclBuffer(2))); !errors.Is(err, ErrUnsafe) {
				t.Fatal(err)
			}
		})
	}
}

func FuzzDarwinACLParser(f *testing.F) {
	f.Add(aclBuffer())
	f.Add(aclBuffer(1))
	f.Add(aclBuffer(2))
	f.Add([]byte{})
	f.Fuzz(func(t *testing.T, data []byte) {
		if err := parseDarwinACL(data); err != nil && !errors.Is(err, ErrUnsafe) {
			t.Fatal(err)
		}
	})
}
