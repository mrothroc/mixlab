package statehome

import (
	"encoding/binary"
	"fmt"
)

// Layout and limits from Darwin sys/attr.h and sys/kauth.h. The supported
// Darwin architectures are little-endian; all lengths are checked before use.
const (
	darwinACLMaxEntries = 128
	darwinFileSecHeader = 44
	darwinACESize       = 24
	darwinACLBufferSize = 12 + darwinFileSecHeader + darwinACLMaxEntries*darwinACESize
)

func parseDarwinACL(buf []byte) error {
	bad := func(reason string) error { return fmt.Errorf("%w: Darwin ACL %s", ErrUnsafe, reason) }
	if len(buf) < 12 {
		return bad("truncated attribute header")
	}
	u32 := binary.LittleEndian.Uint32
	n := int64(u32(buf[:4]))
	if n < 12 || n > int64(len(buf)) || n > darwinACLBufferSize {
		return bad("invalid attribute length")
	}
	offset := int64(int32(u32(buf[4:8])))
	size := int64(u32(buf[8:12]))
	// Darwin may return an empty attribute when there is no extended security.
	if size == 0 {
		if n == 12 && (offset == 0 || offset == 8) {
			return nil
		}
		return bad("invalid empty attribute")
	}
	start := 4 + offset
	if start < 12 || start%4 != 0 || size < darwinFileSecHeader || start+size != n {
		return bad("invalid security reference")
	}
	sec := buf[start:n]
	if u32(sec[:4]) != 0x012cc16d {
		return bad("invalid filesec magic")
	}
	count := u32(sec[36:40])
	flags := u32(sec[40:44])
	// Private filesystem flags and NO_INHERIT are harmless. Deferred inheritance
	// or unknown flags could alter access on rename, so fail closed.
	if flags & ^uint32(0xffff|1<<17) != 0 {
		return bad("unsupported ACL flags")
	}
	if count == 0xffffffff {
		count = 0
	} // KAUTH_FILESEC_NOACL
	if count > darwinACLMaxEntries || size != darwinFileSecHeader+int64(count)*darwinACESize {
		return bad("invalid ACE count or size")
	}
	for i := uint32(0); i < count; i++ {
		entry := sec[darwinFileSecHeader+int(i)*darwinACESize:]
		aceFlags := u32(entry[16:20])
		// Only DENY plus documented inheritance flags is accepted. Reject even
		// zero-right and inherit-only PERMIT entries; never try to resolve users.
		if aceFlags&0xf != 2 || aceFlags & ^uint32(0xf|0x1f0) != 0 {
			return bad("contains granting or unsupported ACE")
		}
	}
	return nil
}
