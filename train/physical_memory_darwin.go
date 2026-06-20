//go:build darwin

package train

import (
	"encoding/binary"
	"fmt"
	"syscall"
)

func physicalMemoryBytes() (uint64, error) {
	raw, err := syscall.Sysctl("hw.memsize")
	if err != nil {
		return 0, err
	}
	if len(raw) == 0 || len(raw) > 8 {
		return 0, fmt.Errorf("hw.memsize sysctl returned %d bytes, want 1..8", len(raw))
	}
	buf := make([]byte, 8)
	copy(buf, raw)
	return binary.LittleEndian.Uint64(buf), nil
}
