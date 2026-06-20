//go:build !darwin && !linux

package train

import "fmt"

func physicalMemoryBytes() (uint64, error) {
	return 0, fmt.Errorf("physical memory lookup is unsupported on this platform")
}
