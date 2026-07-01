package train

import (
	"runtime"
	"syscall"
)

func hostRSSBytes() uint64 {
	var usage syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &usage); err != nil {
		return 0
	}
	if usage.Maxrss <= 0 {
		return 0
	}
	switch runtime.GOOS {
	case "linux":
		return uint64(usage.Maxrss) * 1024
	default:
		return uint64(usage.Maxrss)
	}
}
