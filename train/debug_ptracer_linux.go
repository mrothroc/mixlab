//go:build linux

package train

import (
	"fmt"
	"os"
	"strconv"
	"syscall"
)

// Only the explicitly requested parent supervisor (and its debugger children)
// may attach. Never grant PR_SET_PTRACER_ANY or alter the host's ptrace policy.
func allowDebugParent() error {
	value := os.Getenv("MIXLAB_DEBUG_PTRACER_PID")
	if value == "" {
		return nil
	}
	pid, err := strconv.Atoi(value)
	if err != nil || pid <= 1 || pid != os.Getppid() {
		return fmt.Errorf("MIXLAB_DEBUG_PTRACER_PID must identify the trainer's parent process")
	}
	const prSetPtracer = 0x59616d61
	_, _, errno := syscall.RawSyscall6(syscall.SYS_PRCTL, prSetPtracer, uintptr(pid), 0, 0, 0, 0)
	if errno != 0 {
		// Some hosts still deny ptrace. Keep /proc diagnostics and the watchdog
		// available instead of preventing a legitimate training job from starting.
		fmt.Fprintf(os.Stderr, "debug parent ptrace authorization unavailable: %v\n", errno)
	}
	return nil
}
