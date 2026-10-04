package workerhost

import (
	"errors"
	"fmt"
	"io/fs"
	"testing"

	"golang.org/x/sys/unix"
)

// A process listed by ReadDir can exit before its stat file is read; Linux then
// reports ESRCH rather than ENOENT. Both mean the process is gone.
func TestProcessStatGone(t *testing.T) {
	for _, tc := range []struct {
		err  error
		gone bool
	}{
		{&fs.PathError{Op: "read", Path: "/proc/1/stat", Err: unix.ENOENT}, true},
		{&fs.PathError{Op: "read", Path: "/proc/1/stat", Err: unix.ESRCH}, true},
		{fmt.Errorf("wrapped: %w", unix.ESRCH), true},
		{&fs.PathError{Op: "read", Path: "/proc/1/stat", Err: unix.EACCES}, false},
		{errors.New("other"), false},
	} {
		if got := processStatGone(tc.err); got != tc.gone {
			t.Errorf("processStatGone(%v)=%v want %v", tc.err, got, tc.gone)
		}
	}
}
