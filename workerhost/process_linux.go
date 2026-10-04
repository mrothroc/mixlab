package workerhost

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"golang.org/x/sys/unix"
)

func observeChildExit(pid int) error {
	var info unix.Siginfo
	for {
		err := unix.Waitid(unix.P_PID, pid, &info, unix.WEXITED|unix.WNOWAIT, nil)
		if !errors.Is(err, unix.EINTR) {
			return err
		}
	}
}

func groupHasLiveMembers(pid int) (bool, error) {
	entries, err := os.ReadDir("/proc")
	if err != nil {
		return false, err
	}
	for _, entry := range entries {
		if _, err := strconv.Atoi(entry.Name()); err != nil || !entry.IsDir() {
			continue
		}
		b, err := os.ReadFile(filepath.Join("/proc", entry.Name(), "stat"))
		if processStatGone(err) {
			continue
		}
		if err != nil {
			return false, err
		}
		// comm is parenthesized and can contain spaces or ')'; fields after
		// its final ')' start with state, ppid, pgrp.
		end := strings.LastIndexByte(string(b), ')')
		if end < 0 {
			return false, fmt.Errorf("malformed process stat")
		}
		fields := strings.Fields(string(b[end+1:]))
		if len(fields) < 3 {
			return false, fmt.Errorf("incomplete process stat")
		}
		group, err := strconv.Atoi(fields[2])
		if err != nil {
			return false, err
		}
		if group == pid && fields[0] != "Z" && fields[0] != "X" {
			return true, nil
		}
	}
	return false, nil
}

// processStatGone reports whether a /proc stat read failed because the process
// exited after the directory listing: ENOENT once reaped, ESRCH while exiting.
func processStatGone(err error) bool {
	return errors.Is(err, os.ErrNotExist) || errors.Is(err, unix.ESRCH)
}
