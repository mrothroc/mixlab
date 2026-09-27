package workerhost

import (
	"errors"

	"golang.org/x/sys/unix"
)

// NOTE_EXIT observes death without wait(2), retaining ownership of the PID.
func observeChildExit(pid int) error {
	fd, err := unix.Kqueue()
	if err != nil {
		return err
	}
	defer func() { _ = unix.Close(fd) }()
	unix.CloseOnExec(fd)
	change := []unix.Kevent_t{{Ident: uint64(pid), Filter: unix.EVFILT_PROC, Flags: unix.EV_ADD | unix.EV_ONESHOT, Fflags: unix.NOTE_EXIT}}
	for {
		_, err = unix.Kevent(fd, change, nil, nil)
		if !errors.Is(err, unix.EINTR) {
			break
		}
	}
	if errors.Is(err, unix.ESRCH) {
		// An already-exited direct child cannot have its PID reused before our
		// sole waiter reaps it. There is no other waiter in this owner.
		return nil
	}
	if err != nil {
		return err
	}
	events := make([]unix.Kevent_t, 1)
	for {
		n, err := unix.Kevent(fd, nil, events, nil)
		if errors.Is(err, unix.EINTR) {
			continue
		}
		if err != nil {
			return err
		}
		if n == 1 && events[0].Fflags&unix.NOTE_EXIT != 0 {
			return nil
		}
	}
}

func groupHasLiveMembers(pid int) (bool, error) {
	processes, err := unix.SysctlKinfoProcSlice("kern.proc.pgrp", pid)
	if err != nil {
		return false, err
	}
	const zombie = 5 // SZOMB in Darwin sys/proc.h.
	for _, process := range processes {
		if process.Eproc.Pgid == int32(pid) && process.Proc.P_stat != zombie {
			return true, nil
		}
	}
	return false, nil
}
