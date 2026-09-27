//go:build darwin || linux

package workerhost

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net"
	"os"
	"syscall"
	"time"

	"github.com/mrothroc/mixlab/statehome"
	wc "github.com/mrothroc/mixlab/workercontrol"
	"github.com/mrothroc/mixlab/workerhost/contract"
)

const guardianCommand = "internal-worker-host"

type guardianBegin struct {
	Approval      contract.Approved `json:"approval"`
	Directory     string            `json:"directory"`
	Binary        string            `json:"binary"`
	GuardianBuild string            `json:"guardian_build"`
	Startup       time.Duration     `json:"startup"`
	Grace         time.Duration     `json:"grace"`
	Claim         guardianClaim     `json:"claim"`
}

type guardianEvent struct {
	PID   int    `json:"pid"`
	Error string `json:"error"`
}

func guardianFrame(q guardianBegin, sequence uint64, kind wc.Kind, value any) (wc.Envelope, error) {
	b, err := json.Marshal(value)
	if err != nil {
		return wc.Envelope{}, err
	}
	a := q.Approval.Assignment
	return wc.Envelope{Version: wc.Version, JobID: a.JobID, AttemptID: a.AttemptID,
		Sequence: sequence, CorrelationID: q.Claim.Approval, Kind: kind,
		PayloadKind: "guardian", PayloadVersion: 1, Payload: b}, nil
}

func guardianSend(c net.Conn, q guardianBegin, sequence uint64, kind wc.Kind, v any) error {
	if err := c.SetWriteDeadline(time.Now().Add(5 * time.Second)); err != nil {
		return err
	}
	e, err := guardianFrame(q, sequence, kind, v)
	if err != nil {
		return err
	}
	return wc.WriteFrame(c, e, wc.MaxFrameBytes)
}

func guardianDecode(e wc.Envelope, q guardianBegin, sequence uint64, kind wc.Kind, value any) error {
	if err := json.Unmarshal(e.Payload, value); err != nil {
		return err
	}
	want, err := guardianFrame(q, sequence, kind, value)
	if err != nil {
		return err
	}
	actual, err := json.Marshal(e)
	if err != nil {
		return err
	}
	canonical, err := json.Marshal(want)
	if err != nil || !bytes.Equal(actual, canonical) {
		return fmt.Errorf("guardian frame binding/schema mismatch")
	}
	return nil
}

func guardianSocketPair() (net.Conn, *os.File, error) {
	syscall.ForkLock.RLock()
	fds, err := syscall.Socketpair(syscall.AF_UNIX, syscall.SOCK_STREAM, 0)
	if err == nil {
		syscall.CloseOnExec(fds[0])
		syscall.CloseOnExec(fds[1])
	}
	syscall.ForkLock.RUnlock()
	if err != nil {
		return nil, nil, err
	}
	parent, child := os.NewFile(uintptr(fds[0]), "guardian-parent"), os.NewFile(uintptr(fds[1]), "guardian-child")
	c, err := net.FileConn(parent)
	_ = parent.Close()
	if err != nil {
		_ = child.Close()
		return nil, nil, err
	}
	return c, child, nil
}

func (q guardianBegin) directory() (statehome.Path, error) {
	return statehome.Resolve(statehome.Options{ExactDir: q.Directory}, statehome.Context{Kind: statehome.Worker})
}
