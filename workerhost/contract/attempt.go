// Package contract defines the local command/outcome boundary for worker
// hosting. Values come from authorized node management or external admission,
// never directly from HTTP request bodies.
package contract

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"

	"github.com/mrothroc/mixlab/executionlimits"
	"github.com/mrothroc/mixlab/workerjob"
)

const Version = "mixlab_execution_attempt_v1"

type Limits struct {
	CPUSeconds  int64 `json:"cpu_seconds"`
	MemoryBytes int64 `json:"memory_bytes"`
	DiskBytes   int64 `json:"disk_bytes"`
	LogBytes    int64 `json:"log_bytes"`
}

func (l Limits) Validate(runtimeSeconds int) error {
	return (executionlimits.Limits{RuntimeSeconds: runtimeSeconds, CPUSeconds: l.CPUSeconds, MemoryBytes: uint64(l.MemoryBytes), DiskBytes: uint64(l.DiskBytes), LogBytes: uint64(l.LogBytes)}).Validate()
}

type Approved struct {
	Version       string               `json:"version"`
	ManifestHash  string               `json:"manifest_hash"`
	TransportHash string               `json:"transport_hash"`
	Assignment    workerjob.Assignment `json:"assignment"`
	Limits        Limits               `json:"limits"`
}

func Hash(s string) bool {
	b, err := hex.DecodeString(s)
	return err == nil && len(b) == sha256.Size && hex.EncodeToString(b) == s
}

func (a Approved) Validate() error {
	if a.Version != Version || !Hash(a.ManifestHash) || !Hash(a.TransportHash) {
		return fmt.Errorf("execution approval requires exact manifest and secure transport identities")
	}
	if err := a.Assignment.Validate(); err != nil {
		return err
	}
	if a.Assignment.OutputMaxBytes > workerjob.OutputBudget(uint64(a.Limits.DiskBytes), uint64(a.Limits.LogBytes)) {
		return fmt.Errorf("output exceeds approved disk budget")
	}
	return a.Limits.Validate(a.Assignment.RuntimeSeconds)
}

const (
	Started  = "started"
	Exited   = "exited"
	Failed   = "failed"
	Canceled = "canceled"
)

// Outcome is immutable once published. A terminal outcome is issued only
// after hosting has reaped/fenced the exact attempt and confirmed no child.
type Outcome struct {
	JobID        string `json:"job_id"`
	AttemptID    string `json:"attempt_id"`
	ManifestHash string `json:"manifest_hash"`
	ApprovalHash string `json:"approval_hash"`
	Version      uint64 `json:"version"`
	Kind         string `json:"kind"`
	PID          int    `json:"pid"`
	NoChild      bool   `json:"no_child"`
	Error        string `json:"error"`
}

func (o Outcome) Terminal() bool { return o.Kind != Started }

func (o Outcome) Validate() error {
	if o.JobID == "" || o.AttemptID == "" || !Hash(o.ManifestHash) || o.Version == 0 || o.Version > 2 || o.PID < 0 || len(o.Error) > 4096 {
		return fmt.Errorf("invalid execution outcome identity")
	}
	if o.ApprovalHash == "" {
		if o.Kind != Canceled || o.Version != 1 || o.PID != 0 || !o.NoChild {
			return fmt.Errorf("only fenced unstarted cancellation may omit execution approval")
		}
	} else if !Hash(o.ApprovalHash) {
		return fmt.Errorf("invalid execution approval hash")
	}
	switch o.Kind {
	case Started:
		if o.Version != 1 || o.PID == 0 || o.NoChild || o.Error != "" {
			return fmt.Errorf("invalid started outcome")
		}
	case Exited, Failed, Canceled:
		if !o.NoChild || (o.Kind == Exited && o.Error != "") {
			return fmt.Errorf("terminal outcome requires confirmed child cleanup")
		}
	default:
		return fmt.Errorf("invalid execution outcome kind")
	}
	return nil
}

// Fence prevents a prepared job from ever starting, even when it was canceled
// before transport activation or execution approval existed.
type Fence struct {
	JobID        string `json:"job_id"`
	AttemptID    string `json:"attempt_id"`
	ManifestHash string `json:"manifest_hash"`
}

func (f Fence) Validate() error {
	if f.JobID == "" || len(f.JobID) > 128 || f.AttemptID == "" || len(f.AttemptID) > 128 || !Hash(f.ManifestHash) {
		return fmt.Errorf("invalid preparation fence")
	}
	return nil
}
