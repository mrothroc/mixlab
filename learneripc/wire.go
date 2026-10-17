// Package learneripc adapts learner-owned messages and bounded artifact streams
// to authenticated local IPC. It contains no TLS, credentials or numerical code.
package learneripc

import (
	"encoding/json"
	"fmt"

	private "github.com/mrothroc/mixlab/diloco/learner"
	"github.com/mrothroc/mixlab/diloco/run"
	wc "github.com/mrothroc/mixlab/workercontrol"
)

const PayloadKind = "mixlab_learner_ipc"
const ChunkBytes = 64 << 10

const receiptPendingCode = "receipt_pending"
const receiptPendingMessage = "receipt not yet available"

type request struct {
	Operation string                   `json:"operation"`
	Call      *private.RelayRequest    `json:"call,omitempty"`
	Artifact  *private.ArtifactRequest `json:"artifact,omitempty"`
	Transfer  string                   `json:"transfer,omitempty"`
	Offset    uint64                   `json:"offset,omitempty"`
	Data      []byte                   `json:"data,omitempty"`
}

func (q request) Validate() error {
	if len(q.Data) > ChunkBytes || len(q.Transfer) > 64 {
		return fmt.Errorf("IPC chunk exceeds bound")
	}
	switch q.Operation {
	case "poll":
		if q.Call != nil || q.Artifact != nil || q.Transfer != "" || q.Offset != 0 || len(q.Data) != 0 {
			return fmt.Errorf("invalid poll")
		}
	case "call":
		if q.Call == nil || q.Artifact != nil || q.Transfer != "" || q.Offset != 0 || len(q.Data) != 0 {
			return fmt.Errorf("invalid call")
		}
		return q.Call.Validate()
	case "read_begin", "write_begin":
		if q.Artifact == nil || q.Call != nil || q.Transfer != "" || q.Offset != 0 || len(q.Data) != 0 {
			return fmt.Errorf("invalid artifact begin")
		}
		a := q.Artifact
		if a.Evaluation != nil {
			if a.Assignment != (run.Assignment{}) || a.Blob != a.Evaluation.Model || q.Operation != "read_begin" {
				return fmt.Errorf("invalid evaluation artifact")
			}
			if err := a.Evaluation.Validate(); err != nil {
				return err
			}
		} else if err := a.Assignment.Validate(); err != nil {
			return err
		}
		if err := a.Blob.Validate(); err != nil {
			return err
		}
		want := "read"
		if q.Operation == "write_begin" {
			want = "create"
		}
		if a.Operation != want {
			return fmt.Errorf("artifact operation mismatch")
		}
	case "read_chunk", "write_chunk", "read_end", "write_end":
		if q.Transfer == "" || q.Call != nil || q.Artifact != nil || (q.Operation != "write_chunk" && len(q.Data) != 0) || (q.Operation == "write_chunk" && len(q.Data) == 0) {
			return fmt.Errorf("invalid transfer frame")
		}
	default:
		return fmt.Errorf("unknown learner IPC operation")
	}
	return nil
}

type response struct {
	Reply     *private.RelayResponse `json:"reply,omitempty"`
	Transfer  string                 `json:"transfer,omitempty"`
	Data      []byte                 `json:"data,omitempty"`
	Error     string                 `json:"error,omitempty"`
	ErrorCode string                 `json:"error_code,omitempty"`
}

func (r response) Validate() error {
	if len(r.Transfer) > 64 || len(r.Data) > ChunkBytes || len(r.Error) > 1024 {
		return fmt.Errorf("unbounded learner IPC response")
	}
	if r.ErrorCode != "" && r.ErrorCode != "unavailable" && r.ErrorCode != "rejected" && r.ErrorCode != receiptPendingCode {
		return fmt.Errorf("unknown learner IPC error code")
	}
	if r.ErrorCode == receiptPendingCode && r.Error != receiptPendingMessage {
		return fmt.Errorf("invalid receipt pending result")
	}
	if r.Error != "" {
		if r.Reply != nil || r.Transfer != "" || len(r.Data) != 0 {
			return fmt.Errorf("error response contains result")
		}
	} else if r.ErrorCode != "" {
		return fmt.Errorf("error code without error")
	}
	if r.Reply != nil {
		return r.Reply.Validate()
	}
	return nil
}
func envelope(kind wc.Kind, v any) (wc.Envelope, error) {
	b, err := json.Marshal(v)
	return wc.Envelope{Kind: kind, PayloadKind: PayloadKind, PayloadVersion: 1, Payload: b}, err
}
func check(e wc.Envelope, kind wc.Kind) error {
	if e.Kind != kind || e.PayloadKind != PayloadKind || e.PayloadVersion != 1 {
		return fmt.Errorf("unsupported learner IPC schema")
	}
	return nil
}
func artifactRequest(a run.Assignment, b run.Blob, op string) *private.ArtifactRequest {
	return &private.ArtifactRequest{Assignment: a, Blob: b, Operation: op}
}
