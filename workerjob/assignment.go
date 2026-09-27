// Package workerjob owns the local training-assignment contract. It contains
// no launcher, trust, enrollment, networking, or numerical implementation.
package workerjob

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"strconv"

	"github.com/mrothroc/mixlab/artifact"
	"github.com/mrothroc/mixlab/distributed"
	wc "github.com/mrothroc/mixlab/workercontrol"
)

const Version = "mixlab_local_ddp_assignment_v1"

// Fixed budgets bound the entire attempt, including heartbeats and progress.
const (
	ControlByteBudget    int64  = (2 << 30) + (128 << 20)
	ControlMessageBudget uint64 = 500000
)

// Assignment is produced after agent approval and local dataset resolution,
// never accepted directly from a remote controller. Fresh and exact-resume
// assignments use loopback ring endpoints; cross-host traffic belongs to the
// agent-owned encrypted transport. There is no raw-LAN fallback.
type Assignment struct {
	Version            string                     `json:"version"`
	JobID              string                     `json:"job_id"`
	AttemptID          string                     `json:"attempt_id"`
	BuildID            string                     `json:"build_id"` // executable SHA256
	View               distributed.LocalGroupView `json:"view"`
	Config             json.RawMessage            `json:"config"`
	DatasetSelector    string                     `json:"dataset_selector"`
	TrainPattern       string                     `json:"local_train_pattern"`
	DatasetSHA256      string                     `json:"dataset_sha256"`
	ProgramSHA256      string                     `json:"program_sha256"`
	WeightLayoutSHA256 string                     `json:"weight_layout_sha256"`
	OptimizerSHA256    string                     `json:"optimizer_sha256"`
	RingAddresses      [][]string                 `json:"ring_addresses"`
	RuntimeSeconds     int                        `json:"runtime_seconds"`
	OutputMaxBytes     uint64                     `json:"output_max_bytes,omitempty"`
	CheckpointAt       uint64                     `json:"checkpoint_at,omitempty"`
	Resume             *artifact.Ref              `json:"resume,omitempty"`
	ResumePath         string                     `json:"local_resume_path,omitempty"`
}

func (a Assignment) Validate() error {
	if a.CheckpointAt > 1<<31-1 || (a.CheckpointAt > 0 && a.OutputMaxBytes == 0) {
		return fmt.Errorf("checkpoint stop requires bounded step and output")
	}
	if a.Resume != nil {
		if err := a.Resume.Validate(); err != nil {
			return err
		}
		if !filepath.IsAbs(a.ResumePath) || filepath.Clean(a.ResumePath) != a.ResumePath {
			return fmt.Errorf("local resume artifact path required")
		}
	} else if a.ResumePath != "" {
		return fmt.Errorf("resume path requires approved artifact")
	}
	if a.OutputMaxBytes > artifact.MaxBytes {
		return fmt.Errorf("managed output exceeds artifact limit")
	}
	if a.Version != Version || a.DatasetSelector == "" || !filepath.IsAbs(a.TrainPattern) ||
		filepath.Clean(a.TrainPattern) != a.TrainPattern || a.RuntimeSeconds < 1 || a.RuntimeSeconds > 7*24*3600 {
		return fmt.Errorf("invalid local DDP assignment or runtime limit")
	}
	for _, hash := range []string{a.BuildID, a.DatasetSHA256, a.ProgramSHA256, a.WeightLayoutSHA256, a.OptimizerSHA256} {
		decoded, err := hex.DecodeString(hash)
		if err != nil || len(decoded) != sha256.Size || hex.EncodeToString(decoded) != hash {
			return fmt.Errorf("assignment requires lowercase SHA256 identities")
		}
	}
	if len(a.Config) == 0 || a.Config[0] != '{' || len(a.Config) > 256<<10 || !json.Valid(a.Config) {
		return fmt.Errorf("assignment requires a bounded JSON config object")
	}
	v, err := distributed.NewLocalGroupView(a.View.Membership, a.View.LocalMemberID, a.View.LocalRank, a.View.LaunchAttemptID)
	if err != nil {
		return err
	}
	if v.LaunchAttemptID != a.AttemptID || v.Membership.Backend != "ring" || v.Membership.WorldSize() < 2 || v.Membership.WorldSize() > 64 || len(a.RingAddresses) != v.Membership.WorldSize() {
		return fmt.Errorf("assignment requires a fixed local ring world and exact attempt binding")
	}
	seen := map[string]bool{}
	for _, addresses := range a.RingAddresses {
		if len(addresses) != 1 {
			return fmt.Errorf("local ring requires one IPv4 loopback address per rank")
		}
		host, port, err := net.SplitHostPort(addresses[0])
		p, e := strconv.Atoi(port)
		if err != nil || e != nil || host != "127.0.0.1" || p < 1024 || p > 65535 || seen[addresses[0]] {
			return fmt.Errorf("invalid or duplicate local ring endpoint")
		}
		seen[addresses[0]] = true
	}
	b := wc.Binding{JobID: a.JobID, AttemptID: a.AttemptID, BinaryID: "mixlab", BuildID: a.BuildID, AssignmentDigest: sha256.Sum256([]byte("validate"))}
	return b.Validate()
}

func (a Assignment) Binding() (wc.Binding, error) {
	if err := a.Validate(); err != nil {
		return wc.Binding{}, err
	}
	b, err := json.Marshal(a)
	if err != nil {
		return wc.Binding{}, err
	}
	if len(b) > int(wc.MaxFrameBytes)/2 {
		return wc.Binding{}, fmt.Errorf("assignment too large")
	}
	return wc.Binding{JobID: a.JobID, AttemptID: a.AttemptID, BinaryID: "mixlab", BuildID: a.BuildID, AssignmentDigest: sha256.Sum256(b)}, nil
}

// Decode requires the exact canonical encoding emitted by the owning context;
// this rejects unknown/missing fields, aliases, null scalars and duplicate keys.
func Decode(body []byte, binding wc.Binding) (Assignment, error) {
	var a Assignment
	envelope, err := Envelope(binding, 1, wc.KindAssignment, json.RawMessage(body))
	if err != nil {
		return a, err
	}
	if err := envelope.Validate(); err != nil {
		return a, err
	}
	if err := DecodePayload(body, &a); err != nil {
		return a, err
	}
	b, err := a.Binding()
	if err != nil {
		return a, err
	}
	if b != binding {
		return a, wc.ErrBinding
	}
	return a, nil
}

func DecodePayload(body []byte, dst any) error {
	if len(body) > int(wc.MaxFrameBytes) {
		return wc.ErrFrameSize
	}
	if err := json.Unmarshal(body, dst); err != nil {
		return err
	}
	canonical, err := json.Marshal(dst)
	if err != nil || !bytes.Equal(body, canonical) {
		return fmt.Errorf("noncanonical worker job payload")
	}
	return nil
}

func Digest(value any) (string, error) {
	b, err := json.Marshal(value)
	if err != nil {
		return "", err
	}
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:]), nil
}

// FileDigest is for administrator-installed immutable executables, not a
// substitute for signature approval or protection against hostile local users.
func FileDigest(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer func() { _ = f.Close() }()
	info, err := f.Stat()
	if err != nil || !info.Mode().IsRegular() {
		return "", fmt.Errorf("executable is not a regular file")
	}
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

type Event struct {
	Step      int     `json:"step"`
	Committed uint64  `json:"committed"`
	Loss      float64 `json:"loss"`
	Error     string  `json:"error"`
}

func Envelope(b wc.Binding, seq uint64, kind wc.Kind, value any) (wc.Envelope, error) {
	p, err := json.Marshal(value)
	return wc.Envelope{Version: wc.Version, JobID: b.JobID, AttemptID: b.AttemptID, Sequence: seq,
		CorrelationID: b.AttemptID, Kind: kind, PayloadKind: Version, PayloadVersion: 1, Payload: p}, err
}

func CheckPayload(e wc.Envelope, kind wc.Kind) error {
	if e.Kind != kind || e.PayloadKind != Version || e.PayloadVersion != 1 || e.CorrelationID != e.AttemptID {
		return fmt.Errorf("unexpected worker job message")
	}
	return nil
}
