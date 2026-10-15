package workerjob

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"path/filepath"

	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/distributed"
	"github.com/mrothroc/mixlab/internal/strictjson"
	wc "github.com/mrothroc/mixlab/workercontrol"
)

const LearnerVersion = "mixlab_local_learner_assignment_v1"

// LearnerAssignment is a separate advertised schema, not a singleton DDP view.
// P0 defines it only; existing hosting continues to reject unsupported schemas.
type LearnerAssignment struct {
	Version         string                    `json:"version"`
	JobID           string                    `json:"job_id"`
	AttemptID       string                    `json:"attempt_id"`
	RunID           string                    `json:"run_id"`
	PlanHash        string                    `json:"run_plan_sha256"`
	Member          distributed.LearnerMember `json:"member"`
	Contract        run.Contract              `json:"contract"`
	Coordinator     run.Coordinator           `json:"coordinator"`
	Config          json.RawMessage           `json:"config"`
	DatasetSelector string                    `json:"dataset_selector"`
	TrainPattern    string                    `json:"local_train_pattern"`
	RuntimeSeconds  int                       `json:"runtime_seconds"`
	PrivateMaxBytes uint64                    `json:"private_max_bytes"`
}

func (a LearnerAssignment) Validate() error {
	if a.Version != LearnerVersion || a.DatasetSelector == "" || len(a.DatasetSelector) > 256 || !filepath.IsAbs(a.TrainPattern) || filepath.Clean(a.TrainPattern) != a.TrainPattern || len(a.TrainPattern) > 4096 || a.RuntimeSeconds < 1 || a.RuntimeSeconds > 7*24*3600 || a.PrivateMaxBytes == 0 || a.PrivateMaxBytes > 8<<30 {
		return fmt.Errorf("invalid bounded learner assignment")
	}
	if _, e := distributed.NewLearnerCohort(a.RunID, 0, []distributed.LearnerMember{a.Member}); e != nil {
		return e
	}
	if e := a.Contract.Validate(); e != nil {
		return e
	}
	if e := a.Coordinator.Validate(); e != nil {
		return e
	}
	digest, e := hex.DecodeString(a.PlanHash)
	if e != nil || len(digest) != 32 || hex.EncodeToString(digest) != a.PlanHash {
		return fmt.Errorf("invalid plan digest")
	}
	if len(a.Config) == 0 || len(a.Config) > 256<<10 || a.Config[0] != '{' || !json.Valid(a.Config) {
		return fmt.Errorf("bounded JSON config object required")
	}
	if e := strictjson.Validate(a.Config, 32); e != nil {
		return e
	}
	h, e := Digest(a.Config)
	if e != nil {
		return e
	}
	if h != a.Contract.ConfigHash {
		return fmt.Errorf("learner config checksum mismatch")
	}
	b := wc.Binding{JobID: a.JobID, AttemptID: a.AttemptID, BinaryID: "mixlab", BuildID: a.Contract.BuildHash, AssignmentDigest: sha256.Sum256([]byte("validate"))}
	return b.Validate()
}
func (a LearnerAssignment) Binding() (wc.Binding, error) {
	if e := a.Validate(); e != nil {
		return wc.Binding{}, e
	}
	b, e := json.Marshal(a)
	if e != nil {
		return wc.Binding{}, e
	}
	return wc.Binding{JobID: a.JobID, AttemptID: a.AttemptID, BinaryID: "mixlab", BuildID: a.Contract.BuildHash, AssignmentDigest: sha256.Sum256(b)}, nil
}
func DecodeLearner(body []byte, binding wc.Binding) (LearnerAssignment, error) {
	var a LearnerAssignment
	if e := run.Decode(body, &a); e != nil {
		return a, e
	}
	b, e := a.Binding()
	if e != nil {
		return a, e
	}
	if b != binding {
		return a, wc.ErrBinding
	}
	return a, nil
}
