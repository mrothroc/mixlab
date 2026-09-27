package workerprobe

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"fmt"
)

const PlanVersion = "mixlab_worker_plan_v1"

// Plan is a pure numerical contract for an immutable config, not admission to
// a dataset, a GPU, a lease or a training job.
type Plan struct {
	Version          string `json:"version"`
	BuildID          string `json:"build_id"`
	ConfigHash       string `json:"config_hash"`
	ProgramHash      string `json:"program_hash"`
	WeightLayoutHash string `json:"weight_layout_hash"`
	OptimizerHash    string `json:"optimizer_hash"`
	DType            string `json:"dtype"`
}

func (p Plan) Validate() error {
	if p.Version != PlanVersion || (p.DType != "fp32" && p.DType != "bf16") {
		return fmt.Errorf("invalid numerical plan profile")
	}
	for _, s := range []string{p.BuildID, p.ConfigHash, p.ProgramHash, p.WeightLayoutHash, p.OptimizerHash} {
		b, err := hex.DecodeString(s)
		if err != nil || len(b) != 32 || hex.EncodeToString(b) != s {
			return fmt.Errorf("numerical plan requires canonical SHA256 identities")
		}
	}
	return nil
}
func DecodePlan(b []byte) (Plan, error) {
	var p Plan
	if len(b) > 4096 {
		return p, fmt.Errorf("numerical plan too large")
	}
	if err := json.Unmarshal(b, &p); err != nil {
		return p, err
	}
	canonical, _ := json.Marshal(p)
	if !bytes.Equal(bytes.TrimSuffix(b, []byte("\n")), canonical) {
		return p, fmt.Errorf("noncanonical numerical plan")
	}
	return p, p.Validate()
}
