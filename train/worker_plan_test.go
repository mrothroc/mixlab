package train

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/workerprobe"
)

func TestWorkerPlanMatchesManagedAssignment(t *testing.T) {
	a := managedFixture(t)
	p, err := inspectWorkerConfig(a.Config, a.BuildID)
	if err != nil || p.ProgramHash != a.ProgramSHA256 || p.WeightLayoutHash != a.WeightLayoutSHA256 || p.OptimizerHash != a.OptimizerSHA256 {
		t.Fatal(p, err)
	}
	var out bytes.Buffer
	if err := RunWorkerPlan(bytes.NewReader(a.Config), &out); err != nil {
		t.Fatal(err)
	}
	r, err := workerprobe.DecodePlan(out.Bytes())
	if err != nil || r.ProgramHash != p.ProgramHash || r.OptimizerHash != p.OptimizerHash {
		t.Fatal(r, err)
	}
}

func TestWorkerPlanLearnerPureNumericalContract(t *testing.T) {
	cfg, _, _, _ := learnerFixture(t, 2)
	b, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	p, err := inspectWorkerConfig(b, strings.Repeat("a", 64))
	if err != nil {
		t.Fatal(err)
	}
	if p.Learner == nil || p.Learner.Precision != nil {
		t.Fatal("pure plan resolved a device or omitted learner policy")
	}
	d := cfg.Training.Distributed
	if p.Learner.Outer.InnerSteps != uint64(d.InnerSteps) || p.Learner.Outer.Accumulation != uint64(d.GradientAccumulationSteps) || p.Learner.BatchTokens != cfg.Training.BatchTokens || p.Learner.SequenceLength != cfg.SeqLen {
		t.Fatal("learner recipe changed", p.Learner)
	}
	encoded, err := json.Marshal(p)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := workerprobe.DecodePlan(encoded); err != nil {
		t.Fatal(err)
	}
}

func TestWorkerPlanRejectsAmbiguousAndUnsupportedConfigs(t *testing.T) {
	for _, b := range []string{"{}", `{"model_dim":16,"model_dim":32}`, strings.Repeat(" ", 256<<10+1), strings.Replace(managedTinyConfig, `"backend":"ring"`, `"backend":"nccl"`, 1)} {
		if _, err := inspectWorkerConfig([]byte(b), strings.Repeat("a", 64)); err == nil {
			t.Fatal("accepted invalid plan")
		}
	}
}
