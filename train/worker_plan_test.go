package train

import (
	"bytes"
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

func TestWorkerPlanRejectsAmbiguousAndUnsupportedConfigs(t *testing.T) {
	for _, b := range []string{"{}", `{"model_dim":16,"model_dim":32}`, strings.Repeat(" ", 256<<10+1), strings.Replace(managedTinyConfig, `"backend":"ring"`, `"backend":"nccl"`, 1)} {
		if _, err := inspectWorkerConfig([]byte(b), strings.Repeat("a", 64)); err == nil {
			t.Fatal("accepted invalid plan")
		}
	}
}
