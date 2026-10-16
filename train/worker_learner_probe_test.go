package train

import (
	"bytes"
	"reflect"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/gpu"
	"github.com/mrothroc/mixlab/workerprobe"
)

func TestWorkerLearnerProbePrecisionAndContract(t *testing.T) {
	cfg, _, _, _ := learnerFixture(t, 2)
	b := workerInitialConfigBytes(t, cfg)
	var out bytes.Buffer
	err := RunWorkerLearnerProbe(bytes.NewReader(b), &out)
	if gpu.PrecisionBackend() != "metal" {
		if err == nil || out.Len() != 0 {
			t.Fatal("probe admitted unresolved/non-Metal precision", err)
		}
		return
	}
	if err != nil {
		t.Fatal(err)
	}
	p, err := workerprobe.DecodePlan(out.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	want, err := inspectWorkerConfig(b, p.BuildID)
	if err != nil {
		t.Fatal(err)
	}
	policy, err := trainingPrecision(cfg)
	if err != nil {
		t.Fatal(err)
	}
	want.Learner.Precision = &policy
	if !reflect.DeepEqual(want, p) {
		t.Fatal("device probe altered the pure numerical contract")
	}
}

func TestWorkerLearnerProbeRejectsNonLearner(t *testing.T) {
	for _, b := range []string{managedTinyConfig, "{}", strings.Repeat(" ", 256<<10+1)} {
		var out bytes.Buffer
		if err := RunWorkerLearnerProbe(strings.NewReader(b), &out); err == nil || out.Len() != 0 {
			t.Fatal("non-learner config produced precision evidence", err)
		}
	}
}
