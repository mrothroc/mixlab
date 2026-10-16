package train

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/data"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/distributed"
	"github.com/mrothroc/mixlab/workerjob"
)

func TestPrepareLocalLearnerContract(t *testing.T) {
	cfg, path, _, _ := learnerFixture(t, 2)
	config, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	p, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatal(err)
	}
	spec, err := buildTrainerOptimizerSpec(cfg, shapes)
	if err != nil {
		t.Fatal(err)
	}
	spec.ComputeDType, err = gpuComputeDTypeForTraining(cfg)
	if err != nil {
		t.Fatal(err)
	}
	c := run.Contract{BuildHash: strings.Repeat("a", 64)}
	c.ConfigHash, err = workerjob.Digest(json.RawMessage(config))
	if err != nil {
		t.Fatal(err)
	}
	c.ProgramHash, err = workerjob.Digest(p)
	if err != nil {
		t.Fatal(err)
	}
	c.WeightLayoutHash, err = workerjob.Digest(shapes)
	if err != nil {
		t.Fatal(err)
	}
	c.InnerOptimizerHash, err = optimizerSpecHash(spec)
	if err != nil {
		t.Fatal(err)
	}
	c.DatasetHash, err = data.DistributedDatasetIdentity(path)
	if err != nil {
		t.Fatal(err)
	}
	cohort, err := distributed.NewLearnerCohort("run", 0, []distributed.LearnerMember{{WorkerID: "w0", ParticipantID: "n0", IslandID: "i0", DatasetPartitionID: "p0"}, {WorkerID: "w1", ParticipantID: "n1", IslandID: "i1", DatasetPartitionID: "p1"}})
	if err != nil {
		t.Fatal(err)
	}
	a := workerjob.LearnerAssignment{Version: workerjob.LearnerVersion, JobID: "job", AttemptID: "attempt", RunID: "run", PlanHash: strings.Repeat("b", 64), Member: cohort.OrderedLearners[1], Cohort: &cohort, Contract: c, Coordinator: run.Coordinator{ClusterID: "cluster", PrincipalID: "coordinator", Role: "coordinator", Audience: "coordinator"}, Config: config, DatasetSelector: "train", TrainPattern: path, RuntimeSeconds: 100, PrivateMaxBytes: 1 << 20}
	_, id, err := PrepareLocalLearner(a)
	if err != nil {
		t.Fatal(err)
	}
	if id.Index != 1 || id.Count != 2 || id.Seed != cfg.Training.Seed {
		t.Fatal("wrong stable partition", id)
	}
	for _, field := range []string{"dataset", "program", "layout", "optimizer", "cohort"} {
		t.Run(field, func(t *testing.T) {
			bad := a
			switch field {
			case "dataset":
				bad.Contract.DatasetHash = strings.Repeat("f", 64)
			case "program":
				bad.Contract.ProgramHash = strings.Repeat("f", 64)
			case "layout":
				bad.Contract.WeightLayoutHash = strings.Repeat("f", 64)
			case "optimizer":
				bad.Contract.InnerOptimizerHash = strings.Repeat("f", 64)
			case "cohort":
				bad.Cohort = nil
			}
			if _, _, err := PrepareLocalLearner(bad); err == nil {
				t.Fatal("bad contract accepted")
			}
		})
	}
}
